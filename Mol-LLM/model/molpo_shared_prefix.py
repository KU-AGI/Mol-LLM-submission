from dataclasses import dataclass
import torch

@dataclass
class SharedPrefixPlan:
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    kv_index: torch.Tensor
    max_q: int
    max_k: int

    def validate(self, n_tokens: int) -> None:
        if self.cu_q.numel() != self.cu_k.numel():
            raise ValueError(f'segment count differs: q {self.cu_q.numel()} vs k {self.cu_k.numel()}')
        if int(self.cu_q[0]) != 0 or int(self.cu_k[0]) != 0:
            raise ValueError('cumulative lengths must start at 0')
        if int(self.cu_q[-1]) != n_tokens:
            raise ValueError(f'cu_q ends at {int(self.cu_q[-1])} but the row holds {n_tokens} tokens')
        if int(self.cu_k[-1]) != self.kv_index.numel():
            raise ValueError(f'cu_k ends at {int(self.cu_k[-1])} but kv_index has {self.kv_index.numel()} entries')
        if bool((self.kv_index >= n_tokens).any()) or bool((self.kv_index < 0).any()):
            raise ValueError('kv_index points outside the row')
        qlen = self.cu_q[1:] - self.cu_q[:-1]
        klen = self.cu_k[1:] - self.cu_k[:-1]
        if bool((qlen > klen).any()):
            raise ValueError('a segment has more queries than keys')

def shared_prefix_attention(q, k, v, plan: SharedPrefixPlan, causal: bool=True):
    from flash_attn import flash_attn_varlen_func
    plan.validate(q.shape[0])
    k_dup = k.index_select(0, plan.kv_index)
    v_dup = v.index_select(0, plan.kv_index)
    return flash_attn_varlen_func(q, k_dup, v_dup, plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, causal=causal)

def plan_from_segments(segments, device):
    cu_q, cu_k, kv, off = ([0], [0], [], 0)
    for q_start, q_len, keys in segments:
        if q_start != off:
            raise ValueError(f'segment starts at {q_start} but the running offset is {off}')
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + len(keys))
        kv.extend(keys)
        off += q_len
    qlen = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
    klen = [cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1)]
    return SharedPrefixPlan(
        cu_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        kv_index=torch.tensor(kv, dtype=torch.long, device=device),
        max_q=max(qlen),
        max_k=max(klen),
    )


_ACTIVE_PLAN = None
_ACTIVE_SEG = None

def set_active_plan(plan, seg=None):
    global _ACTIVE_PLAN, _ACTIVE_SEG
    _ACTIVE_PLAN = plan
    _ACTIVE_SEG = seg

def get_active_plan():
    return _ACTIVE_PLAN

def get_active_segments():
    return _ACTIVE_SEG

_PATCHED = False
_WRAPPER = None

def install_hook():
    global _PATCHED
    if _PATCHED:
        return
    from transformers.models.mistral import modeling_mistral as mm
    original = mm._flash_attention_forward

    def wrapper(query_states, key_states, value_states, attention_mask, query_length, *args, **kwargs):
        plan = get_active_plan()
        if plan is None:
            return original(query_states, key_states, value_states, attention_mask, query_length, *args, **kwargs)
        if query_states.size(0) != 1:
            raise ValueError(
                f'shared-prefix plan with batch {query_states.size(0)}: the plan addresses ONE packed row, so a batched call would attend across rows without erroring'
            )
        hq, d = (query_states.size(-2), query_states.size(-1))
        out = shared_prefix_attention(
            query_states.reshape(-1, hq, d),
            key_states.reshape(-1, key_states.size(-2), d),
            value_states.reshape(-1, value_states.size(-2), d),
            plan,
            causal=True,
        )
        return out.reshape(1, -1, hq, d)
    mm._flash_attention_forward = wrapper
    global _WRAPPER
    _WRAPPER = wrapper
    _PATCHED = True

def plan_from_batch(batch, device):
    seg = batch.get('molpo_seg', None)
    if seg is None:
        return (None, None)
    seg = seg[0] if seg.dim() == 2 else seg
    src = batch['molpo_kv_src']
    pre = batch['molpo_kv_pre']
    n_seg = int(seg.max()) + 1
    starts, lens = ([], [])
    for i in range(n_seg):
        pos = (seg == i).nonzero(as_tuple=True)[0]
        starts.append(int(pos[0]))
        lens.append(int(pos.numel()))
    segments = []
    for i in range(n_seg):
        keys = list(range(starts[i], starts[i] + lens[i]))
        j = int(src[i])
        if j >= 0:
            keys = list(range(starts[j], starts[j] + int(pre[i]))) + keys
        segments.append((starts[i], lens[i], keys))
    return (plan_from_segments(segments, device), seg.to(device))

def ensure_hook():
    install_hook()
    from transformers.models.mistral import modeling_mistral as mm
    if mm._flash_attention_forward is not _WRAPPER:
        raise RuntimeError(
            f"shared-prefix plan is active but transformers' _flash_attention_forward is not the hook (got {getattr(mm._flash_attention_forward, '__module__', '?')}.{getattr(mm._flash_attention_forward, '__name__', '?')}). The fused kernel would be bypassed and the reject tail would score NaN instead of raising."
        )
