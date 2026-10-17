from __future__ import annotations
from itertools import islice
import torch
from torch.utils.data import DataLoader
import copy
import re
from contextlib import contextmanager
import torch.nn.functional as F

def format_matched_rejected_answer(chosen, generated):
    pattern = '^(.*?<(FLOAT|BOOLEAN|SELFIES)>)(.*?)(</\\2>.*)$'
    reference = re.fullmatch(pattern, chosen, flags=re.S)
    prediction = re.fullmatch(pattern, generated, flags=re.S)
    if reference is None or prediction is None:
        return generated
    if reference[2] != prediction[2]:
        raise ValueError('chosen/rejected answer grammar differs')
    body = prediction[3]
    if reference[2] in ('FLOAT', 'SELFIES'):
        body = re.sub('\\s+', '', body)
    else:
        body = body.strip()
        if body not in ('True', 'False'):
            raise ValueError('invalid active binary answer')
    leading = re.match('^\\s*', reference[3])[0]
    trailing = re.search('\\s*$', reference[3])[0]
    return reference[1] + leading + body + trailing + reference[4]
ANSWER_CANDIDATE_POLICIES = ('dar',)
S4_POSITIONAL = 'positional_diff_v1'
S4_SCORING_POLICIES = (S4_POSITIONAL,)
S4_DEFAULT_POLICY = S4_POSITIONAL
S4_POLICY_KEY = 'answer_correction_scoring_policy'
S4_SCALE_UNSCALED = 'unscaled_v1'
S4_SCALE_POLICIES = (S4_SCALE_UNSCALED,)
S4_SCALE_KEY = 'answer_correction_scale_policy'

def s4_scale_policy(args):
    policy = getattr(args, S4_SCALE_KEY, None) or S4_SCALE_UNSCALED
    if policy not in S4_SCALE_POLICIES:
        raise ValueError(f'unknown {S4_SCALE_KEY}={policy!r}; expected one of {S4_SCALE_POLICIES}')
    return policy


def s4_beta(args):
    beta = getattr(args, 'answer_correction_beta', None)
    beta = 1.0 if beta is None else float(beta)
    if beta <= 0:
        raise ValueError('answer correction beta must be positive')
    return beta

def s4_pair_loss(margin, beta):
    return F.softplus(-beta * margin) / beta

def s4_pair_slope(margin, beta):
    return torch.sigmoid(-beta * margin)
S4_ELIGIBLE, S4_EQUAL, S4_CHOSEN_PREFIX, S4_REJECTED_PREFIX, S4_EMPTY = (0, 1, 2, 3, 4)
S4_REASONS = {
    S4_EQUAL: 'equal_content',
    S4_CHOSEN_PREFIX: 'chosen_strict_prefix',
    S4_REJECTED_PREFIX: 'rejected_strict_prefix',
}

def s4_scoring_policy(args):
    policy = getattr(args, S4_POLICY_KEY, None) or S4_DEFAULT_POLICY
    if policy not in S4_SCORING_POLICIES:
        raise ValueError(f'unknown {S4_POLICY_KEY}={policy!r}; expected one of {S4_SCORING_POLICIES}')
    return policy

def positional_diff_masks(chosen, rejected):
    chosen, rejected = (list(chosen), list(rejected))
    k = min(len(chosen), len(rejected))
    differ = [chosen[j] != rejected[j] for j in range(k)]
    keep_c = differ + [True] * (len(chosen) - k)
    keep_r = differ + [True] * (len(rejected) - k)
    if not chosen or not rejected:
        reason = S4_EMPTY
    elif chosen == rejected:
        reason = S4_EQUAL
    elif not any(keep_c):
        reason = S4_CHOSEN_PREFIX
    elif not any(keep_r):
        reason = S4_REJECTED_PREFIX
    else:
        reason = S4_ELIGIBLE
    return (keep_c, keep_r, reason)

def _content_ids(labels, mask_ids, pad_id):
    drop = set(mask_ids) | {pad_id, -100}
    return [int(t) for t in labels.tolist() if int(t) not in drop]

def attach_s4_positional_masks(collator, features):
    tok = collator.tokenizer
    mask_ids, pad = (tok.molpo_mask_ids, tok.pad_token_id)
    indices = features['answer_correction_indices'].tolist()
    labels = features['labels']
    if 'seq_lens' in features:
        starts = [0, *features['seq_lens'].cumsum(0).tolist()]
        segment = lambda k: labels[0, starts[k]:starts[k + 1]]
    else:
        segment = lambda k: labels[k]
    if 'answer_correction_base_segments' in features:
        base = int(features['answer_correction_base_segments'])
        rejected_labels = [segment(base + j) for j in range(len(indices))]
    else:
        rejected_labels = list(features['answer_correction_batch']['labels'])
    keep_c, keep_r, len_c, len_r, reasons = ([], [], [], [], [])
    for j, i in enumerate(indices):
        c = _content_ids(segment(i), mask_ids, pad)
        r = _content_ids(rejected_labels[j], mask_ids, pad)
        kc, kr, reason = positional_diff_masks(c, r)
        if reason == S4_EMPTY:
            raise ValueError('S4 pair has an empty content side (data-contract error, not masked away)')
        keep_c.extend(kc)
        keep_r.extend(kr)
        len_c.append(len(c))
        len_r.append(len(r))
        reasons.append(reason)
    features['answer_correction_s4_chosen_keep'] = torch.tensor(keep_c, dtype=torch.bool)
    features['answer_correction_s4_rejected_keep'] = torch.tensor(keep_r, dtype=torch.bool)
    features['answer_correction_s4_chosen_len'] = torch.tensor(len_c, dtype=torch.long)
    features['answer_correction_s4_rejected_len'] = torch.tensor(len_r, dtype=torch.long)
    features['answer_correction_s4_reason'] = torch.tensor(reasons, dtype=torch.long)
    return features

def check_s2_s4_mask_conformance(args):
    if not float(getattr(args, 'answer_correction_weight', 0.0) or 0.0):
        return None
    policy = s4_scoring_policy(args)
    return policy


def answer_candidate_policy(args):
    policy = getattr(args, 'answer_candidate_policy', None) or 'dar'
    if policy not in ANSWER_CANDIDATE_POLICIES:
        raise ValueError(f'unknown answer_candidate_policy={policy!r}; expected one of {ANSWER_CANDIDATE_POLICIES}')
    return policy


def attach_answer_batch(collator, features, samples):
    if not collator.train or not float(getattr(collator.args, 'answer_correction_weight', 0.0) or 0.0):
        return features
    required = ('v8_rejected_answer_text', 'v8_answer_weight', 'v8_answer_type')
    for sample in samples:
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f'v8 answer correction requires dataset columns: {missing}')
    check_s2_s4_mask_conformance(collator.args)
    answer_candidate_policy(collator.args)
    selected = []
    indices = []
    weights = []
    for index, sample in enumerate(samples):
        weight = float(sample['v8_answer_weight'])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f'invalid v8 answer weight: {weight}')
        if not weight:
            continue
        wrong = format_matched_rejected_answer(sample['target_text'], sample['v8_rejected_answer_text'])
        if not wrong or wrong.strip() == sample['target_text'].strip():
            raise ValueError('active answer correction needs a distinct, nonempty rejected answer')
        selected.append(dict(sample, v8_rejected_answer_text=wrong))
        indices.append(index)
        weights.append(weight)
    features['answer_correction_chosen_count'] = torch.tensor(len(samples), dtype=torch.long)
    if not selected:
        return features
    return _attach_selected_pairs(collator, features, selected, indices, weights)


def _attach_selected_pairs(collator, features, selected, indices, weights):
    features['answer_correction_indices'] = torch.tensor(indices, dtype=torch.long)
    features['answer_correction_weights'] = torch.tensor(weights, dtype=torch.float)
    if 'molpo_seg' in features:
        features = append_shared_answer_tails(collator, features, selected, indices)
    else:
        features = _attach_auxiliary_pairs(collator, features, selected)
    if s4_scoring_policy(collator.args) == S4_POSITIONAL:
        features = attach_s4_positional_masks(collator, features)
    return features

def _attach_auxiliary_pairs(collator, features, selected):
    from data_utils import DataCollator
    args = copy.deepcopy(collator.args)
    args.train_molpo = False
    args.eval_molpo = False
    args.answer_correction_weight = 0.0
    args.molpo_share_prefix = False
    args.apply_sequence_packing = False
    plain = DataCollator(
        tokenizer=collator.tokenizer,
        padding=True,
        max_length=collator.max_length,
        return_tensors='pt',
        train=True,
        args=args,
        truncate=collator.truncate,
    )
    pairs = [dict(s, target_text=s['v8_rejected_answer_text']) for s in selected]
    prompts = plain.select_mol_representation([s['prompt_text'] for s in pairs], plain.mol_representation)
    for prompt, sample in zip(prompts, pairs):
        length = len(plain.tokenizer.encode(prompt, add_special_tokens=False))
        length += len(plain.tokenizer.encode(sample['target_text'], add_special_tokens=False))
        if length > plain._cap:
            raise ValueError('v8 answer pair would be truncated; builder must mark it ineligible')
    auxiliary = plain(pairs)
    n = len(selected)
    labels = auxiliary['labels'].clone()
    for token in collator.tokenizer.molpo_mask_ids:
        labels.masked_fill_(labels == token, -100)
    counts = (labels[:, 1:] != -100).sum(-1)
    if bool((counts == 0).any()):
        raise ValueError('v8 answer has no scored content after training truncation')
    auxiliary['answer_content_labels'] = labels
    features['answer_correction_batch'] = auxiliary
    assert auxiliary['input_ids'].shape[0] == n
    return features

def append_shared_answer_tails(collator, features, selected, indices):
    base_segments = features['seq_lens'].numel()
    chosen_count = int(features['answer_correction_chosen_count'])
    if base_segments != 3 * chosen_count:
        raise ValueError('v8 shared answers require the original three-cell layout')
    base_tokens = features['input_ids'].shape[1]
    features['answer_correction_base_tokens'] = torch.tensor(base_tokens, dtype=torch.long)
    features['answer_correction_base_segments'] = torch.tensor(base_segments, dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), features['seq_lens'].cumsum(0)])
    tails, positions, labels, segments, lens, prefixes = ([], [], [], [], [], [])
    for j, (sample, i) in enumerate(zip(selected, indices)):
        lo, hi = (int(starts[i]), int(starts[i + 1]))
        original_labels = features['labels'][0, lo:hi]
        labeled = (original_labels != -100).nonzero().flatten()
        if not labeled.numel():
            raise ValueError('active answer correction has no chosen answer tokens')
        prefix = int(labeled[0])
        wrong = collator.tokenizer.encode(sample['v8_rejected_answer_text'], add_special_tokens=False)
        if prefix < 2 or prefix + len(wrong) > collator._cap:
            raise ValueError('v8 answer cannot share its prefix without truncation')
        carry = int(features['input_ids'][0, lo + prefix - 1])
        if collator.tokenizer.mol_token_id in [carry, *wrong]:
            raise ValueError('S4 answer/carry must not inject an additional graph slot')
        content = [
            t
            for t in wrong
            if t not in collator.tokenizer.molpo_mask_ids
            and t != collator.tokenizer.pad_token_id
        ]
        if not content:
            raise ValueError('v8 rejected answer has no content tokens')
        tail = [carry, *wrong]
        tails.extend(tail)
        positions.extend(range(prefix - 1, prefix + len(wrong)))
        labels.extend([-100, *wrong])
        segments.extend([base_segments + j] * len(tail))
        lens.append(len(tail))
        prefixes.append(prefix - 1)
    for name, values in [
        ('input_ids', tails),
        ('position_ids', positions),
        ('labels', labels),
        ('molpo_seg', segments),
    ]:
        added = torch.tensor([values], dtype=features[name].dtype)
        if name == 'labels':
            added.masked_fill_(added == collator.tokenizer.pad_token_id, -100)
        features[name] = torch.cat([features[name], added], dim=1)
    features['molpo_labels'] = torch.cat(
        [
            features['molpo_labels'],
            torch.full((1, len(tails)), -100, dtype=features['molpo_labels'].dtype),
        ],
        dim=1,
    )
    for name, values in [('seq_lens', lens), ('molpo_kv_src', indices), ('molpo_kv_pre', prefixes)]:
        features[name] = torch.cat([features[name], torch.tensor(values, dtype=features[name].dtype)])
    features['tasks'] = torch.cat([features['tasks'], features['tasks'][indices]])
    features['is_mol_token'] = features['input_ids'] == collator.tokenizer.mol_token_id
    return features

@contextmanager
def original_molpo_inputs(batch, logits, gathered_logps, gathered_selected):
    from model.molpo_shared_prefix import get_active_plan, get_active_segments, set_active_plan
    base = batch.get('answer_correction_base_tokens')
    names = ('labels', 'molpo_labels', 'position_ids')
    values = {name: batch.get(name) for name in names}
    plan, seg = (get_active_plan(), get_active_segments())
    if base is not None:
        base = int(base)
        values = {name: value[:, :base] if value is not None else None for name, value in values.items()}
        if logits is not None:
            logits = logits[:, :base]
        if gathered_logps is not None:
            gathered_selected = gathered_selected[:, :base - 1]
            gathered_logps = gathered_logps[:int(gathered_selected.sum())]
        if seg is None:
            raise ValueError('fused S4 requires explicit active segment identity')
        set_active_plan(plan, seg[:base])
    try:
        yield dict(values, logits=logits, gathered_logps=gathered_logps, gathered_selected=gathered_selected)
    finally:
        set_active_plan(plan, seg)

def answer_preference_loss(chosen_logps, rejected_logps, weights, chosen_count, beta: float=1.0):
    if beta <= 0:
        raise ValueError('answer correction beta must be positive')
    if chosen_logps.shape != rejected_logps.shape or weights.shape != chosen_logps.shape:
        raise ValueError('answer preference pair/weight shapes differ')
    if not bool(torch.isfinite(chosen_logps).all() & torch.isfinite(rejected_logps).all()):
        raise ValueError('nonfinite answer preference scores')
    denominator = torch.as_tensor(chosen_count, device=chosen_logps.device, dtype=torch.float)
    if denominator <= 0:
        raise ValueError('empty chosen population')
    margin = chosen_logps.float() - rejected_logps.float()
    loss = (weights.float() * s4_pair_loss(margin, beta)).sum() / denominator
    return (loss, margin)

def model_answer_correction(model, batch, logits=None, gathered_logps=None, gathered_selected=None):
    weight = float(getattr(model.args, 'answer_correction_weight', 0.0) or 0.0)
    if not weight:
        return (None, {})
    if 'answer_correction_chosen_count' not in batch:
        raise KeyError('answer correction enabled but collator did not attach v8 metadata')
    chosen_count = int(batch['answer_correction_chosen_count'])
    active_pairs = torch.zeros(chosen_count, device=batch.labels.device)
    active_margins = torch.full_like(active_pairs, float('nan'))
    margin_gradients = torch.zeros_like(active_pairs)
    per_task = {
        'answer_correction/active_pair': active_pairs,
        'answer_correction/active_margin': active_margins,
        'answer_correction/margin_gradient_per_row': margin_gradients,
        'answer_correction/loss_per_row': torch.zeros_like(active_pairs),
        'answer_correction/weighted_loss_per_row': torch.zeros_like(active_pairs),
    }
    answer_candidate_policy(model.args)
    if 'answer_correction_indices' not in batch:
        zero = torch.tensor(0.0, device=batch.labels.device)
        extra = (
            {f'answer_correction/{name}': zero for name in S4_COUNTER_NAMES}
            if s4_scoring_policy(model.args) == S4_POSITIONAL
            else {}
        )
        if extra:
            extra['answer_correction/s4_eligible_weight_per_row'] = torch.zeros_like(active_pairs)
            extra['answer_correction/s4_skipped_weight_per_row'] = torch.zeros_like(active_pairs)
        return (
            None,
            dict(
                per_task,
                **extra,
                **{
                    'answer_correction/fraction': zero,
                    'answer_correction/loss': zero,
                    'answer_correction/weighted_loss': zero,
                    'answer_correction/margin_gradient_mass_per_chosen': zero,
                },
            ),
        )
    s4_scoring_policy(model.args)
    s4_scale_policy(model.args)
    return _positional_answer_correction(model, batch, weight, per_task, logits, gathered_logps, gathered_selected)

def _record_answer_loss_per_row(per_task, indices, margin, weights, beta, weight):
    row_loss = per_task['answer_correction/loss_per_row']
    pair_loss = weights.detach().float() * s4_pair_loss(
        margin.detach().float(), beta
    )
    row_loss.index_add_(0, indices, pair_loss.to(row_loss.dtype))
    per_task['answer_correction/weighted_loss_per_row'] = weight * row_loss
S4_COUNTER_NAMES = (
    's4_pairs',
    's4_eligible_pairs',
    's4_skipped_equal_content',
    's4_skipped_chosen_strict_prefix',
    's4_skipped_rejected_strict_prefix',
    's4_skipped_weight',
    's4_eligible_weight',
    's4_retained_tokens_chosen_sum',
    's4_retained_tokens_rejected_sum',
    's4_content_tokens_chosen_sum',
    's4_content_tokens_rejected_sum',
)

class S4WindowCounters:

    def __init__(self):
        self.sums = None

    def add(self, metrics):
        vec = torch.stack(
            [
                torch.as_tensor(metrics[f'answer_correction/{k}'], dtype=torch.float64)
                .detach()
                .cpu()
                for k in S4_COUNTER_NAMES
            ]
        )
        self.sums = vec if self.sums is None else self.sums + vec

    def finish(self, device=None):
        sums = self.sums if self.sums is not None else torch.zeros(len(S4_COUNTER_NAMES), dtype=torch.float64)
        self.sums = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sums = sums.to(device) if device is not None else sums
            torch.distributed.all_reduce(sums)
        return {k: float(v) for k, v in zip(S4_COUNTER_NAMES, sums.cpu().tolist())}

def content_token_logps(labels, positions, dense, flat, selected, drop_ids):
    from model.blip2_stage3 import shifted_per_token_logps, segment_ids
    labels = labels.clone()
    for token in drop_ids:
        labels.masked_fill_(labels == token, -100)
    shifted = labels[:, 1:]
    content = shifted != -100
    seg = segment_ids(positions)
    if flat is not None:
        if bool((content & ~selected).any()):
            raise ValueError('S4 content label outside the gathered label positions')
        if seg is not None:
            index = seg[1:][selected[0]]
        else:
            index = torch.arange(selected.size(0), device=selected.device).unsqueeze(1).expand_as(selected)[selected]
        keep = content[selected]
        return (flat[keep], index[keep], shifted[selected][keep])
    values, mask = shifted_per_token_logps(dense, labels)
    if seg is not None:
        return (values[0][mask[0]], seg[1:][mask[0]], shifted[0][mask[0]])
    rows = torch.arange(mask.size(0), device=mask.device).unsqueeze(1).expand_as(mask)
    return (values[mask], rows[mask], shifted[mask])

def _pair_tokens(values, instance, ids, pair_instance, lengths):
    if instance.numel() > 1 and bool((instance[1:] < instance[:-1]).any()):
        raise ValueError('S4 content tokens are not grouped by instance')
    size = int(max(int(pair_instance.max()), int(instance.max()) if instance.numel() else 0)) + 1
    counts = torch.bincount(instance, minlength=size)
    if not torch.equal(counts[pair_instance], lengths.to(counts.device)):
        raise ValueError('S4 content-token count differs from the collator mask metadata')
    lengths = lengths.to(counts.device)
    starts = counts.cumsum(0) - counts
    pair = torch.repeat_interleave(torch.arange(lengths.numel(), device=counts.device), lengths)
    within = torch.arange(pair.numel(), device=counts.device) - (lengths.cumsum(0) - lengths)[pair]
    position = starts[pair_instance][pair] + within
    return (values[position], ids[position], pair, within)

def positional_keep(ids_c, pair_c, within_c, ids_r, pair_r, within_r, n):
    width = (
        int(
            max(
                int(within_c.max()) if within_c.numel() else -1,
                int(within_r.max()) if within_r.numel() else -1,
            )
        )
        + 1
    )
    chosen = torch.full((n, width), -1, dtype=torch.long, device=ids_c.device)
    rejected = torch.full((n, width), -2, dtype=torch.long, device=ids_c.device)
    chosen[pair_c, within_c] = ids_c.long()
    rejected[pair_r, within_r] = ids_r.long()
    differ = chosen != rejected
    return (differ[pair_c, within_c], differ[pair_r, within_r])

def _positional_answer_correction(model, batch, weight, per_task, logits, gathered_logps, gathered_selected):
    tok = model.blip2model.llm_tokenizer
    drop = [*tok.molpo_mask_ids, tok.pad_token_id]
    indices = batch['answer_correction_indices']
    weights = batch['answer_correction_weights']
    n = int(weights.numel())
    chosen_count = batch['answer_correction_chosen_count']
    for key in ('answer_correction_s4_chosen_keep', 'answer_correction_s4_reason'):
        if key not in batch:
            raise KeyError(f'positional_diff_v1 S4 needs collator metadata {key!r} (collator/model policy mismatch)')
    values, instance, ids = content_token_logps(
        batch['labels'],
        batch.get('position_ids'),
        logits,
        gathered_logps,
        gathered_selected,
        drop,
    )
    len_c, len_r = (batch['answer_correction_s4_chosen_len'], batch['answer_correction_s4_rejected_len'])
    vc, ic, pc, wc = _pair_tokens(values, instance, ids, indices, len_c)
    if 'answer_correction_base_segments' in batch:
        base = int(batch['answer_correction_base_segments'])
        tails = torch.arange(base, base + n, device=indices.device)
        vr, ir, pr, wr = _pair_tokens(values, instance, ids, tails, len_r)
    else:
        from model.molpo_shared_prefix import get_active_plan, get_active_segments, set_active_plan
        plan, seg = (get_active_plan(), get_active_segments())
        auxiliary = batch['answer_correction_batch']
        try:
            set_active_plan(None, None)
            output = model.blip2model(auxiliary)
            aux = content_token_logps(
                auxiliary['labels'],
                auxiliary.get('position_ids'),
                output.get('logits'),
                output.get('gathered_logps'),
                output.get('gathered_selected'),
                drop,
            )
        finally:
            set_active_plan(plan, seg)
        vr, ir, pr, wr = _pair_tokens(*aux, torch.arange(n, device=indices.device), len_r)
    keep_c, keep_r = positional_keep(ic, pc, wc, ir, pr, wr, n)
    if not (
        torch.equal(keep_c, batch['answer_correction_s4_chosen_keep'].to(keep_c.device))
        and torch.equal(
            keep_r, batch['answer_correction_s4_rejected_keep'].to(keep_r.device)
        )
    ):
        raise ValueError('S4 positional masks differ from the collator metadata')

    def side(v, p, keep):
        sums = torch.zeros(n, dtype=v.dtype, device=v.device).index_add_(0, p[keep], v[keep])
        count = torch.zeros(n, dtype=torch.long, device=v.device).index_add_(0, p[keep], torch.ones_like(p[keep]))
        return (sums, count)
    sum_c, cnt_c = side(vc, pc, keep_c)
    sum_r, cnt_r = side(vr, pr, keep_r)
    eligible = (cnt_c > 0) & (cnt_r > 0)
    reason = batch['answer_correction_s4_reason'].to(eligible.device)
    if not torch.equal(eligible, reason == S4_ELIGIBLE):
        raise ValueError('S4 eligibility differs from the collator reason codes')
    q_c = sum_c[eligible] / cnt_c[eligible].to(sum_c.dtype)
    q_r = sum_r[eligible] / cnt_r[eligible].to(sum_r.dtype)
    beta = s4_beta(model.args)
    live_w = weights[eligible]
    loss, margin_live = answer_preference_loss(q_c, q_r, live_w, chosen_count, beta)
    _record_answer_loss_per_row(per_task, indices[eligible], margin_live, live_w, beta, weight)
    margin = torch.zeros(n, dtype=margin_live.dtype, device=margin_live.device)
    margin[eligible] = margin_live.detach()
    slope = torch.zeros_like(margin)
    slope[eligible] = s4_pair_slope(
        margin_live.detach(),
        beta,
    )
    gradient_mass = weight * (weights * slope).sum() / chosen_count.float()
    active_pairs, active_margins, margin_gradients = (
        per_task['answer_correction/active_pair'],
        per_task['answer_correction/active_margin'],
        per_task['answer_correction/margin_gradient_per_row'],
    )
    live_idx = indices[eligible]
    active_pairs[live_idx] = 1.0
    per_anchor = torch.zeros_like(active_pairs).index_add_(
        0, live_idx, torch.ones_like(live_idx, dtype=active_pairs.dtype)
    )
    margin_sum = torch.zeros_like(active_pairs).index_add_(0, live_idx, margin_live.detach().to(active_pairs.dtype))
    active_margins[live_idx] = (margin_sum / per_anchor.clamp_min(1))[live_idx]
    margin_gradients.index_add_(0, indices, (weight * weights * slope).to(margin_gradients.dtype))
    live_w_row = torch.zeros_like(active_pairs).index_add_(
        0, indices, (weights * eligible.to(weights.dtype)).to(active_pairs.dtype)
    )
    skip_w_row = torch.zeros_like(active_pairs).index_add_(
        0, indices, (weights * (~eligible).to(weights.dtype)).to(active_pairs.dtype)
    )
    per_task['answer_correction/s4_eligible_weight_per_row'] = live_w_row
    per_task['answer_correction/s4_skipped_weight_per_row'] = skip_w_row
    f = lambda x: torch.as_tensor(x, dtype=torch.float, device=weights.device).detach()
    n_live = int(eligible.sum())
    metrics = {
        's4_pairs': f(n),
        's4_eligible_pairs': f(n_live),
        's4_skipped_equal_content': f(int((reason == S4_EQUAL).sum())),
        's4_skipped_chosen_strict_prefix': f(int((reason == S4_CHOSEN_PREFIX).sum())),
        's4_skipped_rejected_strict_prefix': f(
            int((reason == S4_REJECTED_PREFIX).sum())
        ),
        's4_skipped_weight': f(weights[~eligible].sum()),
        's4_eligible_weight': f(live_w.sum()),
        's4_retained_tokens_chosen_sum': f(int(cnt_c[eligible].sum())),
        's4_retained_tokens_rejected_sum': f(int(cnt_r[eligible].sum())),
        's4_content_tokens_chosen_sum': f(int(len_c[eligible.to(len_c.device)].sum())),
        's4_content_tokens_rejected_sum': f(
            int(len_r[eligible.to(len_r.device)].sum())
        ),
        'loss': loss.detach(),
        'weighted_loss': (weight * loss).detach(),
        'fraction': f(int(torch.unique(live_idx).numel())) / chosen_count.float(),
        'margin_gradient_mass_per_chosen': gradient_mass.detach(),
    }
    if n_live:
        lc, lr = (len_c.to(cnt_c.device)[eligible], len_r.to(cnt_r.device)[eligible])
        metrics.update(
            {
                'margin': margin_live.detach().float().mean(),
                'margin_nonpositive_fraction': (margin_live.detach() <= 0)
                .float()
                .mean(),
                'active_margin_slope': slope[eligible].float().mean(),
                's4_retained_tokens_chosen': cnt_c[eligible].float().mean(),
                's4_retained_tokens_rejected': cnt_r[eligible].float().mean(),
                's4_retained_fraction_chosen': (cnt_c[eligible] / lc).float().mean(),
                's4_retained_fraction_rejected': (cnt_r[eligible] / lr).float().mean(),
            }
        )
    return (weight * loss, {**per_task, **{f'answer_correction/{k}': v for k, v in metrics.items()}})


TERMS = ('s1', 's2', 's3', 's4')
MODE = 'global_molpo_terms_v1'

def _world():
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    return (torch.distributed.get_world_size() if distributed else 1, distributed)

def count_molpo_terms(batch, *, division=3, pad_token_id=None):
    labels = batch['labels']
    molpo = batch['molpo_labels']
    if labels.shape != molpo.shape or labels.ndim != 2 or division not in (2, 3):
        raise ValueError('invalid MolPO label layout')
    label_mask = labels[:, 1:] != -100
    if pad_token_id is not None:
        label_mask &= labels[:, 1:] != pad_token_id
    graph_mask = molpo[:, 1:] != -100
    if bool((graph_mask & ~label_mask).any()):
        raise ValueError('molpo labels must be a subset of model-scored labels')
    segments = batch.get('molpo_seg')
    seq_lens = batch.get('seq_lens')
    if segments is None and labels.shape[0] == 1 and (seq_lens is not None) and (seq_lens.numel() > 1):
        seq_lens = seq_lens.reshape(-1).to(labels.device)
        if int(seq_lens.sum()) != labels.shape[1]:
            raise ValueError('seq_lens do not cover the packed row')
        segments = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=labels.device), seq_lens)
    if segments is not None:
        segments = segments.reshape(-1)
        if labels.shape[0] != 1 or segments.numel() != labels.numel():
            raise ValueError('explicit packed segments differ from label layout')
        n = int(segments.max()) + 1
        index = segments[1:]
        lc = torch.bincount(index[label_mask[0]], minlength=n)
        gc = torch.bincount(index[graph_mask[0]], minlength=n)
    else:
        positions = batch.get('position_ids')
        if positions is not None and labels.shape[0] == 1 and bool((positions[0, 1:] < positions[0, :-1]).any()):
            raise ValueError('packed MolPO denominator counting requires molpo_seg')
        n = labels.shape[0]
        lc, gc = (label_mask.sum(-1), graph_mask.sum(-1))
    base = int(batch.get('answer_correction_base_segments', n))
    if base > n or base % division or base <= 0:
        raise ValueError('invalid S1-S3 base segment count')
    anchors = base // division
    declared = batch.get('answer_correction_chosen_count')
    if declared is not None and int(declared) != anchors:
        raise ValueError('S4 chosen count differs from original anchors')
    valid = (gc[:anchors] > 0) & (gc[anchors:2 * anchors] > 0)
    if batch.get('molpo_ok') is not None:
        ok = batch['molpo_ok'].reshape(-1)
        if ok.numel() != anchors:
            raise ValueError('molpo_ok differs from chosen population')
        valid &= ok.bool()
    return torch.stack(
        (
            lc[:anchors].sum(),
            valid.sum(),
            gc[2 * anchors : 3 * anchors].sum() if division == 3 else lc.new_zeros(()),
            lc.new_tensor(anchors),
        )
    ).to(torch.int64)

class MolPOWindowReducer:

    def __init__(self, *, accumulation, coefficients):
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError('fixed positive accumulation required')
        weights = torch.tensor(coefficients, dtype=torch.float64)
        if (
            weights.shape != (4,)
            or not torch.isfinite(weights).all()
            or bool((weights < 0).any())
            or (not bool((weights > 0).any()))
        ):
            raise ValueError('four nonnegative coefficients with at least one enabled term required')
        self.accumulation = accumulation
        self.coefficients = weights
        self.active = False
        self.last_stats = None

    def start(self, local_denominators, *, window_id, microbatches):
        if self.active:
            raise RuntimeError('previous MolPO window not finalized')
        if type(microbatches) is not int or not 1 <= microbatches <= self.accumulation:
            raise ValueError('invalid partial/full window size')
        if (
            local_denominators.shape != (4,)
            or local_denominators.dtype != torch.int64
            or bool((local_denominators < 0).any())
        ):
            raise ValueError('four nonnegative int64 local denominators required')
        self.world, distributed = _world()
        self.local_denominators = local_denominators.detach().clone()
        self.denominators = local_denominators.detach().clone()
        if distributed:
            torch.distributed.all_reduce(self.denominators)
        self.window_coefficients = self.coefficients.clone()
        weights = self.window_coefficients.to(self.denominators.device)
        if not bool(((self.denominators > 0) & (weights > 0)).any()):
            raise RuntimeError('entire global window objective is empty; refuse optimizer update')
        self.scales = torch.where(
            self.denominators > 0,
            weights * (self.world * self.accumulation) / self.denominators.clamp_min(1),
            0.0,
        )
        self.sums = torch.zeros(4, dtype=torch.float64, device=self.denominators.device)
        self.seen_counts = torch.zeros_like(self.denominators)
        self.expected_microbatches = microbatches
        self.seen_microbatches = 0
        self.window_id = window_id
        self.active = True

    def add(self, term_sums, actual_counts, *, window_id):
        if not self.active or window_id != self.window_id:
            raise RuntimeError('MolPO window identity mismatch')
        if term_sums.shape != (4,) or actual_counts.shape != (4,) or actual_counts.dtype != torch.int64:
            raise ValueError('four differentiable term sums and exact int64 counts required')
        self.seen_microbatches += 1
        if self.seen_microbatches > self.expected_microbatches:
            raise RuntimeError('too many microbatches for frozen denominator')
        self.sums.add_(term_sums.detach().to(torch.float64))
        self.seen_counts.add_(actual_counts.detach())
        return (term_sums * self.scales.to(term_sums.dtype)).sum()

    @torch.no_grad()
    def finish(self):
        if not self.active:
            raise RuntimeError('no active MolPO window')
        mismatch = int(
            self.seen_microbatches != self.expected_microbatches
            or not torch.equal(self.seen_counts, self.local_denominators)
        )
        failure = self.denominators.new_tensor(mismatch)
        _, distributed = _world()
        if distributed:
            torch.distributed.all_reduce(failure)
            torch.distributed.all_reduce(self.sums)
        if int(failure):
            raise RuntimeError(
                f'lookahead counts/window size differ from actual forwards on at least '
                f'one rank: local '
                f'microbatches={self.seen_microbatches}/{self.expected_microbatches}, '
                f'counts={self.seen_counts.tolist()}/{self.local_denominators.tolist()}'
            )
        if not torch.isfinite(self.sums).all() or bool((self.sums < 0).any()):
            raise RuntimeError('nonfinite/negative global term sum; refuse optimizer update')
        if bool(((self.denominators == 0) & (self.sums != 0)).any()):
            raise RuntimeError('nonzero term numerator with zero denominator')
        means = torch.where(self.denominators > 0, self.sums / self.denominators.clamp_min(1), 0.0)
        weighted = means * self.window_coefficients.to(means.device)
        if not bool(weighted.sum() > 0):
            raise RuntimeError('entire global objective is zero; refuse optimizer/decay update')
        self.last_stats = dict(
            reduction=MODE,
            window_id=self.window_id,
            world_size=self.world,
            configured_accumulation=self.accumulation,
            microbatches=self.seen_microbatches,
            denominators=self.denominators.tolist(),
            sums=self.sums.tolist(),
            means=means.tolist(),
            weighted_terms=weighted.tolist(),
            backward_scales=self.scales.tolist(),
        )
        self.active = False
        return self.last_stats

class WindowTaskEMA:

    def __init__(self, *, decay=0.99, placeholders=()):
        if not 0 <= decay < 1:
            raise ValueError('invalid task EMA decay')
        self.values = {}
        self.decay = float(decay)
        self.placeholders = set(placeholders)
        self.updates = 0
        self.active = False

    def begin(self, device):
        if self.active:
            raise RuntimeError('task EMA window already active')
        self.frozen = dict(self.values)
        self.sums = {}
        self.counts = {}
        self.device = device
        self.active = True

    def seeded(self, tasks, *, device):
        values = self.frozen if self.active else self.values
        return torch.tensor([t in self.placeholders or t in values for t in tasks], device=device, dtype=torch.bool)

    def scales(self, tasks, *, device, dtype):
        values = self.frozen if self.active else self.values
        return torch.tensor(
            [
                0.001 if t in self.placeholders else abs(values.get(t, -1.0))
                for t in tasks
            ],
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def add(self, tasks, chosen_logps, chosen_counts):
        if not self.active or len(tasks) != chosen_logps.numel():
            raise RuntimeError('task EMA accumulation outside its window')
        for i, task in enumerate(tasks):
            if task in self.placeholders or not bool(chosen_counts[i] > 0):
                continue
            self.sums[task] = self.sums.get(task, 0.0) + chosen_logps[i].detach().double()
            self.counts[task] = self.counts.get(task, 0) + 1

    @torch.no_grad()
    def finish(self):
        if not self.active:
            raise RuntimeError('task EMA window missing')
        _, distributed = _world()
        tasks = sorted(self.sums)
        if distributed:
            import torch.distributed as dist
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, tasks)
            tasks = sorted(set().union(*map(set, gathered)))
        sums = torch.tensor([float(self.sums.get(t, 0.0)) for t in tasks], dtype=torch.float64, device=self.device)
        counts = torch.tensor([self.counts.get(t, 0) for t in tasks], dtype=torch.int64, device=self.device)
        if distributed:
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(counts)
        if not torch.isfinite(sums).all():
            raise RuntimeError('nonfinite task EMA window; refusing state update')
        next_values = dict(self.values)
        for i, task in enumerate(tasks):
            if int(counts[i]):
                mean = float(sums[i] / counts[i])
                value = mean if task not in self.values else self.decay * self.values[task] + (1 - self.decay) * mean
                if value >= -1e-06:
                    raise RuntimeError('task chosen-logp scale became zero/nonnegative')
                next_values[task] = value
        self.values = next_values
        self.updates += 1
        self.active = False


class WindowMetadata(dict):

    def to(self, device=None, **kwargs):
        return type(self)(
            (
                (
                    key,
                    (
                        value.to(device=device, **kwargs)
                        if isinstance(value, torch.Tensor)
                        else value
                    ),
                )
                for key, value in self.items()
            )
        )

class MolPOWindowDataLoader(DataLoader):

    def __init__(
        self,
        *args,
        accumulation=1,
        window_batch_limit=None,
        molpo_division=3,
        pad_token_id=None,
        **kwargs
    ):
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError('fixed positive accumulation required')
        if window_batch_limit is not None and (
            not (
                type(window_batch_limit) is int
                and window_batch_limit >= 1
                or (type(window_batch_limit) is float and 0 < window_batch_limit <= 1)
            )
        ):
            raise ValueError('window_batch_limit must be positive count or fraction in (0,1]')
        if kwargs.get('persistent_workers', False):
            raise ValueError('window loader requires fresh epoch workers so collator epoch/cardinal is current')
        super().__init__(*args, **kwargs)
        self.accumulation = accumulation
        self.window_batch_limit = window_batch_limit
        self.molpo_division = molpo_division
        self.pad_token_id = pad_token_id

    def _limit(self):
        if type(self.window_batch_limit) is float:
            return int(len(self) * self.window_batch_limit)
        return min(len(self), self.window_batch_limit) if self.window_batch_limit is not None else len(self)

    def __iter__(self):
        limit = self._limit()
        remaining = iter(islice(super().__iter__(), limit))
        cursor = 0
        while True:
            window = list(islice(remaining, self.accumulation))
            if not window:
                break
            counts = [
                count_molpo_terms(
                    b, division=self.molpo_division, pad_token_id=self.pad_token_id
                )
                for b in window
            ]
            total = torch.stack(counts).sum(0)
            for offset, batch in enumerate(window):
                batch['molpo_window'] = WindowMetadata(
                    window_start=cursor,
                    offset=offset,
                    microbatches=len(window),
                    local_denominators=total.clone(),
                )
                yield batch
            cursor += len(window)
