import torch
from pytorch_lightning.callbacks import Callback

class PreStepEMACallback(Callback):

    def __init__(
        self,
        decay=None,
        horizon_steps=None,
        bias_correction=True,
        use_for_validation=True,
        use_for_test=False,
        update_timing="before_optimizer_step",
    ):
        super().__init__()
        if update_timing != "before_optimizer_step":
            raise ValueError(f"Unsupported EMA update timing: {update_timing}")
        self.update_timing = update_timing
        if (decay is None) == (horizon_steps is None):
            raise ValueError('give exactly one of decay= or horizon_steps=')
        if horizon_steps is not None:
            if horizon_steps < 2:
                raise ValueError(f'horizon_steps must be >= 2, got {horizon_steps}')
            decay = 1.0 - 1.0 / horizon_steps
        if not 0.0 < decay < 1.0:
            raise ValueError(f'EMA decay must be in (0, 1), got {decay}')
        self.decay = decay
        self.horizon_steps = int(round(1.0 / (1.0 - decay)))
        self.bias_correction = bias_correction
        self.use_for_validation = use_for_validation
        self.use_for_test = use_for_test
        self.shadow = None
        self.backup = None
        self.n_updates = 0

    def _trainable(self, pl_module):
        return [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]

    def on_fit_start(self, trainer, pl_module):
        if self.shadow is None:
            self.shadow = {n: p.detach().to(torch.float32).clone() for n, p in self._trainable(pl_module)}
            n_par = sum((v.numel() for v in self.shadow.values()))
            if trainer.is_global_zero:
                print(
                    f'[ema] tracking {len(self.shadow)} tensors / {n_par / 1000000.0:.1f}M params in fp32 ({n_par * 4 / 1000000000.0:.2f} GB), horizon {self.horizon_steps:,} steps (decay {self.decay:.6f})',
                    flush=True,
                )
        else:
            moved = 0
            for n, p in self._trainable(pl_module):
                s = self.shadow.get(n)
                if s is not None and s.device != p.device:
                    self.shadow[n] = s.to(p.device)
                    moved += 1
            if trainer.is_global_zero:
                print(
                    f'[ema] resumed at n_updates={self.n_updates:,}; re-homed {moved}/{len(self.shadow)} shadow tensors onto the parameter devices',
                    flush=True,
                )

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self.n_updates += 1
        d = self.decay
        if self.bias_correction:
            d = min(d, (self.n_updates - 1) / self.n_updates)
        with torch.no_grad():
            for n, p in self._trainable(pl_module):
                s = self.shadow.get(n)
                if s is None:
                    self.shadow[n] = p.detach().to(torch.float32).clone()
                    continue
                s.mul_(d).add_(p.detach().to(torch.float32), alpha=1.0 - d)

    def _swap_in(self, pl_module):
        self.backup = {}
        with torch.no_grad():
            for n, p in self._trainable(pl_module):
                s = self.shadow.get(n)
                if s is None:
                    continue
                self.backup[n] = p.detach().clone()
                p.copy_(s.to(p.dtype))

    def _swap_out(self, pl_module):
        if not self.backup:
            return
        with torch.no_grad():
            for n, p in self._trainable(pl_module):
                b = self.backup.get(n)
                if b is not None:
                    p.copy_(b)
        self.backup = None

    def on_validation_start(self, trainer, pl_module):
        if self.use_for_validation and self.shadow is not None and (self.n_updates > 0):
            self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module):
        if getattr(self, 'use_for_test', False):
            if not self.shadow or self.n_updates <= 0:
                raise RuntimeError(
                    'ema_for_test requested without restored EMA weights/updates; restore the checkpoint through Trainer.test(ckpt_path=...)'
                )
            self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def state_dict(self):
        return {
            'update_timing': self.update_timing,
            'decay': self.decay,
            'n_updates': self.n_updates,
            'shadow': {k: v.cpu() for k, v in (self.shadow or {}).items()},
        }

    def load_state_dict(self, state):
        timing = state.get('update_timing', 'before_optimizer_step')
        if timing != self.update_timing:
            raise ValueError(f"EMA checkpoint update timing differs: {timing}")
        self.decay = state.get('decay', self.decay)
        self.n_updates = state.get('n_updates', 0)
        sh = state.get('shadow') or {}
        self.shadow = {k: v.clone() for k, v in sh.items()} if sh else None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['ema_state'] = self.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state' in checkpoint:
            self.load_state_dict(checkpoint['ema_state'])
