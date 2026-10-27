"""Sine drive for C1 v5: a wave added along one direction at every token of one layer's output.

Every hook works at batch size 1 and changes nothing unless armed.
"""
import numpy as np
import torch


def drive(N: int, T: int, amplitude: float, phase: float = 0.0) -> np.ndarray:
    """amplitude * cos(2 pi t / N + phase) for t = 0 .. T-1, float64.

    Values below 1e-12 of the amplitude (floating-point residue where the cosine is exactly zero,
    for example every other token at N = 4) are set to exactly 0, so those tokens are left untouched.
    """
    t = np.arange(T, dtype=np.float64)
    w = amplitude * np.cos(2.0 * np.pi * t / N + phase)
    w[np.abs(w) < 1e-12 * abs(amplitude)] = 0.0
    return w


class WaveInjector:
    """Adds w[t] * direction at every token t of one layer's output.

    The add is done in float32 and cast once to the model dtype. Where w[t] * direction is exactly
    zero the stored value is kept as is, so a zero wave never flips -0.0 to +0.0. The realized drive
    along the direction, sum over dims of (stored after - before) * direction, is kept per token, so
    the input component can be computed from what the model dtype actually stored.
    """

    def __init__(self, layer: torch.nn.Module, direction: torch.Tensor):
        self.direction = direction                     # unit vector, float32, on the model device
        self.w = None
        self.realized = None
        self.stats = None
        self.handle = layer.register_forward_hook(self._hook)

    def arm(self, w: np.ndarray):
        self.w = torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.direction.device)

    def disarm(self):
        self.w = None

    def _hook(self, module, args, output):
        if self.w is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        assert hidden.shape[0] == 1 and hidden.shape[1] == self.w.shape[0], "batch 1 and one weight per token"
        before = hidden[0].detach().float()
        delta = self.w[:, None] * self.direction[None, :]
        after = torch.where(delta == 0, before, before + delta).to(hidden.dtype)
        del delta
        change = after.float() - before
        along = change @ self.direction
        self.realized = along.double().cpu().numpy()
        off = (change - along[:, None] * self.direction[None, :]).norm(dim=1)
        scale = float(self.w.abs().max())
        self.stats = {"max_abs_change": float(change.abs().max()),
                      "off_over_max_amplitude_median": float(off.quantile(0.5)) / scale if scale > 0 else None}
        new = hidden.clone()
        new[0] = after
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def remove(self):
        self.handle.remove()
