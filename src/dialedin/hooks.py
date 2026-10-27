"""Forward hooks: pulse injection, head-output probes, and the knob.

Every hook works at batch size 1 and changes nothing unless armed.
"""
import numpy as np
import torch

from . import metrics


class PulseInjector:
    """Adds amplitude * direction to one decoder layer's residual-stream output at given positions.

    The add is done in float32 and cast once to the model dtype. The realized delta is measured.
    """

    def __init__(self, layer: torch.nn.Module, direction: torch.Tensor):
        self.direction = direction                     # unit vector, float32, on the model device
        self.positions = None
        self.amplitude = 0.0
        self.realized = None
        self.handle = layer.register_forward_hook(self._hook)

    def arm(self, positions: np.ndarray, amplitude: float):
        self.positions = torch.as_tensor(positions, device=self.direction.device, dtype=torch.long)
        self.amplitude = float(amplitude)

    def disarm(self):
        self.positions = None

    def _hook(self, module, args, output):
        if self.positions is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        before = hidden[0, self.positions, :].detach()
        work = torch.promote_types(torch.float32, self.direction.dtype)
        delta = (self.amplitude * self.direction.to(work)).expand_as(before)
        # Where the delta is exactly zero the value is kept as is, so adding zero never flips -0.0 to +0.0.
        after = torch.where(delta == 0, before.to(work), before.to(work) + delta).to(hidden.dtype)
        hidden[0, self.positions, :] = after
        if self.amplitude > 0:
            d = after.to(work) - before.to(work)
            along = d @ self.direction.to(work)
            off = (d - along[:, None] * self.direction.to(work)).norm(dim=1)
            self.realized = {"along_over_amplitude_median": float(along.quantile(0.5) / self.amplitude),
                             "off_over_amplitude_median": float(off.quantile(0.5) / self.amplitude),
                             "off_over_amplitude_max": float(off.max() / self.amplitude)}
        else:
            self.realized = {"along_over_amplitude_median": None, "off_over_amplitude_median": None,
                             "max_abs_change": float((after.to(work) - before.to(work)).abs().max())}
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    def remove(self):
        self.handle.remove()


class ResidualNormProbe:
    """Records per-token residual-stream norms at one layer's output (for the amplitude rule).

    Armed only for unpulsed baseline passes, so a pulsed pass can never set the amplitude.
    """

    def __init__(self, layer: torch.nn.Module):
        self.norms = None
        self.armed = False
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.armed:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        self.norms = hidden[0].float().norm(dim=-1).cpu().numpy()
        return None

    def remove(self):
        self.handle.remove()


class HeadProbe:
    """Forward pre-hooks on self_attn.o_proj: the input is the gated head outputs, [1, T, heads*head_dim].

    record: stores the whole unpulsed run, on the model device.
    measure: response = pulsed minus unpulsed over the common window [t1, T), in float32, through
             the sealed operator (line removed, Hann, Fourier sum). Keeps per head the target
             magnitude and the two-sided noise floor, and per noise bin the power summed over
             every head of every measured layer.
    zero_check: counts values whose bits differ from the unpulsed run (sham must give 0).
    """

    def __init__(self, layers, layer_ids, n_heads: int, t1: int):
        self.layer_ids = list(layer_ids)
        self.n_heads = n_heads
        self.t1 = t1
        self.mode = None
        self.baseline = {}
        self.result = {}
        self.cos = self.sin = None
        self.handles = [layers[i].self_attn.o_proj.register_forward_pre_hook(self._make(i)) for i in self.layer_ids]

    def set_frequencies(self, N: int, T: int, device, dtype=torch.float32):
        """Builds the operator for target 1/N over [t1, T). Returns the numpy bases for the input train."""
        W = T - self.t1
        freqs, self.n_below = metrics.analysis_frequencies(N, W)
        cos_b, sin_b = metrics.analysis_basis(freqs, self.t1, W)
        self.cos = torch.as_tensor(cos_b, dtype=dtype, device=device)
        self.sin = torch.as_tensor(sin_b, dtype=dtype, device=device)
        return cos_b, sin_b

    @staticmethod
    def _bits(x):
        return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32) if x.element_size() == 4 \
            else x.view(torch.int64)

    def _make(self, i):
        def hook(module, args):
            x = args[0][0]
            if self.mode == "record":
                self.baseline[i] = x.detach().clone()
            elif self.mode == "zero_check":
                self.result[i] = int((self._bits(x.detach()) != self._bits(self.baseline[i])).sum())
            elif self.mode == "measure":
                work = self.cos.dtype
                diff = x.detach()[self.t1:].to(work) - self.baseline[i][self.t1:].to(work)
                re = self.cos.T @ diff
                im = -(self.sin.T @ diff)
                power = (re * re + im * im).reshape(re.shape[0], self.n_heads, -1).sum(-1)    # [F, heads]
                mag = power.sqrt()
                nb = 1 + self.n_below
                floor = mag[1:nb].quantile(0.5, dim=0)
                if nb < mag.shape[0]:
                    floor = torch.maximum(floor, mag[nb:].quantile(0.5, dim=0))
                self.result[i] = {"target": mag[0].cpu().numpy().astype(np.float64),
                                  "floor": floor.cpu().numpy().astype(np.float64),
                                  "bin_power": power[1:].sum(-1).cpu().numpy().astype(np.float64)}
            return None
        return hook

    def collect(self) -> dict:
        """{"target": [layers, heads], "floor": [layers, heads], "bin_power": [bins] summed over layers and heads,
        "n_below": noise bins below k0 (they come first)}."""
        r = [self.result[i] for i in self.layer_ids]
        self.result = {}
        return {"target": np.stack([x["target"] for x in r]), "floor": np.stack([x["floor"] for x in r]),
                "bin_power": np.sum([x["bin_power"] for x in r], axis=0), "n_below": self.n_below}

    def remove(self):
        for h in self.handles:
            h.remove()


class GlobalKnob:
    """Scales post_attention_layernorm output in global layers by g (used for C2, not C1)."""

    def __init__(self, layers, global_ids):
        self.g = 1.0
        self.handles = [layers[i].post_attention_layernorm.register_forward_hook(self._hook) for i in global_ids]

    def _hook(self, module, args, output):
        if self.g == 1.0:
            return None
        return output * self.g

    def remove(self):
        for h in self.handles:
            h.remove()
