"""Head knob for C3: scales chosen heads' gated outputs at the input of self_attn.o_proj by g.

A forward pre-hook on o_proj multiplies each chosen head's 128-dim slice by g. At g = 1 the hook
returns nothing, so the model is unchanged bit for bit. Batch size 1 only.
"""


class HeadKnob:
    def __init__(self, layers, heads, head_dim: int = 128):
        """heads: list of (layer, head) pairs."""
        self.g = 1.0
        self.head_dim = head_dim
        self.by_layer = {}
        for layer, head in heads:
            self.by_layer.setdefault(int(layer), []).append(int(head))
        self.handles = [layers[l].self_attn.o_proj.register_forward_pre_hook(self._make(l)) for l in sorted(self.by_layer)]

    def _make(self, layer):
        heads = sorted(self.by_layer[layer])

        def hook(module, args):
            if self.g == 1.0:
                return None
            x = args[0].clone()
            for h in heads:
                x[..., h * self.head_dim:(h + 1) * self.head_dim] *= self.g
            return (x,) + tuple(args[1:])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
