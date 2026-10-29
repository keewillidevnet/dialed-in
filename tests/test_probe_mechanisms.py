"""Tests for the exploratory mechanism probe: statistics on known inputs, and both parts end to end on a small model
with Glimmer's layer layout (attention that returns weights only in eager mode, an output gate, sandwich norms, MLP)."""
import json
import math
import types

import numpy as np
import torch
from torch import nn

from dialedin import probe_mechanisms as M
from dialedin import pilot_c5 as P

D, HEADS = 64, 32


def test_fast_fraction():
    t = torch.arange(2048, dtype=torch.float32)
    assert M.fast_fraction(torch.cos(2 * math.pi * t / 4)) > 0.99
    assert M.fast_fraction(torch.cos(2 * math.pi * t / 256)) < 0.01
    both = torch.stack([torch.cos(2 * math.pi * t / 4), torch.cos(2 * math.pi * t / 256)], 1)
    assert abs(M.fast_fraction(both) - 0.5) < 0.01


def test_attention_stats():
    L, q0 = 64, 32
    prev = torch.zeros(1, 2, L, L)
    for i in range(L):
        prev[0, :, i, max(i - 1, 0)] = 1.0
    s = M.attention_stats(prev, q0)
    assert np.allclose(s["mean_distance"], 1) and np.allclose(s["entropy"], 0, atol=1e-5)
    assert np.allclose(s["near_mass"], 1) and np.allclose(s["first_token_mass"], 0) and np.allclose(s["peak_mass"], 1)
    sink = torch.zeros(1, 1, L, L); sink[0, 0, :, 0] = 1.0
    s = M.attention_stats(sink, q0)
    assert np.allclose(s["first_token_mass"], 1) and np.allclose(s["mean_distance"], np.arange(q0, L).mean())


class Attn(nn.Module):
    def __init__(self, cfg, local):
        super().__init__()
        self.cfg, self.is_local_attention = cfg, local
        self.window = 31 if local else 10 ** 6
        self.q_proj, self.v_proj = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.gate_proj, self.o_proj = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)

    def forward(self, x):
        L = x.shape[1]
        i, j = torch.arange(L)[:, None], torch.arange(L)[None, :]
        mask = (j <= i) & (j > i - self.window)
        v = self.v_proj(x)
        if self.cfg._attn_implementation == "eager":
            w = mask.float() / mask.float().sum(1, keepdim=True)
            w = w[None, None].expand(1, HEADS, L, L)
            out = torch.einsum("hij,bjhd->bihd", w[0], v.reshape(1, L, HEADS, -1)).reshape(1, L, D)
        else:
            w = None
            out = (mask.float() / mask.float().sum(1, keepdim=True)) @ v[0]
            out = out[None]
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out), w


class RMS(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Layer(nn.Module):
    def __init__(self, cfg, local):
        super().__init__()
        self.self_attn = Attn(cfg, local)
        self.input_layernorm, self.post_attention_layernorm = RMS(), RMS()
        self.pre_feedforward_layernorm, self.post_feedforward_layernorm = RMS(), RMS()
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, h):
        h = h + 0.1 * self.post_attention_layernorm(self.self_attn(self.input_layernorm(h))[0])
        return h + 0.1 * self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))


class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.config = types.SimpleNamespace(_attn_implementation="sdpa", hidden_size=D)
        self.embed = nn.Embedding(1000, D)
        self.layers = nn.ModuleList([Layer(self.config, i != 3) for i in range(7)])

    def forward(self, input_ids, use_cache=False):
        h = self.embed(input_ids)
        for l in self.layers:
            h = l(h)
        return types.SimpleNamespace(last_hidden_state=h)


def test_both_parts_end_to_end(monkeypatch):
    monkeypatch.setattr(M, "TRANSFER_TOKENS", 4096)
    monkeypatch.setattr(M, "RESIDUAL_POINTS", (2, 3, 4, 5))
    monkeypatch.setattr(M, "SEEDS", (1,))
    dec = Dec()
    for p in dec.parameters():
        p.requires_grad_(False)
    M.check_layout(dec.layers)
    names, params = P.attach_lora(dec.layers, [2, 3, 4, 5], 2, 4.0)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(0, 0.5)
    rng = np.random.default_rng(0)
    ids = [torch.tensor([rng.integers(0, 1000, 4096).tolist()])]
    periods, tr = M.transfer(dec, dec.layers, ids, D)
    assert periods == [2, 4, 16, 64, 256, 1024] and P.LoRALinear.scale == 0.0
    assert set(tr) >= {"c5_curve", "L2.attn", "L5.mlp", "L3.out", "L2.heads"}
    assert len(tr["L2.heads"]["S2"]["tilt"]) == HEADS and np.isfinite(tr["c5_curve"]["S0"]["tilt"])
    assert tr["L2.attn"]["S0"]["gains"] != tr["L2.attn"]["S2"]["gains"]      # the adapter changes layer 2
    how, op = M.operating_point(dec, dec, dec.layers, [ids[0][:, :512]], q0=256)
    assert how == "config" and dec.config._attn_implementation == "eager"
    r = op["S0"][2]
    assert len(r["mean_distance"]) == HEADS and abs(sum(r["head_output_share"]) - 1) < 1e-5
    assert np.allclose(r["mean_distance"], 15, atol=1e-3)                     # uniform over a 31-token window
    assert 0 <= r["gate_fast"] <= 1 and "post_feedforward_layernorm_scale_fast" in r
    s = M.summarize(periods, tr, op, [3])
    assert s["operating_point"]["layer3_global"]["mean_distance"]["S0"] > 100
    json.dumps(s)


def test_sdpa_mode_refused():
    dec = Dec()
    st = M.Stats(dec.layers, 8)
    try:
        import pytest
        with pytest.raises(SystemExit, match="eager"):
            dec(torch.tensor([list(range(32))]))
    finally:
        st.remove()
