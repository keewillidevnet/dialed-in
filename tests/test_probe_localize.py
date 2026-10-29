"""Tests for the exploratory localization probe on the small Glimmer-shaped model from test_probe_mechanisms."""
import json

import numpy as np
import torch
from torch import nn

from dialedin import probe_localize as L
from dialedin import pilot_c5 as P
from test_probe_mechanisms import Dec, D


def test_name_parsing():
    assert L.layer_of("layers.12.self_attn.q_proj") == 12 and L.is_attn("layers.2.self_attn.o_proj")
    assert not L.is_attn("layers.2.mlp.0")


def test_localize_end_to_end(monkeypatch):
    monkeypatch.setattr(L, "TOKENS", 4096)
    monkeypatch.setattr(L, "SEEDS", (1,))
    dec = Dec()
    for p in dec.parameters():
        p.requires_grad_(False)
    names, params = P.attach_lora(dec.layers, [2, 3, 4, 5], 2, 4.0)
    with torch.no_grad():
        for b in params[1::2]:
            b.normal_(0, 0.5)
    before = [b.clone() for b in params[1::2]]
    head = nn.Linear(D, 1000, bias=False)
    logits = lambda h: head(h).float()
    ids = [torch.tensor([np.random.default_rng(0).integers(0, 1000, 4096).tolist()])]
    conds = {"all": L.CONDITIONS["all"], "none": lambda n: False, "L2_attention": L.CONDITIONS["L2_attention"],
             "all_but_L2": L.CONDITIONS["all_but_L2"]}
    periods, out = L.run(dec, dec.layers, names, params, ids, D, logits=logits, conditions=conds, log=lambda m: None)
    assert all(torch.equal(a, b) for a, b in zip(before, params[1::2])) and P.LoRALinear.scale == 0.0
    assert out["none"]["tilt"] == out["base"]["tilt"] and out["none"]["fraction_of_full"] == 0.0
    assert abs(out["all"]["fraction_of_full"] - 1.0) < 1e-12 and out["all"]["modules_on"] == len(names)
    n2 = sum(1 for n in names if L.layer_of(n) == 2 and L.is_attn(n))
    assert out["L2_attention"]["modules_on"] == n2 == 4                 # q, v, gate, o in the toy
    assert out["all_but_L2"]["modules_on"] == len(names) - sum(1 for n in names if L.layer_of(n) == 2)
    assert out["all"]["tilt"] != out["base"]["tilt"] and abs(out["base"]["perplexity_ratio"] - 1) < 1e-12
    json.dumps(out)


def test_deep_set_selects_the_right_modules():
    names = [f"layers.{i}.self_attn.{m}" for i in (2, 3) for m in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")]
    names += ["layers.2.mlp.gate_proj", "layers.2.mlp.up_proj"]
    pick = lambda c: [n for n in names if L.DEEP[c](n)]
    assert pick("L2_qk_only") == ["layers.2.self_attn.q_proj", "layers.2.self_attn.k_proj"]
    assert pick("L2_gate_only") == ["layers.2.self_attn.gate_proj"]           # not the MLP's gate_proj
    assert pick("L2_attention_minus_v") == ["layers.2.self_attn.q_proj", "layers.2.self_attn.k_proj",
                                            "layers.2.self_attn.gate_proj", "layers.2.self_attn.o_proj"]
    assert len(pick("L2_attention")) == 5 and len(pick("all")) == len(names)
    assert len(L.DEEP) == 10
