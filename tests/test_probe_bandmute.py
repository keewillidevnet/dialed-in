"""Tests for the exploratory band-mute probe: token categories, bands, controls, muting, and the summary."""
import json

import numpy as np
import torch

from dialedin import probe_bandmute as B
from test_probe_mechanisms import Dec


def test_categories():
    #           0     1     2      3     4     5    6     7     8      9
    dec = ["The", " cat", "s", " sat", ".", " A", " dog", " ran", " The", " cat"]
    ids = [10, 11, 12, 13, 14, 15, 16, 17, 10, 11]
    c = B.categories(ids, dec, score_from=2, far=3)
    assert c[2] == "continuation"                       # "s" after "cat"
    assert c[3] == "other" and c[4] == "other"
    ids2 = [1, 2, 3, 9, 9, 9, 9, 9, 1, 2, 3]
    c2 = B.categories(ids2, ["x"] * 11, score_from=2, far=4)
    assert c2[10] == "far_repeat"                        # trigram (1, 2, 3) seen 8 back; prefix (1, 2) last seen 8 back
    near = [1, 2, 3, 1, 2, 3]
    assert B.categories(near, [" a"] * 6, score_from=2, far=4)[5] == "other"   # repeat only 3 back: not far


def test_bands_and_controls():
    corner = [[2, 4, 8, 16, 32, 64, 256, 512] + [None] * 24, [None] * 32, [3, 300, 1024, 48] + [None] * 28]
    k, b = B.bands(corner, [2, 3, 4])
    assert k == 4 and all(len(v) == 4 for v in b.values())
    assert set(b["short"]).isdisjoint(b["long"]) and max(corner[l - 2][h] for l, h in b["short"]) <= 16
    assert min(corner[l - 2][h] for l, h in b["long"]) >= 256
    ina = set(x for v in b.values() for x in v)
    ctl, moved = B.matched_control(b["short"], ina, 1, layers=range(2, 5))
    assert len(set(ctl)) == 4 and not set(ctl) & ina and moved == 0
    assert B.matched_control(b["short"], ina, 1, layers=range(2, 5)) == (ctl, 0)          # seeded


def test_muter_and_run():
    dec = Dec()
    for p in dec.parameters():
        p.requires_grad_(False)
    ids = list(np.random.default_rng(0).integers(0, 1000, 1500))
    head = torch.nn.Linear(64, 1000, bias=False)
    logits = lambda h: head(h).float()
    base = B.token_losses(dec, logits, ids)
    m = B.Muter(dec.layers)
    seen = []
    hk = dec.layers[2].self_attn.o_proj.register_forward_pre_hook(lambda mod, a: seen.append(a[0].clone()))
    m.set([(2, 0), (2, 5)])
    muted = B.token_losses(dec, logits, ids)
    hk.remove()
    x = seen[0]
    assert not x[..., 0:2].any() and not x[..., 10:12].any() and x[..., 2:4].any()
    m.set([]); assert np.array_equal(B.token_losses(dec, logits, ids), base) and not np.array_equal(muted, base)
    m.remove()
    cats = [B.categories(ids, [" w"] * len(ids))]
    conds = {"base": [], "short": [(2, 0)], "mid": [(3, 1)], "long": [(4, 2)]}
    for n in ("short", "mid", "long"):
        for sd in B.CONTROL_SEEDS:
            conds[f"control_{n}_{sd}"] = [(5, sd)]
    res = B.run(dec, dec.layers, logits, [ids], cats, conds, log=lambda s: None)
    assert res["base"]["all"]["tokens"] == len(ids) - B.SCORE_FROM
    s = B.summarize(res, 1)
    assert set(s["prediction_check"]) and abs(s["relative_change"]["short"]["all"]) > 0
    json.dumps(s, default=bool)
