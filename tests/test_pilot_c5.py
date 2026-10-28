"""CPU checks for the C5 feasibility pilot: LoRA dial at scale 0, lock-in gains, tilt fit, a short training run."""
import math

import numpy as np
import torch

from dialedin import pilot_c5 as P
from test_c4 import ToyLM, D


def test_side_bins_clear_of_tones():
    tb = P.bins(P.PERIODS)
    for k0 in tb:
        sb = P.side_bins(k0, tb)
        assert len(sb) == 2 and all(0 < k <= P.W // 2 for k in sb)
        assert all(abs(k - b) >= 2 for k in sb for b in tb)


def test_slope_recovers_power_law():
    g = torch.tensor([n ** 0.14 for n in P.PERIODS], dtype=torch.float32)
    assert abs(float(P.slope(P.PERIODS, g)) - 0.14) < 1e-5


def test_lora_scale_zero_is_bitwise():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8).to(torch.bfloat16)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    ref = lin(x)
    w = P.LoRALinear(lin, 4, 8.0)
    with torch.no_grad():
        w.B.normal_()
    P.LoRALinear.scale = 0.0
    assert torch.equal(w(x), ref)
    P.LoRALinear.scale = 1.0
    assert not torch.equal(w(x), ref)
    P.LoRALinear.scale = 0.0


def test_input_magnitude_matches_lockin():
    rng = np.random.default_rng(0)
    w, _ = P.tones(P.PERIODS, 1.0, rng)
    B = P.basis(P.PERIODS, "cpu")
    mags = P.input_mags(w, P.PERIODS, B)
    a = 1.0
    expect = a * P.W / 4                        # Hann sum is W/2; a cosine puts half its amplitude in each sign
    for n, m in zip(P.PERIODS, mags.tolist()):
        if n != 2:
            assert abs(m / expect - 1) < 0.02, (n, m, expect)
        else:                                   # Nyquist, phase 0: the full amplitude lands in the cos row
            assert abs(m / (2 * expect) - 1) < 0.02


def test_training_moves_toy_tilt():
    lm = ToyLM()
    for p in lm.parameters():
        p.requires_grad_(False)
    layers = lm.layers
    rig = P.Rig(layers, range(2, len(layers)), heads=4)
    ids = torch.randint(1, 900, (1, P.T))
    names, params = P.attach_lora(layers, [2], r=2, alpha=4.0)
    assert names and all(p.requires_grad for p in params)
    norm = 1.0
    rng = np.random.default_rng(0)
    per = [3, 48, 768]                         # multiples of 3: the toy window averages never null them
    d = P.direction(1, D, "cpu")
    w, _ = P.tones(per, 0.1 * norm, rng)
    P.LoRALinear.scale = 1.0
    with torch.no_grad():
        g0, nz0, _ = P.run_pass(lm, rig, ids, per, w, d)
    t0 = float(P.slope(per, g0))
    assert math.isfinite(t0) and (g0 / nz0).min() > 3
    target = t0 - 0.05
    opt = torch.optim.Adam(params, lr=3e-2)
    undo = P.checkpoint_layers(layers)
    for _ in range(15):
        g, _, _ = P.run_pass(lm, rig, ids, per, w, d)
        loss = (P.slope(per, g) - target) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    undo()
    with torch.no_grad():
        g1, _, _ = P.run_pass(lm, rig, ids, per, w, d)
        P.LoRALinear.scale = 0.0
        gz, _, _ = P.run_pass(lm, rig, ids, per, w, d)
    t1 = float(P.slope(per, g1))
    assert abs(t1 - target) < abs(t0 - target) * 0.75, (t0, t1, target)     # moved at least a quarter of the way
    assert abs(float(P.slope(per, gz)) - t0) < 1e-6           # scale 0 is the base model again
    rig.remove()


def test_kl_is_zero_at_base_and_has_gradient():
    lm = ToyLM()
    f = P.logits_fn(lm.head, 0.5, 3.0)
    hb = torch.randn(P.T, D)
    ha = hb.clone().requires_grad_(True)
    kl = P.kl_to_base(ha, hb, f)
    assert abs(float(kl)) < 1e-6
    kl2 = P.kl_to_base(ha + 0.1 * torch.randn(P.T, D), hb, f)
    kl2.backward()
    assert float(kl2) > 0 and ha.grad is not None and float(ha.grad.abs().sum()) > 0


def test_evaluate_single_tones_on_toy():
    lm = ToyLM()
    rig = P.Rig(lm.layers, range(2, len(lm.layers)), heads=4)
    ids = torch.randint(1, 900, (1, P.T))
    P.attach_lora(lm.layers, [2], r=2, alpha=4.0)
    rows = P.evaluate(lm, rig, ids, [1], [0.0, 1.0], 1.0, D)
    assert len(rows) == 2 and all(len(r["gains"]) == len(P.PERIODS) for r in rows)
    assert rows[0]["tilt"] == rows[1]["tilt"]            # B starts at zero, so scale 1 equals scale 0
    assert rig.delta is None and rig.B is None
    rig.remove()


def test_configure_lengths():
    try:
        assert P.configure(6144) == P.PERIODS and P.W == 4096
        assert P.configure(4096) == (2, 4, 16, 64, 256, 1024) and P.T1 == 1024
    finally:
        assert P.configure(8192) == P.PERIODS and P.W == 6144


def test_perplexity_matches_direct():
    lm = ToyLM()
    f = P.logits_fn(lm.head, 0.5, 3.0)
    ids = torch.randint(1, 900, (1, P.T))
    with torch.no_grad():
        h = lm(ids).last_hidden_state[0]
        ppl = P.perplexity(h, ids, f)
        lp = f(h[P.T1:P.T - 1]).log_softmax(-1)
        direct = math.exp(-float(lp.gather(1, ids[0, P.T1 + 1:, None]).mean()))
    assert abs(ppl / direct - 1) < 1e-5


def test_evaluate_with_periods_and_perplexity():
    lm = ToyLM()
    rig = P.Rig(lm.layers, range(2, len(lm.layers)), heads=4)
    ids = torch.randint(1, 900, (1, P.T))
    rows = P.evaluate(lm, rig, ids, [5], [0.0], 1.0, D, periods=(3, 48, 768), logits=P.logits_fn(lm.head, None, None))
    assert len(rows[0]["gains"]) == 3 and rows[0]["perplexity"] > 1
    rig.remove()
