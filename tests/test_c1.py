"""Tests for the C1 harness and analysis on toy models with known frequency responses. CPU only."""
import json
import os

import numpy as np
import pytest
import torch
from torch import nn

from dialedin import metrics, analyze_c1, run_c1, glimmer
from dialedin.hooks import HeadProbe, PulseInjector, GlobalKnob

D, HEADS, HD = 64, 4, 16
GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
T, T1 = metrics.WINDOW_END, metrics.WINDOW_START


class Attn(nn.Module):
    def __init__(self, window, local):
        super().__init__()
        self.window, self.is_local_attention = window, local
        self.o_proj = nn.Linear(HEADS * HD, D, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(D)[:, : HEADS * HD] * 0.01)

    def forward(self, h):                       # causal moving average: a known filter
        c = torch.cumsum(h, dim=1)
        shifted = torch.zeros_like(c)
        shifted[:, self.window:] = c[:, :-self.window]
        return self.o_proj(((c - shifted) / self.window)[..., : HEADS * HD])


class Layer(nn.Module):
    def __init__(self, window, local):
        super().__init__()
        self.self_attn = Attn(window, local)
        self.post_attention_layernorm = nn.Identity()

    def forward(self, h):
        return h + self.post_attention_layernorm(self.self_attn(h))


class Toy(nn.Module):
    def __init__(self, windows=(1, 1, 8, 64)):
        super().__init__()
        self.layers = nn.ModuleList([Layer(w, i != 3) for i, w in enumerate(windows)])

    def forward(self, h):
        for l in self.layers:
            h = l(h)
        return h


def theory(w, N):
    f = 1.0 / N
    return abs(np.sin(np.pi * f * w) / (w * np.sin(np.pi * f))) if (f * w) % 1 else 0.0


def run_toy(N, windows=(1, 1, 8, 64), aperiodic=False, amplitude=1e-3):
    torch.manual_seed(0)
    model = Toy(windows).double()
    base = torch.randn(1, T, D, dtype=torch.float64)
    d = torch.zeros(D, dtype=torch.float64); d[0] = 1.0
    inj = PulseInjector(model.layers[1], d)
    probe = HeadProbe(model.layers, [2, 3], HEADS, T1)
    probe.mode = "record"; model(base)
    pos = metrics.periodic_positions(N, T)
    if aperiodic:
        pos = metrics.aperiodic_positions(len(pos), T)
    cos_b, sin_b = probe.set_frequencies(N, T, "cpu", dtype=torch.float64)
    inj.arm(pos, amplitude); probe.mode = "measure"; model(base)
    out = probe.collect()
    inp = metrics.train_component(pos, amplitude, cos_b, sin_b, T1, T)
    gain, _ = metrics.gains(out["target"], out["floor"], inp, head_dim=HD)
    return gain, metrics.aggregate_snr(out["target"][:1, :1], out["bin_power"], out["n_below"]), out


# ---------- hooks and operator ----------

def test_sham_bitwise_exact_bf16():
    torch.manual_seed(0)
    model = Toy().to(torch.bfloat16)
    base = torch.randn(1, 2048, D).to(torch.bfloat16)
    base[0, ::16, :] = -0.0                                   # adding zero must not flip -0.0 to +0.0
    inj = PulseInjector(model.layers[1], torch.ones(D) / D ** 0.5)
    probe = HeadProbe(model.layers, [2, 3], HEADS, 1024)
    probe.mode = "record"; model(base)
    inj.arm(metrics.periodic_positions(16, 2048), 0.0); probe.mode = "zero_check"; model(base)
    assert sum(probe.result.values()) == 0


def test_zero_check_is_bitwise():
    probe = HeadProbe([], [], HEADS, 0)
    probe.layer_ids = [0]
    hook = probe._make(0)
    x = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    probe.mode = "record"; hook(None, (x,))
    y = x.clone(); y[0, 1, 2] = -0.0                          # equal as numbers, different bits
    probe.mode = "zero_check"; hook(None, (y,))
    assert probe.result[0] == 1


def test_identity_gain_is_exactly_one():
    # The same operator is applied to response and input, so a pass-through head has gain 1.
    for N in (2, 3, 16, 48, 4096):
        gain, _, _ = run_toy(N, windows=(1, 1, 1, 64))
        assert abs(gain[0, 0] - 1.0) < 1e-9, (N, gain[0, 0])


def test_moving_average_gain_matches_theory():
    for N in (2, 3, 16, 48, 192, 4096):
        gain, _, _ = run_toy(N)
        assert abs(gain[0, 0] - theory(8, N)) < 1e-4 * theory(8, N) + 1e-9, (N, gain[0, 0], theory(8, N))


def test_long_periods_and_2048_window():
    # A 2048-token window passes long periods and notches N = 2048, as a linear filter must.
    # Line removal adds a bias below 0.1 percent when the window holds few periods.
    for N in (4096, 3072, 1536):
        gain, _, _ = run_toy(N, windows=(1, 1, 2048, 64))
        assert abs(gain[0, 0] - theory(2048, N)) < 1e-3 * theory(2048, N), (N, gain[0, 0])
    gain, _, _ = run_toy(2048, windows=(1, 1, 2048, 64))
    assert gain[0, 0] < 1e-8


def test_snr_and_aperiodic_cross_check():
    gp, snr, _ = run_toy(16)
    ga, _, _ = run_toy(16, aperiodic=True)
    assert snr > 1e4
    assert abs(ga[0, 0] / gp[0, 0] - 1) < 0.2


def test_common_window_and_noise_bins():
    W = T - T1
    for N in GRID:
        assert W % N == 0
        k0 = W // N
        b, nb = metrics.noise_bins(N, W)
        assert k0 >= 6 and len(set(b)) == len(b) and nb >= 3
        assert np.all(b[:nb] < k0) and np.all(b[nb:] > k0)
        assert len(b) - nb == (0 if N == 2 else 8)
        r = b % k0
        assert np.all(np.minimum(r, k0 - r) > 1) and np.all((b > 0) & (b <= W / 2))


def test_smooth_drift_does_not_pass_validity():
    # A non-periodic drift must not look like signal at the longest N, where it leaks most.
    W = T - T1
    t = np.arange(T1, T, dtype=np.float64)
    for N in (4096, 3072, 2048, 512):
        f, nb = metrics.analysis_frequencies(N, W)
        c, s = metrics.analysis_basis(f, T1, W)
        for x in (np.log(t), 1 / t, (t / T) ** 2, np.exp(-t / 6000), t ** -0.5):
            v = metrics.component(x, c, s)
            assert v[0] / metrics.two_sided_floor(v[1:], nb) < 1.0, N


def test_operator_removes_lines_and_leaks_little():
    W = T - T1
    for N in (4096, 1536, 64):
        f, _ = metrics.analysis_frequencies(N, W)
        c, s = metrics.analysis_basis(f, T1, W)
        t = np.arange(T1, T, dtype=np.float64)
        line = metrics.component(3.0 + 0.01 * t, c, s)
        assert np.all(line < 1e-6)
        sig = metrics.component(np.cos(2 * np.pi * t / N + 0.7), c, s)
        assert np.median(sig[1:]) / sig[0] < 1e-4


def test_train_component_matches_dense():
    W = T - T1
    c, s = metrics.analysis_basis(metrics.analysis_frequencies(64, W)[0], T1, W)
    pos = metrics.periodic_positions(64, T)
    x = np.zeros(W); x[pos[pos >= T1] - T1] = 0.3
    assert abs(metrics.train_component(pos, 0.3, c, s, T1, T) - metrics.component(x, c, s)[0]) < 1e-9


def test_noise_correction_uses_chi_square_median():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((20000, 2 * HD))
    mags = np.sqrt((noise ** 2).sum(1))
    floor = np.median(mags)
    mean_power = (mags ** 2).mean()
    assert abs(floor ** 2 / metrics.median_over_mean_power(2 * HD) / mean_power - 1) < 0.01


def test_knob():
    torch.manual_seed(0)
    model = Toy()
    x = torch.randn(1, 256, D)
    ref = model(x)
    knob = GlobalKnob(model.layers, [3])
    assert torch.equal(model(x), ref)
    knob.g = 0.0
    h = x
    for l in model.layers[:3]:
        h = l(h)
    assert torch.allclose(model(x), h)


# ---------- outcomes ----------

def classify(curves, valid=None):
    v = np.ones(len(GRID), bool) if valid is None else valid
    if not isinstance(curves, dict):
        curves = {s: curves for s in "ABC"}
    return metrics.classify_c1(GRID, {s: c / np.nanmax(c[v]) for s, c in curves.items()}, {s: v for s in curves})


G = np.array(GRID, dtype=np.float64)


def test_falling_gain_is_high_pass():
    o = classify(1 / (1 + (G / 300) ** 2))
    assert o["outcome"] == "HP" and 250 < o["per_set"]["A"]["cutoff_N"] < 350


def test_rising_gain_is_low_pass():
    o = classify(G ** 2 / (G ** 2 + 300.0 ** 2))
    assert o["outcome"] == "LP" and 250 < o["per_set"]["A"]["cutoff_N"] < 350


def test_flat_then_rising_is_low_pass():
    # The expected shape: each pulse also reaches its own token, so short N sit on a floor.
    c = 1 + 9 * (G / 1000) ** 2 / (1 + (G / 1000) ** 2)
    assert classify(c)["outcome"] == "LP"


def test_resonance():
    o = classify(np.exp(-((np.log(G) - np.log(128)) ** 2) / 0.5) + 0.1)
    assert o["outcome"] == "R" and o["per_set"]["A"]["peak_N"] == 128


def test_truncated_rise_is_not_resonance():
    # A rising curve whose long-N points failed validity must not look like a peak.
    c = G ** 2 / (G ** 2 + 3000.0 ** 2)
    v = G <= 512
    o = classify(np.where(v, c, np.nan), v)
    assert o["outcome"] != "R"
    assert o["per_set"]["A"]["valid_right"] == 0


def test_band_pass_hump_is_not_a_cutoff():
    # Short end at 0.3 of the maximum: that end is not a passband, so no HP cutoff.
    c = np.interp(np.log(G), np.log([2, 64, 512, 4096]), [0.3, 1.0, 1.0, 0.1])
    o = classify(c)
    assert o["per_set"]["A"]["direction"] is None and o["outcome"] not in ("HP", "LP")


def test_rise_then_partial_fall_is_not_low_pass():
    c = np.interp(np.log(G), np.log([2, 8, 4096]), [0.2, 1.0, 0.45])
    o = classify(c)
    assert o["per_set"]["A"]["direction"] is None and o["outcome"] != "LP"


def test_one_low_edge_point_is_not_resonance():
    c = np.interp(np.log(G), np.log([2, 4, 4096]), [1.0, 1.0, 0.1]); c[0] = 0.5
    o = classify(c)
    assert o["outcome"] != "R" and o["per_set"]["A"]["low_left"] == 1


def test_single_notch_is_not_a_cutoff():
    c = np.ones(len(G)); c[GRID.index(2048)] = 0.3
    o = classify(c)
    assert o["outcome"] == "S" and o["per_set"]["A"]["direction"] is None      # a replicated notch is structure, not a kill


def test_gentle_slope_is_tilt():
    o = classify(1 - 0.03 * np.log2(G))
    assert o["outcome"] == "T" and o["tilt_direction"] == "HP"


def test_mixed_directions_are_not_a_cutoff():
    o = classify({"A": 1 / (1 + (G / 300) ** 2), "B": 1 / (1 + (G / 300) ** 2), "C": G ** 2 / (G ** 2 + 300.0 ** 2)})
    assert o["outcome"] not in ("HP", "LP", "R")


def test_flat_and_too_few_valid():
    assert classify(1 + np.random.default_rng(0).normal(0, 0.01, len(G)))["outcome"] == "F"
    few = np.zeros(len(G), bool); few[:5] = True
    o = classify(1 / (1 + (G / 300) ** 2), few)
    assert o["outcome"] == "U" and o["inconclusive_reason"]          # underpowered is inconclusive, never F


def test_power_laws_are_tilts_not_cutoffs():
    for c, d in ((G ** 0.3, "LP"), (G ** -0.15, "HP"), (G ** 0.6, "LP"), (G ** -0.5, "HP")):
        o = classify(c)
        assert o["outcome"] == "T" and o["tilt_direction"] == d, (d, o["outcome"])


def test_first_order_filters_are_cutoffs():
    o = classify(1 / np.sqrt(1 + (G / 300) ** 2))
    assert o["outcome"] == "HP" and 450 < o["per_set"]["A"]["cutoff_N"] < 600
    o = classify(G / np.sqrt(G ** 2 + 300.0 ** 2))
    assert o["outcome"] == "LP" and 150 < o["per_set"]["A"]["cutoff_N"] < 200


def test_unreplicated_shape_is_inconclusive():
    rng = np.random.default_rng(3)
    curves = {s: 0.3 + rng.random(len(G)) for s in "ABC"}
    assert classify(curves)["outcome"] == "U"


def test_cutoff_directions_exclusive():
    rng = np.random.default_rng(1)
    for _ in range(500):
        c = np.abs(rng.normal(1, 0.6, len(G)))
        co = metrics.cutoff(GRID, c / c.max(), np.ones(len(G), bool))
        assert co["direction"] in ("HP", "LP", None)


def test_box_filter_null():
    b = metrics.box_filter_null(GRID, 2048)
    assert b[GRID.index(2048)] == 0 and b[GRID.index(1024)] == 0 and b[GRID.index(4096)] > 0.6


# ---------- analysis ----------

SPEC = {"pulse": {"interval_grid_N": GRID},
        "inputs": {"prompt_sets": {"A": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, "B": {6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                                   "C": {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}}}}
REC = {"layers_measured": list(range(2, 52)), "seal_sha256": "seal", "code_sha256": {"x": "y"},
       "platform": {"gpu": "toy"}}


def fake_book(set_name, gid, curve_fn, seed, snr=50.0, a=0.1, low_snr_above=None):
    rng = np.random.default_rng(seed)
    per = {}
    for N in GRID:
        inp = 7.0
        target = curve_fn(N) * inp * (1 + 0.02 * rng.standard_normal((50, 32)))
        s = 1.0 if (low_snr_above and N > low_snr_above) else snr
        bin_power = np.full(16, float(np.sum(target ** 2)) / s ** 2)
        nb = 16 if N == 2 else 8
        floor = np.full((50, 32), np.sqrt(bin_power[0] / 1600))
        cell = {"input_mag": inp, "target": target.tolist(), "floor": floor.tolist(), "bin_power": bin_power.tolist(),
                "n_below": nb,
                "realized_pulse": {"off_over_amplitude_median": 0.01}}
        cell["aperiodic"] = {"input_mag": inp, "target": target.tolist(), "floor": floor.tolist(),
                             "bin_power": bin_power.tolist(), "n_below": nb}
        per[str(N)] = cell
    return {"set": set_name, "gutenberg_id": gid, "a": a, "sham_differences_open": 0, "sham_differences_close": 0,
            "global_layers": list(range(3, 52, 4)), "per_N": per, "record": dict(REC)}


def write_run(tmp_path, curve_fn, **kw):
    for i, s in enumerate("ABC"):
        for j in range(5):
            gid = 5 * i + j + 1
            json.dump(fake_book(s, gid, curve_fn, gid, **kw), open(tmp_path / f"book_{s}_{gid}.json", "w"))
    json.dump({"phase": "A", "rows": [], "chosen_a": 0.1, "sham_differences": 0, "record": dict(REC)},
              open(tmp_path / "phaseA.json", "w"))
    json.dump({"utc": "x", "restart_reason": None}, open(tmp_path / "book_A_1.ATTEMPT.20261027T000000Z.json", "w"))
    json.dump({"what": "book_B_6", "error": "x"}, open(tmp_path / "book_B_6.FAILED.20261027T000000Z.json", "w"))


def test_analysis_end_to_end(tmp_path):
    write_run(tmp_path, lambda N: 1 / (1 + (N / 300) ** 2))
    r = analyze_c1.analyze(str(tmp_path), SPEC, "seal", {"x": "y"})
    assert r["C1"]["outcome"] == "HP" and r["complete"]
    lo, hi = r["bootstrap"]["A"]["cutoff_N_95_by_direction"]["HP"]
    assert lo <= r["C1"]["per_set"]["A"]["cutoff_N"] <= hi
    assert r["bootstrap"]["A"]["direction_fractions"]["HP"] > 0.9
    assert "C1b_descriptive" in r and r["diagnostic_noise_corrected_outcome"] == "HP"
    assert abs(r["curves"]["A"]["snr_median_over_books"][0] - 50) < 1
    json.dumps(r)


def test_analysis_validity_drops_low_snr_points(tmp_path):
    write_run(tmp_path, lambda N: N ** 2 / (N ** 2 + 3000.0 ** 2), low_snr_above=512)
    r = analyze_c1.analyze(str(tmp_path), SPEC)
    assert r["C1"]["per_set"]["A"]["invalid_N"] == [g for g in GRID if g > 512]
    assert r["C1"]["outcome"] != "R"


def test_partial_flag_refused_on_complete_run(tmp_path):
    write_run(tmp_path, lambda N: 1 / (1 + (N / 300) ** 2))
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC, partial=True)


def test_analysis_refuses_bad_runs(tmp_path):
    write_run(tmp_path, lambda N: 1 / (1 + (N / 300) ** 2))
    (tmp_path / "book_C_15.json").unlink()
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC)
    r = analyze_c1.analyze(str(tmp_path), SPEC, partial=True)
    assert r["C1"]["outcome"] is None and not r["complete"]
    (tmp_path / "phaseA.json").rename(tmp_path / "phaseA.bak")
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC, partial=True)
    (tmp_path / "phaseA.bak").rename(tmp_path / "phaseA.json")
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), dict(SPEC, inputs=dict(SPEC["inputs"], text_sha256={1: "other"})), partial=True)
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC, seal_hash="other", partial=True)
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC, code={"x": "z"}, partial=True)
    b = json.load(open(tmp_path / "book_A_1.json")); b["a"] = 0.05
    json.dump(b, open(tmp_path / "book_A_1.json", "w"))
    with pytest.raises(SystemExit):
        analyze_c1.analyze(str(tmp_path), SPEC, partial=True)


# ---------- runner ----------

def test_write_json_is_atomic_and_never_overwrites(tmp_path):
    p = str(tmp_path / "x.json")
    run_c1.write_json(p, {"a": 1})
    assert not os.path.exists(p + ".tmp")
    with pytest.raises(SystemExit):
        run_c1.write_json(p, {"a": 2})
    assert json.load(open(p)) == {"a": 1}


def test_ledger_is_append_only_wall_time(tmp_path):
    l = run_c1.Ledger(str(tmp_path), "abcdef123456789", 3600)
    l.add("one"); l.add("two")
    lines = open(l.path).read().splitlines()
    assert len(lines) == 2 and l.path.endswith(".jsonl")
    l2 = run_c1.Ledger(str(tmp_path), "abcdef123456789", 3600)
    assert abs(l2.seconds - l.seconds) < 0.01
    with pytest.raises(SystemExit):
        l2.check(3601)


def test_check_code(tmp_path):
    f = tmp_path / "m.py"; f.write_text("x = 1\n")
    spec = {"analysis": {"code_sha256": {"m.py": glimmer.sha256_file(str(f))}}}
    assert glimmer.check_code(str(tmp_path), spec)
    f.write_text("x = 2\n")
    with pytest.raises(SystemExit):
        glimmer.check_code(str(tmp_path), spec)


def test_runs_under_seal_states(tmp_path):
    spec = {"inputs": {"prompt_sets": {"A": {1: 0}}}}
    cases = (("r1", {"book_A_1.json": {}}), ("r2", {"book_A_1.FAILED.x.json": {}}), ("r3", {}),
             ("r4", {"phaseA.json": {"stopped": "underpowered"}}))
    for i, (rid, files) in enumerate(cases):
        (tmp_path / rid).mkdir()
        run_c1.write_json(str(tmp_path / rid / "STARTED.json"), {"seal_sha256": "s", "utc": f"2026-10-2{i}"})
        for f, body in files.items():
            run_c1.write_json(str(tmp_path / rid / f), body)
    states = {r["run_id"]: r["state"] for r in run_c1.runs_under_seal(str(tmp_path), "s", spec)}
    assert states == {"r1": "complete", "r2": "failed", "r3": "open", "r4": "stopped"}
    assert run_c1.runs_under_seal(str(tmp_path), "other", spec) == []
    (tmp_path / "x").mkdir()
    run_c1.write_json(str(tmp_path / "x" / "STARTED.json"), {"other_experiment": True})
    assert len(run_c1.runs_under_seal(str(tmp_path), "s", spec)) == 4


def test_admission_rules(tmp_path):
    spec = {"inputs": {"prompt_sets": {"A": {1: 0}}}}
    raw = str(tmp_path)
    run_c1.admit("A", raw, str(tmp_path / "r1"), "s", spec)
    with pytest.raises(SystemExit):                                   # same id twice
        run_c1.admit("A", raw, str(tmp_path / "r1"), "s", spec, "x")
    with pytest.raises(SystemExit):                                   # second run without a reason
        run_c1.admit("A", raw, str(tmp_path / "r2"), "s", spec)
    run_c1.write_json(str(tmp_path / "r1" / "phaseA.json"), {"stopped": None})
    run_c1.admit("A", raw, str(tmp_path / "r2"), "s", spec, "crash on r1")
    run_c1.write_json(str(tmp_path / "r2" / "phaseA.json"), {"stopped": None})
    with pytest.raises(SystemExit):                                   # older run may not continue
        run_c1.admit("B", raw, str(tmp_path / "r1"), "s", spec)
    run_c1.admit("B", raw, str(tmp_path / "r2"), "s", spec)
    with pytest.raises(SystemExit):                                   # Phase B needs STARTED.json
        (tmp_path / "r9").mkdir(); run_c1.write_json(str(tmp_path / "r9" / "phaseA.json"), {})
        run_c1.admit("B", raw, str(tmp_path / "r9"), "s", spec)
    run_c1.write_json(str(tmp_path / "r2" / "book_A_1.json"), {})
    with pytest.raises(SystemExit):                                   # complete run: no new run at all
        run_c1.admit("A", raw, str(tmp_path / "r3"), "s", spec, "any reason")


def test_stopped_run_is_final(tmp_path):
    spec = {"inputs": {"prompt_sets": {"A": {1: 0}}}}
    run_c1.admit("A", str(tmp_path), str(tmp_path / "r1"), "s", spec)
    run_c1.write_json(str(tmp_path / "r1" / "phaseA.json"), {"stopped": "no amplitude qualified"})
    with pytest.raises(SystemExit):
        run_c1.admit("A", str(tmp_path), str(tmp_path / "r2"), "s", spec, "try again")


def test_run_lock(tmp_path):
    lock = run_c1.RunLock(str(tmp_path), "seal")
    with pytest.raises(SystemExit):
        run_c1.RunLock(str(tmp_path), "seal")
    lock.release()
    run_c1.RunLock(str(tmp_path), "seal").release()


def toy_session(tmp_path, grid=(4, 16, 64)):
    torch.manual_seed(0)
    model = Toy((1, 1, 7, 63))          # odd windows: no exact zero of the response at N = 2
    s = object.__new__(run_c1.Session)
    s.T, s.t1, s.grid = T, T1, list(grid)
    s.layer_ids, s.global_ids, s.device = [2, 3], [3], "cpu"
    s.layers = model.layers
    s.decoder = lambda input_ids, use_cache: model(input_ids)
    s.spec = {"inputs": {"prompt_sets": {"A": {1342: "x"}, "B": {7: "y"}}, "text_sha256": {1342: "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
                                                                                     7: "a1fce4363854ff888cff4b8e7875d600c2682390412a8cf79b37d0b11148b0fa"}},
              "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}
    s.ledger = run_c1.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    s.seal_hash, s.code, s.platform, s.direction_sha = "test", {"f": "h"}, {"global_layers": [3]}, "d"
    s.tok = None
    s.texts = {1342: "x", 7: "y"}
    d = torch.zeros(D); d[0] = 1.0
    s.direction = d
    s.injector = run_c1.PulseInjector(model.layers[1], d)
    s.norms = run_c1.ResidualNormProbe(model.layers[1])
    s.probe = run_c1.HeadProbe(model.layers, s.layer_ids, HEADS, s.t1)
    data = {1342: torch.randn(1, s.T, D), 7: torch.randn(1, s.T, D)}
    s.ids = lambda gid: (data[int(gid)], "tok")
    return s


def test_phases_end_to_end(tmp_path):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    run_c1.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["chosen_a"] == 0.1 and a["sham_differences"] == 0
    assert set(a["rows"][0]["by_N"]) == {"2", "16", "128", "1024"}
    assert a["stopped"] is None and a["power_gate"]["snr"] > 3
    assert a["record"]["code_sha256"] == {"f": "h"}
    run_c1.phase_b(s, str(out))
    book = json.load(open(out / "book_A_1342.json"))
    assert book["sham_differences_open"] == 0 and book["sham_differences_close"] == 0
    cell = book["per_N"]["16"]
    gain = np.array(cell["target"])[0, 0] / cell["input_mag"]
    assert abs(gain - theory(7, 16)) < 1e-3
    assert book["record"]["offsets"]["16"] == int(metrics.periodic_positions(16, T)[0])
    assert book["record"]["window"] == [T1, T]
    n_lines = len(open(s.ledger.path).read().splitlines())
    run_c1.phase_b(s, str(out))                                # resume writes nothing new
    assert len(open(s.ledger.path).read().splitlines()) == n_lines
    spec = {"pulse": {"interval_grid_N": s.grid}, "inputs": s.spec["inputs"]}
    r = analyze_c1.analyze(str(out), spec, "test", {"f": "h"})        # the analysis reads real runner output
    assert r["complete"] and r["books"] == 2 and r["C1"]["outcome"] in ("R", "HP", "LP", "T", "F", "S", "U")
    s.ledger.cap_s = 0.0
    with pytest.raises(SystemExit):
        s.forward(torch.randn(1, 16, D), "over budget")


def test_phase_b_checks_phase_a(tmp_path):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    run_c1.phase_a(s, str(out))
    s.code = {"f": "changed"}
    with pytest.raises(SystemExit):
        run_c1.phase_b(s, str(out))


def test_failed_sham_writes_record_and_blocks_silent_retry(tmp_path):
    s = toy_session(tmp_path, grid=(16,))
    out = tmp_path / "run"; out.mkdir()
    run_c1.phase_a(s, str(out))
    real = s.sham_differences
    s.sham_differences = lambda ids: 5
    with pytest.raises(RuntimeError):
        run_c1.phase_b(s, str(out))
    failed = [p for p in os.listdir(out) if ".FAILED." in p]
    assert len(failed) == 1 and not (out / "book_A_1342.json").exists()
    s.sham_differences = real
    with pytest.raises(SystemExit):
        run_c1.phase_b(s, str(out))
    run_c1.phase_b(s, str(out), restart_reason="test: forced sham failure")
    assert json.load(open(out / "book_A_1342.json"))["restart_reason"].startswith("test")


def test_budget_preflight_refuses_phase_b(tmp_path):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    run_c1.phase_a(s, str(out))
    s.ledger.cap_s = s.ledger.seconds + 1e-6
    with pytest.raises(SystemExit):
        run_c1.phase_b(s, str(out))
    assert not any(p.startswith("book_") for p in os.listdir(out))


def test_underpowered_phase_a_stops_with_U(tmp_path, monkeypatch):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    monkeypatch.setattr(run_c1.metrics, "aggregate_snr", lambda *a: 1.0)
    with pytest.raises(SystemExit):
        run_c1.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"].startswith("underpowered") and a["outcome_if_stopped"] == "U"
    with pytest.raises(SystemExit):
        run_c1.phase_b(s, str(out))
