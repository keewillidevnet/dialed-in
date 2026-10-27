"""Tests for C2: tilt metric, outcome rules, knob conditions, runner and analysis. CPU only."""
import hashlib
import json

import numpy as np
import pytest
import torch

from dialedin import metrics, metrics_c2, analyze_c2, run_c2
from dialedin.hooks import GlobalKnob, HeadProbe
from dialedin.wave import WaveInjector
from test_c1 import Toy, D, HEADS

T, T1 = metrics.WINDOW_END, metrics.WINDOW_START
P = [2, 4, 16, 64, 256, 1024, 4096]
G = np.array(P, dtype=np.float64)


# ---------- metrics ----------

def test_tilt_recovers_power_law_exponent():
    for beta in (0.14, -0.3, 0.0):
        assert abs(metrics_c2.tilt(P, 3.0 * G ** beta, np.ones(7, bool)) - beta) < 1e-12
    v = np.ones(7, bool); v[:3] = False
    assert metrics_c2.tilt(P, G ** 0.2, v) is None                       # 4 valid points < 5
    assert metrics_c2.tilt(P, np.r_[0.0, G[1:] ** 0.2], np.ones(7, bool)) is None


def tilts(g05, g10, g15, l05, l15):
    return {s: {"G0.5": g05, "G1.0": g10, "G1.5": g15, "L0.5": l05, "L1.5": l15} for s in "ABC"}


def test_classify_outcomes():
    o = metrics_c2.classify_c2(tilts(0.08, 0.14, 0.20, 0.14, 0.145))
    assert o["outcome"] == "K1" and o["direction"] == "+" and o["prediction_held"] is True
    o = metrics_c2.classify_c2(tilts(0.20, 0.14, 0.08, 0.14, 0.14))
    assert o["outcome"] == "K1" and o["direction"] == "-" and o["prediction_held"] is False
    assert metrics_c2.classify_c2(tilts(0.08, 0.14, 0.20, 0.10, 0.18))["outcome"] == "K2"   # local moves as much
    assert metrics_c2.classify_c2(tilts(0.139, 0.14, 0.141, 0.14, 0.14))["outcome"] == "K0"  # too small
    assert metrics_c2.classify_c2(tilts(0.08, 0.20, 0.14, 0.14, 0.14))["outcome"] == "K0"    # not monotone
    assert metrics_c2.classify_c2(tilts(None, 0.14, 0.20, 0.14, 0.14))["outcome"] == "U"


def test_mixed_sets_are_not_tunable():
    t = tilts(0.08, 0.14, 0.20, 0.14, 0.14)
    t["C"] = {"G0.5": 0.20, "G1.0": 0.14, "G1.5": 0.08, "L0.5": 0.14, "L1.5": 0.14}
    assert metrics_c2.classify_c2(t)["outcome"] == "K0"


# ---------- toy model with layers after the global one ----------

def toy_model():
    torch.manual_seed(0)
    return Toy((1, 1, 7, 63, 5, 3))          # layer 3 is global; layers 4 and 5 read its output


def test_knob_at_one_is_bitwise_identity_and_does_not_reach_layer1():
    model = toy_model()
    x = torch.randn(1, 512, D)
    ref = model(x)
    gk, lk = GlobalKnob(model.layers, [3]), GlobalKnob(model.layers, [2])
    assert torch.equal(model(x), ref)
    h1 = {}
    model.layers[1].register_forward_hook(lambda m, a, o: h1.__setitem__("x", o.clone()))
    gk.g = 1.5; model(x); a = h1["x"]
    gk.g, lk.g = 1.0, 0.5; model(x); b = h1["x"]
    assert torch.equal(a, b)


def toy_session(tmp_path, periods=(4, 16, 64, 256, 1024)):
    model = toy_model()
    s = object.__new__(run_c2.Session)
    s.T, s.t1, s.grid = T, T1, list(periods)
    s.layer_ids, s.global_ids, s.device = [2, 3, 4, 5], [3], "cpu"
    s.layers = model.layers
    s.decoder = lambda input_ids, use_cache: model(input_ids)
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    s.spec = {"inputs": {"prompt_sets": {"A": {1342: "x"}, "B": {7: "y"}}, "text_sha256": {1342: h("x"), 7: h("y")}},
              "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}
    s.ledger = run_c2.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    s.seal_hash, s.code, s.platform, s.direction_sha = "test", {"f": "h"}, {"global_layers": [3]}, "d"
    s.tok, s.texts = None, {1342: "x", 7: "y"}
    d = torch.zeros(D); d[0] = 1.0
    s.direction = d
    s.injector = WaveInjector(model.layers[1], d)
    s.norms = run_c2.ResidualNormProbe(model.layers[1])
    s.probe = HeadProbe(model.layers, s.layer_ids, HEADS, s.t1)
    s.knob_global, s.knob_local = GlobalKnob(model.layers, [3]), GlobalKnob(model.layers, [2])
    s.knob_norms = run_c2.KnobNormProbe(model.layers, [3], [2])
    data = {1342: torch.randn(1, s.T, D), 7: torch.randn(1, s.T, D)}
    s.ids = lambda gid: (data[int(gid)], "tok")
    return s


def test_set_condition():
    model = toy_model()
    s = object.__new__(run_c2.Session)
    s.knob_global, s.knob_local = GlobalKnob(model.layers, [3]), GlobalKnob(model.layers, [2])
    s.set_condition(("G", 1.5)); assert (s.knob_global.g, s.knob_local.g) == (1.5, 1.0)
    s.set_condition(("L", 0.5)); assert (s.knob_global.g, s.knob_local.g) == (1.0, 0.5)


def test_phases_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c2, "PHASE_A_N", (4, 1024))
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c2.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and a["chosen_a"] == run_c2.A and len(a["rows"]) == 5
    assert len({r["median_norm"] for r in a["rows"]}) == 1
    with torch.no_grad():
        run_c2.phase_b(s, str(out))
    book = json.load(open(out / "book_A_1342.json"))
    assert sorted(book["per_condition"]) == sorted(analyze_c2.CONDITIONS)
    assert book["knob_layer_output_norms"]["global"] > 0 and book["knob_layer_output_norms"]["local"] > 0
    assert all(v == 0 for v in book["sham_differences"].values()) and book["sham_differences_close"] == 0
    g10 = np.array(book["per_condition"]["G1.0"]["256"]["target"])
    g15 = np.array(book["per_condition"]["G1.5"]["256"]["target"])
    assert np.array_equal(g10[:2], g15[:2])           # layers 2 and 3 read before the global knob acts
    assert not np.array_equal(g10[2:], g15[2:])       # layers 4 and 5 read after it
    spec = {"drive": {"periods_N": s.grid}, "inputs": s.spec["inputs"]}
    r = analyze_c2.analyze(str(out), spec, "test", {"f": "h"})
    assert r["complete"] and r["C2"]["outcome"] in ("K1", "K2", "K0", "U")
    assert r["book_tilts"][0]["gutenberg_id"] == 1342 and "diagnostic_knob_layer_output_norm_median" in r
    assert set(r["curves"]["A"]) == set(analyze_c2.CONDITIONS)
    json.dumps(r)


def test_underpowered_phase_a_stops_with_U(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c2, "PHASE_A_N", (4, 4096))
    monkeypatch.setattr(run_c2.metrics, "aggregate_snr", lambda *a: 1.0)
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    with pytest.raises(SystemExit), torch.no_grad():
        run_c2.phase_a(s, str(out))
    assert json.load(open(out / "phaseA.json"))["stopped"].startswith("not linear or underpowered")
    with pytest.raises(SystemExit):
        run_c2.phase_b(s, str(out))


# ---------- analysis on synthetic books ----------

SPEC = {"drive": {"periods_N": P},
        "inputs": {"prompt_sets": {"A": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, "B": {6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                                   "C": {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}}}}
REC = {"layers_measured": list(range(2, 52)), "seal_sha256": "seal", "code_sha256": {"x": "y"}, "platform": {"gpu": "toy"}}
BETA = {"G1.0": 0.14, "G0.5": 0.09, "G1.5": 0.19, "L0.5": 0.138, "L1.5": 0.142}


def fake_cell(gain_value, rng, snr=50.0):
    inp = 7.0
    target = gain_value * inp * (1 + 0.01 * rng.standard_normal((50, 32)))
    bin_power = np.full(16, float(np.sum(target ** 2)) / snr ** 2)
    return {"input_mag": inp, "input_mag_nominal": inp, "target": target.tolist(),
            "floor": np.full((50, 32), np.sqrt(bin_power[0] / 1600)).tolist(),
            "bin_power": bin_power.tolist(), "n_below": 8, "realized_drive": {"realized_over_nominal": 1.0}}


def write_run(tmp_path):
    for i, s in enumerate("ABC"):
        for j in range(5):
            gid = 5 * i + j + 1
            rng = np.random.default_rng(gid)
            per = {c: {str(N): fake_cell(0.5 * N ** b, rng) for N in P} for c, b in BETA.items()}
            book = {"set": s, "gutenberg_id": gid, "a": 0.1, "sham_differences": {c: 0 for c in BETA},
                    "sham_differences_close": 0, "global_layers": list(range(3, 52, 4)), "per_condition": per,
                    "record": dict(REC)}
            json.dump(book, open(tmp_path / f"book_{s}_{gid}.json", "w"))
    json.dump({"phase": "A", "chosen_a": 0.1, "sham_differences": 0, "record": dict(REC)},
              open(tmp_path / "phaseA.json", "w"))
    json.dump({"utc": "x"}, open(tmp_path / "book_A_1.ATTEMPT.20261027T000000Z.json", "w"))


def test_analysis_end_to_end(tmp_path):
    write_run(tmp_path)
    r = analyze_c2.analyze(str(tmp_path), SPEC, "seal", {"x": "y"})
    c = r["C2"]
    assert c["outcome"] == "K1" and c["direction"] == "+" and c["prediction_held"]
    assert abs(c["per_set"]["A"]["tilts"]["G1.0"] - 0.14) < 0.005
    lo, hi = r["bootstrap"]["A"]["delta_global_95"]
    assert lo <= c["per_set"]["A"]["delta_global"] <= hi
    assert r["book_signs"] == {"books": 15, "global_up_steeper": 15, "local_up_steeper": 15}
    json.dumps(r)


def test_analysis_refuses_dirty_sham_and_partial(tmp_path):
    write_run(tmp_path)
    b = json.load(open(tmp_path / "book_A_1.json")); b["sham_differences"]["L1.5"] = 3
    json.dump(b, open(tmp_path / "book_A_1.json", "w"))
    with pytest.raises(SystemExit):
        analyze_c2.analyze(str(tmp_path), SPEC)
    with pytest.raises(SystemExit):
        analyze_c2.analyze(str(tmp_path), SPEC, partial=True)


def test_g1_reproduction_check(tmp_path):
    write_run(tmp_path)
    c1 = tmp_path / "c1"; c1.mkdir()
    b = json.load(open(tmp_path / "book_A_2.json"))
    json.dump({"per_N": {k: {"target": v["target"]} for k, v in b["per_condition"]["G1.0"].items()}},
              open(c1 / "book_A_2.json", "w"))
    r = analyze_c2.analyze(str(tmp_path), SPEC, c1_run_dir=str(c1))
    assert r["diagnostic_G1_vs_C1v5"]["cells_compared"] == 7 and r["diagnostic_G1_vs_C1v5"]["status"] == "compared"
    assert analyze_c2.analyze(str(tmp_path), SPEC, c1_run_dir=str(tmp_path / "nope"))["diagnostic_G1_vs_C1v5"]["status"] == "not found"
    assert r["diagnostic_G1_vs_C1v5"]["max_relative_difference"] == 0.0


def test_common_validity_across_conditions():
    # One condition losing a period must not change which periods the other conditions are fit over.
    curves = {c: [({"whole": 0.5 * np.array(P, float) ** b}, np.full(7, 50.0)) for _ in range(5)]
              for c, b in BETA.items()}
    curves["L1.5"] = [({"whole": c[0]["whole"]}, np.r_[np.full(6, 50.0), 1.0]) for c in curves["L1.5"]]
    t, v = analyze_c2.set_tilts(curves, P)
    assert not v["L1.5"][-1] and v["G1.0"][-1]
    assert abs(t["G1.0"] - 0.14) < 1e-12                     # fit over 6 common periods, still exact on a power law
