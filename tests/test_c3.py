"""Tests for C3: head knob, head selection and control, outcome rules, runner and analysis. CPU only."""
import hashlib
import json

import numpy as np
import pytest
import torch

from dialedin import metrics, metrics_c3, analyze_c3, run_c3
from dialedin.headknob import HeadKnob
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector
from test_c1 import Toy, D, HEADS
from test_c2 import fake_cell, P

T, T1 = metrics.WINDOW_END, metrics.WINDOW_START
HD = 16                                     # toy head dim (HEADS x HD = 64 = D)


def test_head_knob_scales_only_chosen_heads():
    torch.manual_seed(0)
    model = Toy((1, 1, 7, 63))
    x = torch.randn(1, 64, D)
    out_ref = model(x)
    knob = HeadKnob(model.layers, [(2, 1), (2, 3)], head_dim=HD)
    seen = {}
    # registered after the knob, so it sees the knob's output
    model.layers[2].self_attn.o_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].clone()))
    model(x); ref = seen["x"]
    assert torch.equal(model(x), out_ref)                       # g = 1: bitwise unchanged
    for g in (0.0, 2.0):
        knob.g = g
        model(x); y = seen["x"]
        for h in range(HEADS):
            sl = slice(h * HD, (h + 1) * HD)
            want = ref[..., sl] * g if h in (1, 3) else ref[..., sl]
            assert torch.equal(y[..., sl], want), (g, h)


def scan_rows():
    rng = np.random.default_rng(0)
    rows = []
    for l in range(2, 52):
        for h in range(32):
            big = l == 2 and h < 20
            rows.append({"layer": l, "head": h, "low_gain": (1.0 if big else 0.1) * (1 + rng.random()),
                         "size": (1.0 if big else 0.3) * (1 + rng.random())})
    rows.sort(key=lambda r: -r["low_gain"])
    return {"version": 2, "set": "A", "by_low_gain": rows}


def test_selection_is_limited_to_early_layers():
    scan = scan_rows()
    scan["by_low_gain"].insert(0, {"layer": 30, "head": 0, "low_gain": 99.0, "size": 1.0})
    sel = run_c3.selected_heads(scan)
    assert len(sel) == 32 and all(l in run_c3.SELECT_LAYERS for l, _ in sel)
    assert run_c3.measured_layers(sel) == list(range(6, 52))


def test_control_is_size_matched_disjoint_and_deterministic():
    scan = scan_rows()
    sel = run_c3.selected_heads(scan)
    ctrl, ratio = run_c3.control_heads(scan, sel)
    assert (ctrl, ratio) == run_c3.control_heads(scan, sel)
    assert len(ctrl) == 32 and not set(ctrl) & set(sel) and len(set(ctrl)) == 32
    assert all(l in run_c3.SELECT_LAYERS for l, _ in ctrl)
    assert 0 < ratio <= 1.5
    assert all(l != 3 for l, _ in ctrl) or any(l == 3 for l, _ in sel)       # global layer 3 only for layer-3 picks
    cs, cc = run_c3.composition(sel), run_c3.composition(ctrl)
    assert cc["2"] == min(cs["2"], 12)                                       # layer 2 is filled first, up to its 12 spares
    assert run_c3.composition_gap(sel, ctrl) == sum(abs(cs[k] - cc[k]) for k in cs)


def test_layer_preference():
    assert run_c3.layer_preference(2) == [2, 4, 5]
    assert run_c3.layer_preference(5) == [5, 4, 2]
    assert run_c3.layer_preference(3) == [3, 2, 4, 5]


def test_load_scan_checks_hash_version_and_set(tmp_path):
    scan = scan_rows()
    f = tmp_path / "scan.json"; f.write_text(json.dumps(scan))
    spec = {"heads": {"selection": {"file": "scan.json", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}}}
    assert run_c3.load_scan(str(tmp_path), spec)["set"] == "A"
    spec["heads"]["selection"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        run_c3.load_scan(str(tmp_path), spec)
    for bad in (dict(scan, set="B"), dict(scan, version=1)):
        f.write_text(json.dumps(bad))
        spec["heads"]["selection"]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        with pytest.raises(SystemExit):
            run_c3.load_scan(str(tmp_path), spec)


def t3(s0, s05, s1, s15, s2, r0, r2):
    return {s: {"S0.0": s0, "S0.5": s05, "S1.0": s1, "S1.5": s15, "S2.0": s2, "R0.0": r0, "R2.0": r2} for s in "BC"}


def test_classify_c3():
    c = lambda t, r=0.9, gap=0: metrics_c3.classify_c3(t, r, gap)
    o = c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.139, 0.141))
    assert o["outcome"] == "K1" and o["direction"] == "+" and o["prediction_held"]
    assert c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.139, 0.141), 0.3)["outcome"] == "K2"   # control too small to claim specificity
    assert c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.139, 0.141), 0.9, 10)["outcome"] == "K2"  # layer mix too different
    # scale-aware: at ratio 0.6 the control change counts 1/0.6 as much
    assert c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.13, 0.17), 1.0)["outcome"] == "K1"      # 0.12 >= 2 x 0.04
    assert c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.13, 0.17), 0.6)["outcome"] == "K2"      # 0.12 < 2 x 0.04 / 0.6
    o = c(t3(0.20, 0.17, 0.14, 0.11, 0.08, 0.14, 0.14))
    assert o["outcome"] == "K1" and o["direction"] == "-" and o["prediction_held"] is False
    assert c(t3(0.08, 0.11, 0.14, 0.17, 0.20, 0.10, 0.18))["outcome"] == "K2"
    assert c(t3(0.135, 0.137, 0.14, 0.142, 0.145, 0.14, 0.14))["outcome"] == "K0"
    assert c(t3(0.08, 0.15, 0.14, 0.17, 0.20, 0.14, 0.14))["outcome"] == "K0"   # not monotone
    assert c(t3(None, 0.11, 0.14, 0.17, 0.20, 0.14, 0.14))["outcome"] == "U"


# ---------- runner on a toy model ----------

def toy_session(tmp_path, periods=(4, 16, 64, 256, 1024)):
    torch.manual_seed(0)
    model = Toy((1, 1, 7, 63, 5, 3))
    s = object.__new__(run_c3.Session)
    s.T, s.t1, s.grid = T, T1, list(periods)
    s.selected, s.random = [(2, 0), (2, 1)], [(2, 2), (2, 3)]
    s.layer_ids, s.global_ids, s.device = [3, 4, 5], [3], "cpu"
    s.layers = model.layers
    s.decoder = lambda input_ids, use_cache: model(input_ids)
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    s.spec = {"inputs": {"prompt_sets": {"B": {7: "y"}, "C": {9: "z"}}, "text_sha256": {1342: h("x"), 7: h("y"), 9: h("z")}},
              "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}
    s.ledger = run_c3.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    s.seal_hash, s.code, s.platform, s.direction_sha = "test", {"f": "h"}, {"global_layers": [3]}, "d"
    s.tok, s.texts = None, {1342: "x", 7: "y", 9: "z"}
    d = torch.zeros(D); d[0] = 1.0
    s.direction = d
    s.injector = WaveInjector(model.layers[1], d)
    s.norms = run_c3.ResidualNormProbe(model.layers[1])
    s.probe = HeadProbe(model.layers, s.layer_ids, HEADS, s.t1)
    s.knob_selected, s.knob_random = HeadKnob(model.layers, s.selected, HD), HeadKnob(model.layers, s.random, HD)
    s.slice_norms = run_c3.SliceNormProbe(model.layers, s.selected + s.random, HD)
    s.size_ratio, s.layer_gap = 0.8, 0
    data = {1342: torch.randn(1, s.T, D), 7: torch.randn(1, s.T, D), 9: torch.randn(1, s.T, D)}
    s.ids = lambda gid: (data[int(gid)], "tok")
    return s


def test_phases_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c3, "PHASE_A_N", (4, 1024))
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c3.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and len(a["rows"]) == 5
    with torch.no_grad():
        run_c3.phase_b(s, str(out))
    book = json.load(open(out / "book_B_7.json"))
    assert sorted(book["per_condition"]) == sorted(analyze_c3.CONDITIONS)
    assert all(v == 0 for v in book["sham_differences"].values()) and book["sham_differences_close"] == 0
    s1 = np.array(book["per_condition"]["S1.0"]["256"]["target"])
    s0 = np.array(book["per_condition"]["S0.0"]["256"]["target"])
    assert not np.array_equal(s1, s0)                         # the knob reaches the downstream layers
    assert book["record"]["knob"]["selected_heads"] == [[2, 0], [2, 1]]
    assert book["slice_norms"]["selected_sum"] > 0 and book["slice_norms"]["control_sum"] > 0
    spec = {"drive": {"periods_N": s.grid}, "inputs": s.spec["inputs"]}
    r = analyze_c3.analyze(str(out), spec, "test", {"f": "h"})
    assert r["complete"] and r["C3"]["outcome"] in ("K1", "K2", "K0", "U")
    json.dumps(r)


# ---------- analysis on synthetic books ----------

SPEC = {"drive": {"periods_N": P},
        "inputs": {"prompt_sets": {"B": {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, "C": {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}}}}
REC = {"layers_measured": list(range(6, 52)), "seal_sha256": "seal", "code_sha256": {"x": "y"}, "platform": {"gpu": "toy"},
       "knob": {"control_size_ratio": 0.8, "layer_gap": 0}}
BETA = {"S1.0": 0.14, "S0.0": 0.08, "S0.5": 0.11, "S1.5": 0.17, "S2.0": 0.20, "R0.0": 0.139, "R2.0": 0.141}


def write_run(tmp_path):
    for i, s in enumerate("BC"):
        for j in range(5):
            gid = 6 + 5 * i + j
            rng = np.random.default_rng(gid)
            per = {c: {str(N): fake_cell(0.5 * N ** b, rng) for N in P} for c, b in BETA.items()}
            for c in per.values():
                for cell in c.values():
                    cell["target"] = cell["target"][:46]
                    cell["floor"] = cell["floor"][:46]
            book = {"set": s, "gutenberg_id": gid, "a": 0.1, "sham_differences": {c: 0 for c in BETA},
                    "sham_differences_close": 0, "global_layers": list(range(3, 52, 4)), "per_condition": per,
                    "record": dict(REC)}
            json.dump(book, open(tmp_path / f"book_{s}_{gid}.json", "w"))
    json.dump({"phase": "A", "chosen_a": 0.1, "sham_differences": 0, "record": dict(REC)},
              open(tmp_path / "phaseA.json", "w"))


def test_analysis_end_to_end(tmp_path):
    write_run(tmp_path)
    r = analyze_c3.analyze(str(tmp_path), SPEC, "seal", {"x": "y"})
    c = r["C3"]
    assert c["outcome"] == "K1" and c["direction"] == "+" and c["prediction_held"]
    lo, hi = r["bootstrap"]["B"]["delta_selected_95"]
    assert lo <= c["per_set"]["B"]["delta_selected"] <= hi
    assert r["book_signs"] == {"books": 10, "selected_up_steeper": 10, "random_up_steeper": 10}


def test_s1_reproduction_aligns_layers(tmp_path):
    write_run(tmp_path)
    c1 = tmp_path / "c1"; c1.mkdir()
    b = json.load(open(tmp_path / "book_B_6.json"))
    full = {k: {"target": [[0.0] * 32] * 4 + v["target"]} for k, v in b["per_condition"]["S1.0"].items()}
    json.dump({"per_N": full, "record": {"layers_measured": list(range(2, 52))}}, open(c1 / "book_B_6.json", "w"))
    r = analyze_c3.analyze(str(tmp_path), SPEC, c1_run_dir=str(c1))
    assert r["diagnostic_S1_vs_C1v5"]["cells_compared"] == 7
    assert r["diagnostic_S1_vs_C1v5"]["max_relative_difference"] == 0.0
