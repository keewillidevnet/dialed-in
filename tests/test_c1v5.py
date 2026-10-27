"""Tests for the C1 v5 sine-drive harness and analysis on toy models with known responses. CPU only."""
import json

import numpy as np
import pytest
import torch

from dialedin import metrics, analyze_c1v5, run_c1v5
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector, drive
from test_c1 import Toy, theory, D, HEADS, GRID, fake_book as v4_fake_book

T, T1 = metrics.WINDOW_END, metrics.WINDOW_START


def run_toy(N, windows=(1, 1, 8, 64), amplitude=0.05, phase=0.0, dtype=torch.float32):
    torch.manual_seed(0)
    model = Toy(windows).to(dtype)
    base = torch.randn(1, T, D).to(dtype)
    d = torch.zeros(D); d[0] = 1.0
    inj = WaveInjector(model.layers[1], d)
    probe = HeadProbe(model.layers, [2, 3], HEADS, T1)
    with torch.no_grad():
        probe.mode = "record"; model(base)
        cos_b, sin_b = probe.set_frequencies(N, T, "cpu")
        w = drive(N, T, amplitude, phase)
        inj.arm(w); probe.mode = "measure"; model(base); inj.disarm()
    out = probe.collect()
    inp = float(metrics.component(inj.realized[T1:], cos_b[:, :1], sin_b[:, :1])[0])
    nominal = float(metrics.component(w[T1:], cos_b[:, :1], sin_b[:, :1])[0])
    gain, _ = metrics.gains(out["target"], out["floor"], inp, head_dim=16)
    return gain, out, inp / nominal


def test_drive_shape():
    w = drive(4, 8, 2.0)
    assert np.allclose(w, [2, 0, -2, 0, 2, 0, -2, 0], atol=1e-12)
    assert np.allclose(drive(2, 4, 1.0, run_c1v5.PHASE_CONTROL), [0.70710678, -0.70710678] * 2)


def test_zero_wave_is_bitwise_sham_bf16():
    torch.manual_seed(0)
    model = Toy().to(torch.bfloat16)
    base = torch.randn(1, 2048, D).to(torch.bfloat16)
    base[0, ::16, :] = -0.0
    inj = WaveInjector(model.layers[1], torch.ones(D) / D ** 0.5)
    probe = HeadProbe(model.layers, [2, 3], HEADS, 1024)
    with torch.no_grad():
        probe.mode = "record"; model(base)
        inj.arm(np.zeros(2048)); probe.mode = "zero_check"; model(base)
    assert sum(probe.result.values()) == 0


def test_identity_gain_is_one():
    for N in (2, 3, 16, 4096):
        gain, _, _ = run_toy(N, windows=(1, 1, 1, 64))
        assert abs(gain[0, 0] - 1.0) < 1e-5, (N, gain[0, 0])


def test_moving_average_matches_theory():
    for N in (2, 3, 16, 48, 2048, 4096):
        gain, _, _ = run_toy(N)
        assert abs(gain[0, 0] - theory(8, N)) < 1e-3 * theory(8, N) + 1e-6, (N, gain[0, 0], theory(8, N))


def test_control_phase_gives_same_gain():
    for N in (3, 16, 1024):                       # N = 2 is a structural zero of the 8-token average
        g0, _, _ = run_toy(N)
        g1, _, _ = run_toy(N, phase=run_c1v5.PHASE_CONTROL)
        assert abs(g1[0, 0] / g0[0, 0] - 1) < 1e-3, N


def test_realized_drive_in_bf16():
    gain, out, ratio = run_toy(64, dtype=torch.bfloat16, amplitude=0.5)
    assert abs(ratio - 1) < 0.01
    assert abs(gain[0, 0] - theory(8, 64)) < 0.02


def test_sine_drive_snr_beats_pulses_at_long_N():
    # The reason for v5: a sine puts all of its energy at 1/N, a pulse train only 1/N of it.
    _, out, _ = run_toy(4096, dtype=torch.bfloat16, amplitude=0.05)
    snr = metrics.aggregate_snr(out["target"][:1, :1], out["bin_power"], out["n_below"])
    assert snr > 10


# ---------- analysis ----------

SPEC = {"drive": {"interval_grid_N": GRID},
        "inputs": {"prompt_sets": {"A": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, "B": {6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                                   "C": {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}}}}


def fake_book(set_name, gid, curve_fn, seed, **kw):
    b = v4_fake_book(set_name, gid, curve_fn, seed, **kw)
    for N in GRID:
        cell = b["per_N"][str(N)]
        ctrl = cell.pop("aperiodic")
        cell.pop("realized_pulse")
        cell["realized_drive"] = {"realized_over_nominal": 1.0}
        cell["input_mag_nominal"] = cell["input_mag"]
        cell["control"] = dict(ctrl, realized_drive={"realized_over_nominal": 1.0})
    return b


def write_run(tmp_path, curve_fn, **kw):
    for i, s in enumerate("ABC"):
        for j in range(5):
            gid = 5 * i + j + 1
            json.dump(fake_book(s, gid, curve_fn, gid, **kw), open(tmp_path / f"book_{s}_{gid}.json", "w"))
    rec = fake_book("A", 1, curve_fn, 1)["record"]
    json.dump({"phase": "A", "rows": [], "chosen_a": 0.1, "sham_differences": 0, "record": rec},
              open(tmp_path / "phaseA.json", "w"))
    json.dump({"utc": "x"}, open(tmp_path / "book_A_1.ATTEMPT.20261027T000000Z.json", "w"))


def test_analysis_end_to_end(tmp_path):
    write_run(tmp_path, lambda N: 1 / (1 + (N / 300) ** 2))
    r = analyze_c1v5.analyze(str(tmp_path), SPEC, "seal", {"x": "y"})
    assert r["C1"]["outcome"] == "HP" and r["complete"]
    assert r["diagnostic_control_phase_outcome"] == "HP"
    assert abs(r["diagnostic_control_over_main_gain_median"][16] - 1) < 1e-9
    assert r["realized_drive_over_nominal_median"] == 1.0
    json.dumps(r)


# ---------- runner ----------

def toy_session(tmp_path, grid=(4, 16, 64)):
    torch.manual_seed(0)
    model = Toy((1, 1, 7, 63))
    s = object.__new__(run_c1v5.Session)
    s.T, s.t1, s.grid = T, T1, list(grid)
    s.layer_ids, s.global_ids, s.device = [2, 3], [3], "cpu"
    s.layers = model.layers
    s.decoder = lambda input_ids, use_cache: model(input_ids)
    import hashlib
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    s.spec = {"inputs": {"prompt_sets": {"A": {1342: "x"}, "B": {7: "y"}}, "text_sha256": {1342: h("x"), 7: h("y")}},
              "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}
    s.ledger = run_c1v5.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    s.seal_hash, s.code, s.platform, s.direction_sha = "test", {"f": "h"}, {"global_layers": [3]}, "d"
    s.tok, s.texts = None, {1342: "x", 7: "y"}
    d = torch.zeros(D); d[0] = 1.0
    s.direction = d
    s.injector = WaveInjector(model.layers[1], d)
    s.norms = run_c1v5.ResidualNormProbe(model.layers[1])
    s.probe = HeadProbe(model.layers, s.layer_ids, HEADS, s.t1)
    data = {1342: torch.randn(1, s.T, D), 7: torch.randn(1, s.T, D)}
    s.ids = lambda gid: (data[int(gid)], "tok")
    return s


def test_phases_end_to_end(tmp_path):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c1v5.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["chosen_a"] == 0.1 and a["sham_differences"] == 0 and a["stopped"] is None
    assert set(a["rows"][0]["by_N"]) == {"2", "16", "128", "1024", "4096"}
    assert a["power_gate"]["N"] == 4096 and a["power_gate"]["snr"] > 3
    with torch.no_grad():
        run_c1v5.phase_b(s, str(out))
    book = json.load(open(out / "book_A_1342.json"))
    assert book["sham_differences_open"] == 0 and book["sham_differences_close"] == 0
    cell = book["per_N"]["16"]
    assert abs(np.array(cell["target"])[0, 0] / cell["input_mag"] - theory(7, 16)) < 1e-3
    assert abs(np.array(cell["control"]["target"])[0, 0] / cell["control"]["input_mag"] - theory(7, 16)) < 1e-3
    assert book["record"]["drive"]["phase_control"] == run_c1v5.PHASE_CONTROL
    spec = {"drive": {"interval_grid_N": s.grid}, "inputs": s.spec["inputs"]}
    r = analyze_c1v5.analyze(str(out), spec, "test", {"f": "h"})
    assert r["complete"] and r["books"] == 2
    assert s.ledger.path.endswith(".jsonl") and "c1v5_" in s.ledger.path


def test_underpowered_phase_a_stops_with_U(tmp_path, monkeypatch):
    s = toy_session(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    monkeypatch.setattr(run_c1v5.metrics, "aggregate_snr", lambda *a: 1.0)
    with pytest.raises(SystemExit), torch.no_grad():
        run_c1v5.phase_a(s, str(out))
    assert json.load(open(out / "phaseA.json"))["stopped"].startswith("underpowered")


def test_admission_and_lock(tmp_path):
    spec = {"inputs": {"prompt_sets": {"A": {1: 0}}}}
    run_c1v5.admit("A", str(tmp_path), str(tmp_path / "r1"), "s", spec)
    with pytest.raises(SystemExit):
        run_c1v5.admit("A", str(tmp_path), str(tmp_path / "r2"), "s", spec)
    lock = run_c1v5.RunLock(str(tmp_path), "seal")
    with pytest.raises(SystemExit):
        run_c1v5.RunLock(str(tmp_path), "seal")
    lock.release()
    assert "c1v5_" in lock.path
