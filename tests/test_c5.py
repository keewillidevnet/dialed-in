"""Tests for C5: dial outcomes, training on a toy model, adapter checks, phases A, B and R, analysis, admission."""
import hashlib
import json

import numpy as np
import pytest
import torch

from dialedin import metrics, metrics_c5, analyze_c5, run_c5
from dialedin import pilot_c5 as P
from test_c4 import ToyLM, ToyFull, D

T = metrics.WINDOW_END
GRID = [3, 12, 48, 192, 768]                       # multiples of 3: the toy's odd windows never null them


def odd_windows(lm):
    for i, layer in enumerate(lm.layers):
        layer.self_attn.sliding_window = 63 if i == 3 else 1499   # odd, so no power-of-2 period is nulled


def toy_spec():
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()
    return {"inputs": {"prompt_sets": {"A": {7: "y"}}, "text_sha256": {1342: h("x"), 7: h("y"), 9: h("z")},
                       "length_tokens": T},
            "drive": {"periods_N": GRID}, "length_tokens": T,
            "training": {"family": "all", "layers": [2], "rank": 2, "alpha": 4.0, "steps": 3, "lr": 3e-3,
                         "clip": 1.0, "beta": 5.0, "kl_budget": 0.02, "beta_excess": 100.0, "kl_rollback": 0.5,
                         "shift": -0.2, "scale_range": [0.25, 2.0], "periods_per_step": 3, "tokens": 6144,
                         "periods_N": list(P.PERIODS), "seed": 0, "direction_seed_offset": 10000,
                         "base_seeds": [20001], "regions": [{"gutenberg_id": 9, "start": 0}]},
            "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01, "training_hours_estimate": 0.01,
                       "recall_seconds_per_trial_estimate": 0.01, "switch_seconds_per_pass_estimate": 0.01}}


def toy_world(tmp_path):
    lm = ToyLM()
    lm.config.hidden_size = D
    odd_windows(lm)
    for p in lm.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)
    toks = {g: rng.integers(1, 900, size=T + 10).tolist() for g in (1342, 7, 9)}
    texts = {1342: "x", 7: "y", 9: "z"}
    tok = lambda text, add_special_tokens=False: {"input_ids": toks[{"x": 1342, "y": 7, "z": 9}[text]]}
    return lm, ToyFull(lm), texts, tok


def trained_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_c5, "CHECK_LAYER", 4)
    spec = toy_spec()
    lm, full, texts, tok = toy_world(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    ledger = run_c5.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    platform = {"global_layers": [3]}
    run_c5.train(spec, "testseal", {"f": "h"}, ledger, None, str(out), loaded=(texts, full, lm, lm.layers, tok, platform))
    P.configure(8192)
    lm, full, _, _ = toy_world(tmp_path)            # a fresh model, as in a new process on the pod
    s = object.__new__(run_c5.Session)
    s.spec, s.texts, s.tok, s.model, s.decoder, s.layers, s.platform = spec, texts, tok, full, lm, lm.layers, platform
    s.ledger = ledger
    s.seal_hash, s.code = "testseal", {"f": "h"}
    run_c5.Session._setup(s, str(out))
    for hd in s.probe.handles:
        hd.remove()
    from dialedin.hooks import HeadProbe
    from dialedin.wave import WaveInjector
    s.probe = HeadProbe(lm.layers, s.layer_ids, 4, s.t1)
    d = torch.zeros(D); d[0] = 1.0
    s.injector.remove(); s.direction = d
    s.injector = WaveInjector(lm.layers[1], d)
    return s, out, spec


def test_dial_outcomes():
    t = {"S0": 0.13, "S0.5": 0.12, "S1": 0.105, "S1.5": 0.097, "S2": 0.087}
    ok = {c: 0.99 for c in metrics_c5.CONDITIONS}
    o = metrics_c5.dial_outcome(t, [-0.04] * 5, ok)
    assert o["outcome"] == "L1" and o["direction"] == "flatter" and o["spearman"] == -1.0
    assert metrics_c5.dial_outcome(t, [-0.04] * 5, dict(ok, S2=1.3))["outcome"] == "L2"
    assert metrics_c5.dial_outcome(t, [-0.04] * 5, dict(ok, **{"S0.5": 1.2}))["outcome"] == "L2"   # every scale gated
    assert metrics_c5.dial_outcome(t, [-0.04, -0.04, -0.04, None, 0.01], ok)["outcome"] == "L0"
    flat = {"S0": 0.13, "S0.5": 0.128, "S1": 0.126, "S1.5": 0.124, "S2": 0.122}
    assert metrics_c5.dial_outcome(flat, [-0.01] * 5, ok)["outcome"] == "L0"        # too small
    zig = {"S0": 0.13, "S0.5": 0.08, "S1": 0.12, "S1.5": 0.07, "S2": 0.09}
    assert metrics_c5.dial_outcome(zig, [-0.04] * 5, ok)["outcome"] == "L0"         # not monotone
    steep = {"S0": 0.10, "S0.5": 0.11, "S1": 0.12, "S1.5": 0.13, "S2": 0.14}
    o = metrics_c5.dial_outcome(steep, [0.04] * 5, ok)
    assert o["outcome"] == "L0" and o["direction"] == "steeper"                   # trained to flatten
    assert metrics_c5.dial_outcome(dict(t, S1=None), [], ok)["outcome"] == "U"


def test_choose_family():
    from dialedin import compare_pilots as C
    row = lambda fam, c1, elig=True: {"family": fam, "change_1": c1, "eligible": elig}
    assert C.choose_family([row("all", -0.20), row("qk", -0.12)]) == "all"
    assert C.choose_family([row("all", -0.20), row("qk", -0.19), row("gate", -0.185)]) == "qk"
    assert C.choose_family([row("all", -0.20), row("gate", -0.23)]) == "gate"
    assert C.choose_family([row("all", -0.2, False), row("qk", -0.1, False)]) == "all"
    assert C.choose_family([row("all", -0.2, False), row("v", -0.1)]) == "v"


def test_train_check_and_phases_end_to_end(tmp_path, monkeypatch):
    s, out, spec = trained_run(tmp_path, monkeypatch)
    tj = json.load(open(out / "training.json"))
    assert len(tj["steps"]) == 3 and tj["adapter_sha256"] == run_c5.sha_file(str(out / "adapter.pt"))
    ad = s.adapter_checks()
    assert ad["as_expected"], ad
    assert s.logits_check()["as_expected"]
    monkeypatch.setattr(run_c5, "PHASE_A_N", (3, 768))
    with torch.no_grad():
        run_c5.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and len(a["rows"]) == 5
    with torch.no_grad():
        run_c5.phase_b(s, str(out))
    book = json.load(open(out / "book_A_7.json"))
    assert sorted(book["per_condition"]) == sorted(metrics_c5.CONDITIONS)
    assert all(v == 0 for v in book["sham_differences"].values()) and book["sham_differences_close"] == 0
    assert P.LoRALinear.scale == 0.0
    r = analyze_c5.analyze(str(out), spec, "testseal", {"f": "h"})
    assert r["complete"] and r["C5"]["outcome"] in ("L1", "L2", "L0") and None not in r["tilts"].values()
    json.dumps(r)


def test_tampered_adapter_is_refused(tmp_path, monkeypatch):
    s, out, spec = trained_run(tmp_path, monkeypatch)
    ad = torch.load(out / "adapter.pt")
    ad["B"][0] = ad["B"][0] + 1.0
    torch.save(ad, out / "adapter.pt")
    lm, full, _, _ = toy_world(tmp_path)
    s2 = object.__new__(run_c5.Session)
    for k in ("spec", "texts", "tok", "platform", "ledger", "seal_hash", "code"):
        setattr(s2, k, getattr(s, k))
    s2.model, s2.decoder, s2.layers = full, lm, lm.layers
    with pytest.raises(SystemExit, match="sha256"):
        run_c5.Session._setup(s2, str(out))


def test_recall_trials_and_behavior_phase(tmp_path, monkeypatch):
    s, out, spec = trained_run(tmp_path, monkeypatch)
    monkeypatch.setitem(run_c5.RECALL, "context_tokens", 2048)
    monkeypatch.setitem(run_c5.RECALL, "filler_start", 100)
    monkeypatch.setattr(run_c5, "SWITCH", {"start": 50, "context": 512, "after": 256, "noise_tokens": 8,
                                            "noise_seed": 7, "bins": ((1, 16), (17, 64), (65, 256))})
    s.tok = lambda t, add_special_tokens=False: {"input_ids": [ord(c) % 900 + 1 for c in t]}
    s.texts = {7: "abcdefghij" * 300, 9: "klmnopqrstuvwxyz" * 200}
    s.spec = dict(s.spec, inputs=dict(s.spec["inputs"], prompt_sets={"A": {7: "y", 9: "z"}}))
    trials = run_c5.recall_trials(s)
    assert len(trials) == 8 and all(len(t["ids"]) == 2048 for t in trials)
    t0 = trials[0]
    ans = s.tok(run_c5.RECALL["answer"].format(key=t0["key"]))["input_ids"]
    assert t0["ids"][t0["answer_from"]:] == ans
    needle = s.tok(run_c5.RECALL["needle"].format(key=t0["key"]))["input_ids"]
    at = int(0.1 * (2048 - len(needle) - len(s.tok(run_c5.RECALL["question"])["input_ids"]) - len(ans)))
    assert t0["ids"][at:at + len(needle)] == needle
    sw = run_c5.switch_inputs(s)
    assert sw["pairs"] == [(7, 9), (9, 7)]
    assert sw["switch"][(7, 9)][:512] == sw["clean"][7][:512] and sw["switch"][(7, 9)][512:] == sw["clean"][9][512:]
    assert sw["noise"][9][:512] == sw["clean"][9][:512] and sw["noise"][9][520:] == sw["clean"][9][520:]
    assert set(sw["noise"][9][512:520]) <= set(sw["clean"][9][:512])          # noise comes from the context only
    with torch.no_grad():
        run_c5.phase_r(s, str(out))
    rec = json.load(open(out / "behavior.json"))
    assert len(rec["recall"]) == 8 * 5 and len(rec["switch"]) == 5 and len(rec["noise"]) == 5
    assert all(len(r["excess_curve"]) == 256 for r in rec["switch"]) and len(rec["noise"][0]["excess_curve"]) == 248
    assert set(rec["switch"][0]["excess_binned"]) == {"1-16", "17-64", "65-256"}


def test_token_losses_match_direct(tmp_path, monkeypatch):
    s, out, spec = trained_run(tmp_path, monkeypatch)
    ids = list(range(1, 900))
    loss = run_c5.token_losses(s, ids, 500, 899)
    with torch.no_grad():
        h = s.decoder(torch.tensor([ids])).last_hidden_state[0]
        lp = s.logits(h[499:898]).log_softmax(-1)
        direct = -lp.gather(1, torch.tensor(ids[500:899])[:, None])[:, 0].double().numpy()
    assert np.allclose(loss, direct, atol=1e-5)


def test_binned():
    c = np.arange(1, 2049, dtype=float)
    b = run_c5.binned(c)
    assert b["1-16"] == np.mean(np.arange(1, 17)) and b["1025-2048"] == np.mean(np.arange(1025, 2049))
    b2 = run_c5.binned(c[32:], offset=32)          # noise: the curve starts 32 tokens after the seam
    assert "1-16" not in b2 and "17-64" not in b2 and b2["33-64"] == np.mean(np.arange(33, 65))


def test_admission_order(tmp_path):
    spec = toy_spec()
    raw = tmp_path / "raw"; raw.mkdir()
    out = raw / "c5v1-x"
    run_c5.admit("T", str(raw), str(out), "seal", spec)
    with pytest.raises(SystemExit, match="training.json"):
        run_c5.admit("A", str(raw), str(out), "seal", spec)
    (out / "training.json").write_text("{}")
    run_c5.admit("A", str(raw), str(out), "seal", spec)
    with pytest.raises(SystemExit, match="phaseA.json"):
        run_c5.admit("B", str(raw), str(out), "seal", spec)
    with pytest.raises(SystemExit, match="after Phase B"):
        run_c5.admit("R", str(raw), str(out), "seal", spec)
    with pytest.raises(SystemExit, match="already trained"):
        run_c5.admit("T", str(raw), str(raw / "c5v1-y"), "seal", spec)
    with pytest.raises(SystemExit, match="already started"):
        run_c5.admit("T", str(raw), str(out), "seal", spec)


def test_rollback_path(tmp_path, monkeypatch):
    spec = toy_spec()
    spec["training"]["kl_rollback"] = -1.0                 # every step is "bad": restore and halve the rate
    lm, full, texts, tok = toy_world(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    ledger = run_c5.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    run_c5.train(spec, "testseal", {"f": "h"}, ledger, None, str(out),
                 loaded=(texts, full, lm, lm.layers, tok, {"global_layers": [3]}))
    P.configure(8192)
    tj = json.load(open(out / "training.json"))
    assert len(tj["rollbacks"]) == 3 and all(r["restored_to_step"] == -1 for r in tj["rollbacks"])
    assert abs(tj["rollbacks"][-1]["new_lr"] - spec["training"]["lr"] / 8) < 1e-12
    ad = torch.load(out / "adapter.pt")
    assert all(float(b.abs().max()) == 0.0 for b in ad["B"])   # restored to the untrained snapshot


def test_one_training_per_seal_and_phase_a_retry(tmp_path):
    spec = toy_spec()
    raw = tmp_path / "raw"; raw.mkdir()
    first = raw / "c5v1-a"
    run_c5.admit("T", str(raw), str(first), "seal", spec)
    (first / "training.json").write_text("{}")
    with pytest.raises(SystemExit, match="already trained"):
        run_c5.admit("T", str(raw), str(raw / "c5v1-b"), "seal", spec, restart_reason="crash")
    (first / "phaseA.FAILED.x.json").write_text("{}")
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c5.admit("A", str(raw), str(first), "seal", spec)
    run_c5.admit("A", str(raw), str(first), "seal", spec, restart_reason="sham failed: driver fault")


def test_negative_base_tilt_is_u():
    t = {"S0": -0.01, "S0.5": -0.02, "S1": -0.03, "S1.5": -0.04, "S2": -0.05}
    assert metrics_c5.dial_outcome(t, [-0.04] * 5, {c: 1.0 for c in metrics_c5.CONDITIONS})["outcome"] == "U"
