"""Tests for C6: outcome rule, the frozen adapter loaded with its hash, switch inputs, measurement and analysis."""
import json

import numpy as np
import pytest
import torch

from dialedin import analyze_c6, metrics_c6, run_c5, run_c6
from dialedin import pilot_c5 as P
from test_c5 import toy_spec, toy_world


def make_adapter(tmp_path):
    """Train a toy C5 adapter (3 steps) and return its path and sha256."""
    spec = toy_spec()
    lm, full, texts, tok = toy_world(tmp_path)
    out = tmp_path / "c5run"; out.mkdir()
    ledger = run_c5.Ledger(str(tmp_path / "ledger"), "testseal0000", 3600)
    run_c5.train(spec, "testseal", {"f": "h"}, ledger, None, str(out), loaded=(texts, full, lm, lm.layers, tok, {"global_layers": [3]}))
    P.configure(8192)
    path = out / "adapter.pt"
    return str(path), run_c6.sha_file(str(path)), spec["training"]


def c6_world(tmp_path, monkeypatch, sha=None):
    monkeypatch.setattr(run_c6, "CHECK_BOOK", 7)
    monkeypatch.setattr(run_c6, "CHECK_LAYER", 4)
    path, real_sha, tr = make_adapter(tmp_path)
    lm, full, _, _ = toy_world(tmp_path)                       # a fresh model, as in a new process
    rng = np.random.default_rng(1)
    toks = {g: rng.integers(1, 900, size=2000).tolist() for g in (7, 9, 11)}
    names = {"p": 7, "q": 9, "r": 11}
    tok = lambda text, add_special_tokens=False: {"input_ids": toks[names[text]]}
    texts = {7: "p", 9: "q", 11: "r"}
    spec = {"inputs": {"prompt_sets": {"B": {9: "q"}, "C": {11: "r", 7: "p"}}}, "control": {"seed": 12345},
            "switch": {"start": 100, "context": 512, "after": 256, "noise_tokens": 8, "noise_seed": 7},
            "adapter": {"file": path, "sha256": sha or real_sha, "family": tr["family"], "layers": tr["layers"],
                        "rank": tr["rank"], "alpha": tr["alpha"], "trained_on_regions": [{"gutenberg_id": 9}]},
            "budget": {"max_gpu_hours": 1, "seconds_per_forward_estimate": 0.01}}
    return spec, texts, full, lm, tok


def test_switch_outcome():
    up = {"S0": 500.0, "S0.5": 510.0, "S1": 525.0, "S1.5": 540.0, "S2": 560.0}
    o = metrics_c6.switch_outcome(up, [0.1] * 9 + [-0.01])
    assert o["outcome"] == "B1" and o["direction"] == "rises" and o["spearman"] == 1.0
    assert metrics_c6.switch_outcome(up, [0.1] * 7 + [-0.01] * 3)["outcome"] == "B0"            # 7 of 10 books
    small = {"S0": 500.0, "S0.5": 505.0, "S1": 510.0, "S1.5": 515.0, "S2": 520.0}
    assert metrics_c6.switch_outcome(small, [0.04] * 10)["outcome"] == "B0"                       # +4 percent
    down = {k: 1000 - v for k, v in up.items()}
    assert metrics_c6.switch_outcome(down, [-0.1] * 10)["outcome"] == "B0"                         # falls
    assert metrics_c6.switch_outcome(dict(up, S1=float("nan")), [0.1] * 10)["outcome"] == "U"
    assert metrics_c6.switch_outcome(dict(up, S0=0.0), [0.1] * 10)["outcome"] == "U"


def test_session_measure_and_analyze(tmp_path, monkeypatch):
    spec, texts, full, lm, tok = c6_world(tmp_path, monkeypatch)
    s = run_c6.Session(spec, texts, full, lm, lm.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger())
    assert s.adapter_checks()["as_expected"] and s.logits_check()["as_expected"]
    inp = run_c6.switch_inputs(spec, tok, texts)
    assert len(inp["pairs"]) == 6 and inp["gids"] == [9, 11, 7]
    x, y = inp["pairs"][0]
    assert inp["switch"][(x, y)][:512] == inp["clean"][x][:512] and inp["switch"][(x, y)][512:] == inp["clean"][y][512:]
    assert set(inp["noise"][y][512:520]) <= set(inp["clean"][y][:512])
    with torch.no_grad():
        m = run_c6.measure(s, inp)
    assert sorted(m) == sorted(run_c6.ORDER) and P.LoRALinear.scale == 0.0
    assert len(m["S0"]["switch_curve_mean"]) == 256 and len(m["S0"]["noise_curve_mean"]) == 248
    d = {"books": inp["gids"], "conditions": m}
    r = analyze_c6.analyze(d, spec)
    assert r["C6"]["outcome"] in ("B1", "B0", "U") and len(r["bootstrap_95_relative_rise"]) == 2
    assert r["secondary"]["rise_targets_trained_books"] is not None and r["secondary"]["clean_perplexity_ratio"]["S0"] == 1.0
    json.dumps(r)


def test_token_losses_consistent(tmp_path, monkeypatch):
    spec, texts, full, lm, tok = c6_world(tmp_path, monkeypatch)
    s = run_c6.Session(spec, texts, full, lm, lm.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger())
    ids = list(range(1, 700))
    loss = s.token_losses(ids, 300, 699)
    with torch.no_grad():
        h = lm(torch.tensor([ids])).last_hidden_state[0]
        lp = s.logits(h[299:698]).log_softmax(-1)
        direct = -lp.gather(1, torch.tensor(ids[300:699])[:, None])[:, 0].double().numpy()
    assert np.allclose(loss, direct, atol=1e-5)


def test_wrong_adapter_hash_refused(tmp_path, monkeypatch):
    spec, texts, full, lm, tok = c6_world(tmp_path, monkeypatch, sha="0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        run_c6.Session(spec, texts, full, lm, lm.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger())


def test_one_run_per_seal(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    run_c6.admit(str(raw), str(raw / "c6v1-a"), "seal")
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c6.admit(str(raw), str(raw / "c6v1-b"), "seal")
    run_c6.admit(str(raw), str(raw / "c6v1-b"), "seal", restart_reason="pod crashed")
    (raw / "c6v1-b" / "switch.json").write_text("{}")
    with pytest.raises(SystemExit, match="complete"):
        run_c6.admit(str(raw), str(raw / "c6v1-c"), "seal", restart_reason="again")


def test_random_control_matches_norms_and_analysis_reads_it(tmp_path, monkeypatch):
    spec, texts, full, lm, tok = c6_world(tmp_path, monkeypatch)
    s = run_c6.Session(spec, texts, full, lm, lm.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger())
    inp = run_c6.switch_inputs(spec, tok, texts)
    with torch.no_grad():
        m = run_c6.measure(s, inp)
        rep = run_c6.measure(s, run_c6.switch_inputs(spec, tok, texts, [9, 11], start=50), ("S0", "S2"))
        trained = [p.detach().clone() for p in s.params]
        rows = run_c6.random_control(s, 12345)
        ctl = run_c6.measure(s, inp, ("S2",))
    assert all(abs(r["control_norm"] - r["trained_norm"]) <= 1e-4 * max(r["trained_norm"], 1e-12) for r in rows)
    assert any(not torch.equal(a, b) for a, b in zip(trained, s.params))
    fake_c5 = {"switch": [{"condition": c, "per_pair_total": dict(rep[c]["switch_pair_total"])} for c in ("S0", "S2")]}
    d = {"books": inp["gids"], "conditions": m, "reproduction_of_c5": rep,
         "control_random_adapter": {"S2": ctl["S2"], "norms": rows, "seed": 12345}}
    r = analyze_c6.analyze(d, spec, fake_c5)
    assert r["reproduction_of_c5"]["max_relative_difference"] == 0.0
    sp = r["secondary"]["switch_and_clean_loss_sums"]["S0"]
    assert abs((sp["switch_loss_sum"] - sp["clean_loss_sum"]) - r["excess"]["S0"]) < 1e-6
    assert r["control_random_adapter"]["relative_rise"] is not None and r["secondary"]["source_books_rising"] in range(4)
    json.dumps(r)


def test_unfilled_hash_refused_outside_smoke(tmp_path, monkeypatch):
    spec, texts, full, lm, tok = c6_world(tmp_path, monkeypatch, sha="FILL_AT_SEAL")
    with pytest.raises(SystemExit, match="sha256"):
        run_c6.Session(spec, texts, full, lm, lm.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger())
    lm2, full2, _, _ = __import__("test_c5").toy_world(tmp_path)
    s = run_c6.Session(spec, texts, full2, lm2, lm2.layers, tok, {"global_layers": [3]}, str(tmp_path), run_c6.NullLedger(),
                       allow_unfilled=True)
    assert len(s.adapter_sha) == 64
