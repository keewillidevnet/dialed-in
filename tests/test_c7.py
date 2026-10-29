"""Tests for C7: outcome rule, condition masks, Phase A and B on the toy model with a two-layer adapter, analysis,
the reproduction check, admission and refusals."""
import copy
import json

import numpy as np
import pytest
import torch

from dialedin import analyze_c7, metrics_c7, run_c5, run_c7
from dialedin import pilot_c5 as P
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector
from test_c4 import D
from test_c5 import GRID, toy_spec, toy_world

QKV = ["layers.2.self_attn.o_proj"]                 # the toy has one linear module per layer; it stands in for q, k, v


def make_adapter(tmp_path):
    spec = toy_spec()
    spec["training"]["layers"] = [2, 3]
    lm, full, texts, tok = toy_world(tmp_path)
    out = tmp_path / "c5run"; out.mkdir()
    ledger = run_c5.Ledger(str(tmp_path / "ledger5"), "testseal0000", 3600)
    run_c5.train(spec, "testseal", {"f": "h"}, ledger, None, str(out), loaded=(texts, full, lm, lm.layers, tok, {"global_layers": [3]}))
    return str(out / "adapter.pt"), run_c7.sha_file(str(out / "adapter.pt")), spec


def c7_spec(path, sha, c5spec):
    tr = c5spec["training"]
    return {"inputs": {"prompt_sets": {"B": {7: "y"}}, "text_sha256": c5spec["inputs"]["text_sha256"],
                       "length_tokens": c5spec["inputs"]["length_tokens"]},
            "drive": {"periods_N": GRID},
            "adapter": {"file": path, "sha256": sha, "family": tr["family"], "layers": tr["layers"],
                        "rank": tr["rank"], "alpha": tr["alpha"]},
            "conditions": {"BASE": {"scale": 0.0}, "FULL": {"scale": 2.0}, "QKV2": {"scale": 2.0, "modules": list(QKV)},
                           "REST": {"scale": 2.0}},
            "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}


def session(tmp_path, monkeypatch, spec=None, sha=None, allow_unfilled=False):
    monkeypatch.setattr(run_c7, "CHECK_LAYER", 4)
    monkeypatch.setattr(run_c7, "PHASE_A_BOOK", 7)
    path, real_sha, c5spec = make_adapter(tmp_path)
    P.configure(8192)
    spec = spec or c7_spec(path, sha or real_sha, c5spec)
    lm, full, texts, tok = toy_world(tmp_path)
    s = object.__new__(run_c7.Session)
    s.repo, s.spec, s.seal_hash, s.code = "", spec, "testseal", {"f": "h"}
    s.ledger = run_c7.Ledger(str(tmp_path / "ledger7"), "testseal0000", 3600)
    s.texts, s.tok, s.model, s.decoder, s.layers, s.platform = texts, tok, full, lm, lm.layers, {"global_layers": [3]}
    run_c7.Session._setup(s, allow_unfilled)
    for hd in s.probe.handles:
        hd.remove()
    s.probe = HeadProbe(lm.layers, s.layer_ids, 4, s.t1)
    d = torch.zeros(D); d[0] = 1.0
    s.injector.remove(); s.direction = d
    s.injector = WaveInjector(lm.layers[1], d)
    return s, spec


def test_localization_outcome():
    t = {"BASE": 0.14, "FULL": 0.105, "QKV2": 0.112, "REST": 0.138}
    ok = [dict(t)] * 5                                                       # every book like the set: agrees
    o = metrics_c7.localization_outcome(t, ok)
    assert o["outcome"] == "M1" and abs(o["share_QKV2"] - 0.8) < 1e-9 and abs(o["share_REST"] - 2 / 35) < 1e-9
    assert o["books_agree_QKV2"] == 5
    assert metrics_c7.localization_outcome(dict(t, REST=0.12), ok)["outcome"] == "M2"     # REST share 0.57
    assert metrics_c7.localization_outcome(dict(t, QKV2=0.125), ok)["outcome"] == "M0"    # share 0.43
    neg = metrics_c7.localization_outcome(dict(t, QKV2=0.088, REST=0.165), ok)            # REST steepens: share -0.71
    assert neg["outcome"] == "M0" and not neg["share_QKV2_in_range"] and not neg["exclusive"]
    assert metrics_c7.localization_outcome(dict(t, REST=0.152), ok)["outcome"] == "M2"    # REST share -0.34: opposes
    over = metrics_c7.localization_outcome(dict(t, QKV2=0.09), ok)                        # QKV2 share 1.43: overshoot
    assert over["outcome"] == "M0" and not over["share_QKV2_in_range"]
    weak = dict(t, QKV2=0.1385)                                                           # a book barely flattened
    assert not metrics_c7.book_agrees(weak) and metrics_c7.book_agrees(t)
    assert not metrics_c7.book_agrees(dict(t, FULL=0.15)) and not metrics_c7.book_agrees(dict(t, QKV2=None))
    assert metrics_c7.localization_outcome(t, ok[:3] + [weak, dict(t, QKV2=None)])["outcome"] == "M0"   # 3 of 5
    assert metrics_c7.localization_outcome(t, ok[:4] + [weak])["outcome"] == "M1"                       # 4 of 5
    assert metrics_c7.localization_outcome(dict(t, FULL=0.13), ok)["outcome"] == "U"      # FULL only -7%
    assert metrics_c7.localization_outcome(dict(t, BASE=-0.01), ok)["outcome"] == "U"
    assert metrics_c7.localization_outcome(dict(t, QKV2=None), ok)["outcome"] == "U"
    edge = metrics_c7.localization_outcome({"BASE": 0.1, "FULL": 0.08, "QKV2": 0.0878, "REST": 0.0942}, ok)
    assert edge["outcome"] == "M1"                                                        # shares 0.61 and 0.29
    out = metrics_c7.localization_outcome({"BASE": 0.1, "FULL": 0.08, "QKV2": 0.0882, "REST": 0.0938}, ok)
    assert out["outcome"] == "M0" and not out["sufficient"] and not out["exclusive"]      # shares 0.59 and 0.31


def test_placeholders_refused():
    run_c7.check_no_placeholders({"a": {"b": ["x", "fine"]}, "c": 1.0})
    with pytest.raises(SystemExit, match="placeholder"):
        run_c7.check_no_placeholders({"budget": {"pre_seal_smoke": "SMOKE_RESULT"}})
    with pytest.raises(SystemExit, match="adapter.sha256"):
        run_c7.check_no_placeholders({"adapter": {"sha256": "FILL_AT_SEAL"}})


def test_adapter_matches_c5(tmp_path):
    d = tmp_path / "results" / "raw" / "c5"; d.mkdir(parents=True)
    (d / "adapter.pt").write_bytes(b"weights")
    sha = run_c7.sha_file(str(d / "adapter.pt"))
    (d / "training.json").write_text(json.dumps({"adapter_sha256": sha}))
    spec = {"adapter": {"file": "results/raw/c5/adapter.pt", "sha256": sha}}
    run_c7.adapter_matches_c5(str(tmp_path), spec)
    with pytest.raises(SystemExit, match="agree"):
        run_c7.adapter_matches_c5(str(tmp_path), {"adapter": {"file": "results/raw/c5/adapter.pt", "sha256": "0" * 64}})
    (d / "training.json").write_text(json.dumps({"adapter_sha256": "1" * 64}))
    with pytest.raises(SystemExit, match="agree"):
        run_c7.adapter_matches_c5(str(tmp_path), spec)


def test_condition_modules_and_spec_checks():
    names = ["layers.2.self_attn.q_proj", "layers.2.self_attn.k_proj", "layers.2.self_attn.v_proj",
             "layers.2.self_attn.o_proj", "layers.3.mlp.up_proj"]
    spec = {"conditions": {"QKV2": {"modules": names[:3]}}}
    on = run_c7.condition_modules(spec, names)
    assert on["QKV2"] == names[:3] and on["REST"] == names[3:] and on["FULL"] == names
    with pytest.raises(SystemExit, match="QKV2"):
        run_c7.condition_modules({"conditions": {"QKV2": {"modules": ["layers.9.self_attn.q_proj"]}}}, names)
    with pytest.raises(SystemExit, match="QKV2"):
        run_c7.condition_modules({"conditions": {"QKV2": {"modules": [names[0], names[0]]}}}, names)
    good = {"conditions": {"BASE": {"scale": 0}, "FULL": {"scale": 2}, "QKV2": {"scale": 2}, "REST": {"scale": 2}}}
    run_c7.check_condition_spec(good)
    bad = copy.deepcopy(good); bad["conditions"]["QKV2"]["scale"] = 1.5
    with pytest.raises(SystemExit, match="scales"):
        run_c7.check_condition_spec(bad)


def test_masks_restore_exactly_and_checks_pass(tmp_path, monkeypatch):
    s, _ = session(tmp_path, monkeypatch)
    assert s.adapter_modules == ["layers.2.self_attn.o_proj", "layers.3.self_attn.o_proj"]
    saved = [b.clone() for b in s.params[1::2]]
    s.set_condition("QKV2")
    assert torch.equal(s.params[1], saved[0]) and not s.params[3].any() and P.LoRALinear.scale == 2.0
    s.set_condition("REST")
    assert not s.params[1].any() and torch.equal(s.params[3], saved[1])
    s.set_condition("BASE")
    assert all(torch.equal(b, x) for b, x in zip(s.params[1::2], saved)) and P.LoRALinear.scale == 0.0
    ad = s.adapter_checks()
    assert ad["as_expected"], ad
    assert ad["modules_on"] == {"BASE": 2, "FULL": 2, "QKV2": 1, "REST": 1}
    assert s.logits_check()["as_expected"]


def test_phases_and_analysis_end_to_end(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    monkeypatch.setattr(run_c7, "PHASE_A_N", (3, 768))
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c7.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and [r["condition"] for r in a["rows"]] == list(run_c7.ORDER)
    (out / "book_B_7.ATTEMPT.20260101T000000000000Z.json").write_text("{}")      # an earlier attempt
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c7.phase_b(s, str(out))
    with torch.no_grad():
        run_c7.phase_b(s, str(out), restart_reason="test retry")
    book = json.load(open(out / "book_B_7.json"))
    assert book["restart_reason"] == "test retry"
    assert sorted(book["per_condition"]) == sorted(metrics_c7.CONDITIONS) and book["sham_differences_close"] == 0
    assert book["record"]["conditions"]["QKV2"]["modules_on"] == QKV and P.LoRALinear.scale == 0.0
    # a fake C1 v5 record from BASE's own cells: the reproduction check must read 0 and be complete
    c1 = tmp_path / "c1"; c1.mkdir()
    rec = {"median_norm": book["amplitude_reference"]["BASE"],
           "per_N": {str(N): {"target": book["per_condition"]["BASE"][str(N)]["target"],
                              "input_mag": book["per_condition"]["BASE"][str(N)]["input_mag"]} for N in GRID}}
    (c1 / "book_B_7.json").write_text(json.dumps(rec))
    r = analyze_c7.analyze(str(out), spec, "testseal", {"f": "h"}, c1_run_dir=str(c1))
    assert r["complete"] and r["C7"]["outcome"] in ("M1", "M2", "M0", "U") and None not in r["tilts"].values()
    rp = r["reproduction_BASE_vs_C1v5"]
    assert rp["status"] == "complete" and rp["max_relative_difference"] == 0.0 and rp["max_relative_difference_input_mag"] == 0.0
    assert r["tilts"]["FULL"] != r["tilts"]["BASE"] and r["modules_on"]["QKV2"] == 1
    assert set(r["diagnostic_groups"]) == {"layer2", "layers3_51", "layers6_51"}
    assert r["bootstrap_resamples_drawn"] == 1000 and r["bootstrap_resamples_used"]["full_change"] == 1000
    json.dumps(r)
    rec = copy.deepcopy(rec)                                                   # the record shares lists with book
    rec["per_N"][str(GRID[0])]["target"][0][0] *= 1.5
    (c1 / "book_B_7.json").write_text(json.dumps(rec))
    assert abs(analyze_c7.reproduction([book], GRID, str(c1))["max_relative_difference"] - 1 / 3) < 1e-6
    empty = tmp_path / "c1empty"; empty.mkdir()
    rp = analyze_c7.reproduction([book], GRID, str(empty))
    assert rp["status"] == "incomplete" and rp["books_missing"] == [7]
    # partial: a spec expecting a second book gives no outcome, numbers kept apart
    spec2 = copy.deepcopy(spec); spec2["inputs"]["prompt_sets"]["B"][9] = "z"
    with pytest.raises(SystemExit, match="Incomplete"):
        analyze_c7.analyze(str(out), spec2)
    rpart = analyze_c7.analyze(str(out), spec2, partial=True)
    assert rpart["C7"]["outcome"] is None and "partial_diagnostic" in rpart["C7"] and "warning" in rpart
    # refusals: QKV2 modules not as sealed, an unclean sham, a wrong seal
    spec3 = copy.deepcopy(spec); spec3["conditions"]["QKV2"]["modules"] = ["layers.3.self_attn.o_proj"]
    with pytest.raises(SystemExit, match="QKV2 modules"):
        analyze_c7.analyze(str(out), spec3)
    with pytest.raises(SystemExit, match="seal hash"):
        analyze_c7.analyze(str(out), spec, "otherseal")
    book["sham_differences"]["FULL"] = 3
    (out / "book_B_7.json").unlink(); (out / "book_B_7.json").write_text(json.dumps(book))
    with pytest.raises(SystemExit, match="sham"):
        analyze_c7.analyze(str(out), spec)


def test_phase_a_stops_by_rule_and_budget(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    monkeypatch.setattr(run_c7, "PHASE_A_N", (3, 768))
    out = tmp_path / "run"; out.mkdir()
    (out / "STARTED.json").write_text(json.dumps({"seal_sha256": "testseal", "utc": "2026"}))
    monkeypatch.setattr(run_c7.metrics, "aggregate_snr", lambda *a: 0.0)        # underpowered under every condition
    with torch.no_grad(), pytest.raises(SystemExit, match="underpowered"):
        run_c7.phase_a(s, str(out))
    assert json.load(open(out / "phaseA.json"))["stopped"]
    assert run_c7.run_state(str(out), spec) == "stopped"
    with pytest.raises(SystemExit, match="stopped"):
        run_c7.admit(str(tmp_path), str(tmp_path / "new"), "testseal", spec, restart_reason="x")
    with pytest.raises(SystemExit, match="Phase A check"):
        run_c7.phase_b(s, str(out))
    out2 = tmp_path / "run2"; out2.mkdir()
    s.ledger.cap_s = 1.0
    s.spec = dict(s.spec, budget={"max_gpu_hours": 1, "seconds_per_pass_estimate": 100})
    with pytest.raises(SystemExit, match="Budget cap"):
        run_c7.phase_a(s, str(out2))
    assert not (out2 / "phaseA.json").exists()


def test_setup_check_failure_writes_stopped_record(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    out = tmp_path / "run"; out.mkdir()
    monkeypatch.setattr(run_c7.Session, "adapter_checks", lambda self: {"as_expected": False})
    with pytest.raises(SystemExit, match="setup checks failed"):
        run_c7.phase_a(s, str(out))
    pa = json.load(open(out / "phaseA.json"))
    assert pa["stopped"] and pa["outcome_if_stopped"] == "U" and not list(out.glob("phaseA.FAILED.*"))


def test_continue_rules(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    spec = {"inputs": {"prompt_sets": {"B": {7: "y"}}}}
    run_c7.admit(str(raw), str(raw / "c7v1-a"), "seal", spec)
    (raw / "c7v1-a" / "phaseA.FAILED.20260101T000000000000Z.json").write_text("{}")
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c7.admit(str(raw), str(raw / "c7v1-a"), "seal", spec, new=False)
    run_c7.admit(str(raw), str(raw / "c7v1-a"), "seal", spec, restart_reason="pod crashed", new=False)
    with pytest.raises(SystemExit, match="STARTED"):
        run_c7.admit(str(raw), str(raw / "c7v1-zzz"), "seal", spec, new=False)


def test_wrong_and_unfilled_hash_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="sha256"):
        session(tmp_path, monkeypatch, sha="0" * 64)
    (tmp_path / "b").mkdir(); (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit, match="sha256"):
        session(tmp_path / "b", monkeypatch, sha="FILL_AT_SEAL")
    s, _ = session(tmp_path / "c", monkeypatch, sha="FILL_AT_SEAL", allow_unfilled=True)   # the smoke path only
    assert len(s.adapter_sha) == 64


def test_admission(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    spec = {"inputs": {"prompt_sets": {"B": {7: "y"}}}}
    run_c7.admit(str(raw), str(raw / "c7v1-a"), "seal", spec)
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c7.admit(str(raw), str(raw / "c7v1-b"), "seal", spec)
    run_c7.admit(str(raw), str(raw / "c7v1-b"), "seal", spec, restart_reason="pod crashed")
    with pytest.raises(SystemExit, match="newer"):
        run_c7.admit(str(raw), str(raw / "c7v1-a"), "seal", spec, new=False)
    run_c7.admit(str(raw), str(raw / "c7v1-b"), "seal", spec, new=False)
    (raw / "c7v1-b" / "book_B_7.json").write_text("{}")
    with pytest.raises(SystemExit, match="complete"):
        run_c7.admit(str(raw), str(raw / "c7v1-c"), "seal", spec, restart_reason="again")
    (raw / "c7v1-a" / "phaseA.json").write_text(json.dumps({"stopped": "not linear"}))
    assert run_c7.run_state(str(raw / "c7v1-a"), spec) == "stopped"


def test_outcome_reasons():
    t = {"BASE": 0.14, "FULL": 0.105, "QKV2": 0.112, "REST": 0.138}
    ok = [dict(t)] * 5
    assert metrics_c7.localization_outcome(t, ok)["reason"] is None
    assert "outside" in metrics_c7.localization_outcome(dict(t, QKV2=0.09), ok)["reason"]
    assert "books agree" in metrics_c7.localization_outcome(t, ok[:3] + [dict(t, QKV2=None)] * 2)["reason"]
    assert "share(REST)" in metrics_c7.localization_outcome(dict(t, REST=0.152), ok)["reason"]
