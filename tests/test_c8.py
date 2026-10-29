"""Tests for C8: outcome rule, six condition masks, Phase A and B on the toy model with a four-layer adapter, analysis,
the reproduction check, admission and refusals. Forked from test_c7.py."""
import copy
import json

import numpy as np
import pytest
import torch

from dialedin import analyze_c8 as analyze_c7, metrics_c8, run_c5, run_c8 as run_c7
from dialedin import pilot_c5 as P
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector
from test_c4 import D
from test_c5 import GRID, toy_spec, toy_world

# the toy has one linear module per layer (o_proj); each condition gets a distinct set, standing in for the real ones
MODS = {"QKV2": ["layers.2.self_attn.o_proj", "layers.3.self_attn.o_proj"], "V2": ["layers.2.self_attn.o_proj"],
        "QK2": ["layers.3.self_attn.o_proj"], "V4": ["layers.4.self_attn.o_proj"]}
QKV = MODS["V2"]


def make_adapter(tmp_path):
    spec = toy_spec()
    spec["training"]["layers"] = [2, 3, 4, 5]
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
            "conditions": {"BASE": {"scale": 0.0, "modules": "all"}, "FULL": {"scale": 2.0, "modules": "all"},
                           **{c: {"scale": 2.0, "modules": list(m)} for c, m in MODS.items()}},
            "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}


def session(tmp_path, monkeypatch, spec=None, sha=None, allow_unfilled=False):
    monkeypatch.setattr(run_c7, "CHECK_LAYER", 5)          # after every adapted toy layer, as layer 10 is on Glimmer
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


def test_one_matrix_outcome():
    t = {"BASE": 0.14, "FULL": 0.10, "QKV2": 0.108, "V2": 0.118, "QK2": 0.128, "V4": 0.138}
    ok = [dict(t)] * 5
    o = metrics_c8.one_matrix_outcome(t, ok)                           # V2 share 0.55, V4 0.05
    assert o["outcome"] == "V1" and abs(o["share_V2"] - 0.55) < 1e-9 and o["reason"] is None
    assert abs(o["share_V2_of_QKV2"] - 0.55 / 0.8) < 1e-9 and abs(o["additivity_V2_plus_QK2_over_QKV2"] - 0.85 / 0.8) < 1e-9
    assert metrics_c8.one_matrix_outcome(dict(t, V4=0.126), ok)["outcome"] == "V2S"     # control share 0.35
    assert metrics_c8.one_matrix_outcome(dict(t, V4=0.154), ok)["outcome"] == "V2S"     # control opposes: -0.35
    low = metrics_c8.one_matrix_outcome(dict(t, V2=0.122), ok)                          # share 0.45
    assert low["outcome"] == "V0" and "outside" in low["reason"]
    assert metrics_c8.one_matrix_outcome(dict(t, V2=0.083), ok)["outcome"] == "V0"     # share 1.43: overshoot
    weak = dict(t, V2=0.1385)
    assert not metrics_c8.book_agrees(weak) and metrics_c8.book_agrees(t)
    assert metrics_c8.one_matrix_outcome(t, ok[:4] + [weak])["outcome"] == "V1"
    assert "books agree" in metrics_c8.one_matrix_outcome(t, ok[:3] + [weak, weak])["reason"]
    assert metrics_c8.one_matrix_outcome(dict(t, FULL=0.122), ok)["outcome"] == "U"    # FULL only -12.9%
    assert metrics_c8.one_matrix_outcome(dict(t, BASE=-0.01), ok)["outcome"] == "U"
    assert metrics_c8.one_matrix_outcome(dict(t, V4=None), ok)["outcome"] == "U"
    edge = metrics_c8.one_matrix_outcome(dict(t, V2=0.1198, V4=0.1282), ok)            # shares 0.505 and 0.295
    assert edge["outcome"] == "V1"


def test_condition_modules_and_spec_checks():
    names = [f"layers.{l}.self_attn.{m}" for l in (2, 3, 4, 5) for m in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")]
    q, k, v = (f"layers.2.self_attn.{m}" for m in ("q_proj", "k_proj", "v_proj"))
    spec = {"conditions": {"BASE": {"modules": "all"}, "FULL": {"modules": "all"}, "QKV2": {"modules": [q, k, v]},
                           "V2": {"modules": [v]}, "QK2": {"modules": [q, k]}, "V4": {"modules": ["layers.4.self_attn.v_proj"]}}}
    on = run_c7.condition_modules(spec, names)
    assert on["QKV2"] == [q, k, v] and on["V2"] == [v] and on["FULL"] == names and on["V4"] == ["layers.4.self_attn.v_proj"]
    bad = copy.deepcopy(spec); bad["conditions"]["V4"]["modules"] = ["layers.9.self_attn.v_proj"]
    with pytest.raises(SystemExit, match="V4"):
        run_c7.condition_modules(bad, names)
    same = copy.deepcopy(spec); same["conditions"]["QK2"]["modules"] = [v]
    with pytest.raises(SystemExit, match="same modules"):
        run_c7.condition_modules(same, names)
    good = {"conditions": {c: {"scale": run_c7.SCALES[c]} for c in run_c7.ORDER}}
    run_c7.check_condition_spec(good)
    bad = copy.deepcopy(good); bad["conditions"]["V2"]["scale"] = 1.5
    with pytest.raises(SystemExit, match="scales"):
        run_c7.check_condition_spec(bad)
    extra = copy.deepcopy(good); extra["conditions"]["REST"] = {"scale": 2.0}
    with pytest.raises(SystemExit, match="conditions"):
        run_c7.check_condition_spec(extra)


def test_placeholders_refused():
    run_c7.check_no_placeholders({"a": {"b": ["x", "fine"]}, "c": 1.0})
    with pytest.raises(SystemExit, match="placeholder"):
        run_c7.check_no_placeholders({"budget": {"pre_seal_smoke": "SMOKE_RESULT"}})


def test_masks_restore_exactly_and_checks_pass(tmp_path, monkeypatch):
    s, _ = session(tmp_path, monkeypatch)
    assert len(s.adapter_modules) == 4
    saved = [b.clone() for b in s.params[1::2]]
    s.set_condition("V2")
    assert torch.equal(s.params[1], saved[0]) and not s.params[3].any() and not s.params[5].any() and P.LoRALinear.scale == 2.0
    s.set_condition("BASE")
    assert all(torch.equal(b, x) for b, x in zip(s.params[1::2], saved)) and P.LoRALinear.scale == 0.0
    ad = s.adapter_checks()
    assert ad["as_expected"], ad
    assert ad["modules_on"] == {"BASE": 4, "FULL": 4, "QKV2": 2, "V2": 1, "QK2": 1, "V4": 1}
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
    assert sorted(book["per_condition"]) == sorted(metrics_c8.CONDITIONS) and book["sham_differences_close"] == 0
    assert book["record"]["conditions"]["V2"]["modules_on"] == QKV and P.LoRALinear.scale == 0.0
    # a fake C1 v5 record from BASE's own cells: the reproduction check must read 0 and be complete
    c1 = tmp_path / "c1"; c1.mkdir()
    rec = {"median_norm": book["amplitude_reference"]["BASE"],
           "per_N": {str(N): {"target": book["per_condition"]["BASE"][str(N)]["target"],
                              "input_mag": book["per_condition"]["BASE"][str(N)]["input_mag"]} for N in GRID}}
    (c1 / "book_B_7.json").write_text(json.dumps(rec))
    r = analyze_c7.analyze(str(out), spec, "testseal", {"f": "h"}, c1_run_dir=str(c1))
    assert r["complete"] and r["C8"]["outcome"] in ("V1", "V2S", "V0", "U") and None not in r["tilts"].values()
    rp = r["reproduction_BASE_vs_C1v5"]
    assert rp["status"] == "complete" and rp["max_relative_difference"] == 0.0 and rp["max_relative_difference_input_mag"] == 0.0
    assert r["tilts"]["FULL"] != r["tilts"]["BASE"] and r["modules_on"]["V2"] == 1 and r["modules_on"]["QKV2"] == 2
    assert set(r["diagnostic_groups"]) == {"layer2", "layers3_51", "layers6_51"}
    assert r["bootstrap_resamples_drawn"] == 1000 and r["bootstrap_resamples_used"]["full_change"] == 1000
    assert set(r["bootstrap_95"]) == {"share_V2", "share_V4", "share_QKV2", "share_QK2", "full_change"}
    assert r["secondary_mean_book_share_V2"]["books"] in (0, 1)
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
    assert rpart["C8"]["outcome"] is None and "partial_diagnostic" in rpart["C8"] and "warning" in rpart
    # refusals: QKV2 modules not as sealed, an unclean sham, a wrong seal
    spec3 = copy.deepcopy(spec); spec3["conditions"]["V4"]["modules"] = ["layers.5.self_attn.o_proj"]
    with pytest.raises(SystemExit, match="V4 modules"):
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
    run_c7.admit(str(raw), str(raw / "c8v1-a"), "seal", spec)
    (raw / "c8v1-a" / "phaseA.FAILED.20260101T000000000000Z.json").write_text("{}")
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c7.admit(str(raw), str(raw / "c8v1-a"), "seal", spec, new=False)
    run_c7.admit(str(raw), str(raw / "c8v1-a"), "seal", spec, restart_reason="pod crashed", new=False)
    with pytest.raises(SystemExit, match="STARTED"):
        run_c7.admit(str(raw), str(raw / "c8v1-zzz"), "seal", spec, new=False)


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
    run_c7.admit(str(raw), str(raw / "c8v1-a"), "seal", spec)
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c7.admit(str(raw), str(raw / "c8v1-b"), "seal", spec)
    run_c7.admit(str(raw), str(raw / "c8v1-b"), "seal", spec, restart_reason="pod crashed")
    with pytest.raises(SystemExit, match="newer"):
        run_c7.admit(str(raw), str(raw / "c8v1-a"), "seal", spec, new=False)
    run_c7.admit(str(raw), str(raw / "c8v1-b"), "seal", spec, new=False)
    (raw / "c8v1-b" / "book_B_7.json").write_text("{}")
    with pytest.raises(SystemExit, match="complete"):
        run_c7.admit(str(raw), str(raw / "c8v1-c"), "seal", spec, restart_reason="again")
    (raw / "c8v1-a" / "phaseA.json").write_text(json.dumps({"stopped": "not linear"}))
    assert run_c7.run_state(str(raw / "c8v1-a"), spec) == "stopped"
