"""Tests for C9: the necessity outcome rule, all_except masks, the sealed token segment, Phase A and B on the toy model
with a four-layer adapter, analysis with both read windows, admission, retry reasons and refusals. Forked from
test_c8.py."""
import copy
import json

import numpy as np
import pytest
import torch

from dialedin import analyze_c9, metrics_c9, run_c5, run_c9
from dialedin import pilot_c5 as P
from dialedin.hooks import HeadProbe
from dialedin.wave import WaveInjector
from test_c4 import D
from test_c5 import GRID, toy_spec, toy_world

# the toy has one linear module per layer (o_proj), layers 2 to 5; each condition gets a distinct set, standing in for
# the real ones (V2: layer 2's module alone; NOV2, NOV4, NOG2: all but one module)
M = {l: f"layers.{l}.self_attn.o_proj" for l in (2, 3, 4, 5)}
MODS = {"V2": [M[2]], "NOV2": {"all_except": [M[2]]}, "NOV4": {"all_except": [M[4]]}, "NOG2": {"all_except": [M[3]]}}
START = 7                                     # the toy texts have T + 10 tokens


def make_adapter(tmp_path):
    spec = toy_spec()
    spec["training"]["layers"] = [2, 3, 4, 5]
    lm, full, texts, tok = toy_world(tmp_path)
    out = tmp_path / "c5run"; out.mkdir()
    ledger = run_c5.Ledger(str(tmp_path / "ledger5"), "testseal0000", 3600)
    run_c5.train(spec, "testseal", {"f": "h"}, ledger, None, str(out), loaded=(texts, full, lm, lm.layers, tok, {"global_layers": [3]}))
    return str(out / "adapter.pt"), run_c9.sha_file(str(out / "adapter.pt")), spec


def c9_spec(path, sha, c5spec, start=START):
    tr = c5spec["training"]
    return {"inputs": {"prompt_sets": {"A": {7: "y"}}, "text_sha256": c5spec["inputs"]["text_sha256"],
                       "length_tokens": c5spec["inputs"]["length_tokens"], "segment_start_token": start},
            "drive": {"periods_N": GRID},
            "adapter": {"file": path, "sha256": sha, "family": tr["family"], "layers": tr["layers"],
                        "rank": tr["rank"], "alpha": tr["alpha"]},
            "conditions": {"BASE": {"scale": 0.0, "modules": "all"}, "FULL": {"scale": 2.0, "modules": "all"},
                           **{c: {"scale": 2.0, "modules": copy.deepcopy(m)} for c, m in MODS.items()}},
            "budget": {"max_gpu_hours": 1, "seconds_per_pass_estimate": 0.01}}


def session(tmp_path, monkeypatch, spec=None, sha=None, allow_unfilled=False, start=START):
    monkeypatch.setattr(run_c9, "CHECK_LAYER", 5)          # after every adapted toy layer, as layer 10 is on Glimmer
    monkeypatch.setattr(run_c9, "PHASE_A_BOOK", 7)
    path, real_sha, c5spec = make_adapter(tmp_path)
    P.configure(8192)
    spec = spec or c9_spec(path, sha or real_sha, c5spec, start)
    lm, full, texts, tok = toy_world(tmp_path)
    s = object.__new__(run_c9.Session)
    s.repo, s.spec, s.seal_hash, s.code = "", spec, "testseal", {"f": "h"}
    s.ledger = run_c9.Ledger(str(tmp_path / "ledger9"), "testseal0000", 3600)
    s.texts, s.tok, s.model, s.decoder, s.layers, s.platform = texts, tok, full, lm, lm.layers, {"global_layers": [3]}
    run_c9.Session._setup(s, allow_unfilled)
    for hd in s.probe.handles:
        hd.remove()
    s.probe = HeadProbe(lm.layers, s.layer_ids, 4, s.t1)
    d = torch.zeros(D); d[0] = 1.0
    s.injector.remove(); s.direction = d
    s.injector = WaveInjector(lm.layers[1], d)
    return s, spec


# main: BASE 0.14, FULL 0.10 (-28.6 percent). V2 keeps 0.55, NOV2 0.45, NOV4 0.95, NOG2 0.97.
MAIN = {"BASE": 0.14, "FULL": 0.10, "V2": 0.118, "NOV2": 0.122, "NOV4": 0.102, "NOG2": 0.1012}
# downstream: BASE 0.156, FULL 0.150 (-3.8 percent). V2 keeps 0.5, NOV2 0.5, NOV4 0.9.
DOWNT = {"BASE": 0.156, "FULL": 0.150, "V2": 0.153, "NOV2": 0.153, "NOV4": 0.1506, "NOG2": 0.1505}


def test_necessity_outcome():
    ok = [dict(MAIN)] * 5
    o = metrics_c9.necessity_outcome(MAIN, DOWNT, ok)
    assert o["outcome"] == "N1" and o["reason"] is None and o["books_ordered"] == 5
    assert abs(o["retained_NOV2"] - 0.45) < 1e-9 and abs(o["retained_NOG2"] - 0.97) < 1e-9
    assert abs(o["gap_sufficiency_minus_necessity"] - (0.55 - 0.55)) < 1e-9
    d = o["downstream"]
    assert d["judged"] and d["passes"] and abs(d["retained_NOV2"] - 0.5) < 1e-9 and abs(d["retained_NOV4"] - 0.9) < 1e-9
    # main passes, downstream fails its control: N1L
    o = metrics_c9.necessity_outcome(MAIN, dict(DOWNT, NOV4=0.1536), ok)         # NOV4 keeps 0.4
    assert o["outcome"] == "N1L" and "layer 4 v" in o["reason"]
    # downstream FULL too small to judge: N1L with the floor named
    o = metrics_c9.necessity_outcome(MAIN, dict(DOWNT, FULL=0.153, V2=0.1545, NOV2=0.1545, NOV4=0.1533), ok)
    assert o["outcome"] == "N1L" and not o["downstream"]["judged"] and "less than" in o["reason"]
    # downstream undefined: N1L, not U
    o = metrics_c9.necessity_outcome(MAIN, dict(DOWNT, NOV2=None), ok)
    assert o["outcome"] == "N1L" and not o["downstream"]["judged"]
    # main fails: NOV2 keeps too much; NOG2 keeps too little; too few books
    assert metrics_c9.necessity_outcome(dict(MAIN, NOV2=0.114), DOWNT, ok)["outcome"] == "N0"          # keeps 0.65
    low = metrics_c9.necessity_outcome(dict(MAIN, NOG2=0.11), DOWNT, ok)                                # keeps 0.75
    assert low["outcome"] == "N0" and "layer 2 gate" in low["reason"]
    flip = dict(MAIN, NOV2=0.1012, NOG2=0.122)                                     # the book orders them the other way
    assert not metrics_c9.book_orders(flip) and metrics_c9.book_orders(MAIN)
    assert metrics_c9.necessity_outcome(MAIN, DOWNT, ok[:4] + [flip])["outcome"] == "N1"
    assert "books" in metrics_c9.necessity_outcome(MAIN, DOWNT, ok[:3] + [flip, flip])["reason"]
    # replication gate and U
    r0 = metrics_c9.necessity_outcome(dict(MAIN, V2=0.13), DOWNT, ok)                                   # V2 keeps 0.25
    assert r0["outcome"] == "R0" and not r0["main_passes"]
    assert metrics_c9.necessity_outcome(dict(MAIN, FULL=0.122), DOWNT, ok)["outcome"] == "U"            # -12.9 percent
    assert metrics_c9.necessity_outcome(dict(MAIN, BASE=-0.01), DOWNT, ok)["outcome"] == "U"
    assert metrics_c9.necessity_outcome(dict(MAIN, NOG2=None), DOWNT, ok)["outcome"] == "U"
    # just inside both main bars (NOV2 keeps 0.5995, NOG2 0.8005) passes; just outside fails
    assert metrics_c9.necessity_outcome(dict(MAIN, NOV2=0.11602, NOG2=0.10798), DOWNT, ok)["main_passes"]
    assert not metrics_c9.necessity_outcome(dict(MAIN, NOV2=0.11598), DOWNT, ok)["main_passes"]


def test_condition_modules_and_spec_checks():
    names = [f"layers.{l}.self_attn.{m}" for l in (2, 3, 4, 5) for m in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")]
    v2, v4, o2 = "layers.2.self_attn.v_proj", "layers.4.self_attn.v_proj", "layers.2.self_attn.gate_proj"
    spec = {"conditions": {"BASE": {"modules": "all"}, "FULL": {"modules": "all"}, "V2": {"modules": [v2]},
                           "NOV2": {"modules": {"all_except": [v2]}}, "NOV4": {"modules": {"all_except": [v4]}},
                           "NOG2": {"modules": {"all_except": [o2]}}}}
    on = run_c9.condition_modules(spec, names)
    assert on["V2"] == [v2] and on["FULL"] == names and len(on["NOV2"]) == len(names) - 1 and v2 not in on["NOV2"]
    assert v4 not in on["NOV4"] and o2 not in on["NOG2"] and v2 in on["NOG2"]
    for bad_mods in ({"all_except": ["layers.9.self_attn.v_proj"]}, {"all_except": []}, {"all_except": [v2, v2]},
                     {"all_except": names}, {"all_except": [v2], "also": 1}):
        bad = copy.deepcopy(spec); bad["conditions"]["NOV4"]["modules"] = bad_mods
        with pytest.raises(SystemExit, match="NOV4"):
            run_c9.condition_modules(bad, names)
    same = copy.deepcopy(spec); same["conditions"]["NOG2"]["modules"] = {"all_except": [v2]}
    with pytest.raises(SystemExit, match="same modules"):
        run_c9.condition_modules(same, names)
    good = {"conditions": {c: {"scale": run_c9.SCALES[c]} for c in run_c9.ORDER}}
    run_c9.check_condition_spec(good)
    bad = copy.deepcopy(good); bad["conditions"]["NOV2"]["scale"] = 1.5
    with pytest.raises(SystemExit, match="scales"):
        run_c9.check_condition_spec(bad)
    extra = copy.deepcopy(good); extra["conditions"]["QKV2"] = {"scale": 2.0}
    with pytest.raises(SystemExit, match="conditions"):
        run_c9.check_condition_spec(extra)


def test_placeholders_refused():
    run_c9.check_no_placeholders({"a": {"b": ["x", "fine"]}, "c": 1.0})
    with pytest.raises(SystemExit, match="placeholder"):
        run_c9.check_no_placeholders({"budget": {"pre_seal_smoke": "SMOKE_RESULT"}})


def test_segment_is_the_sealed_tokens(tmp_path, monkeypatch):
    s, _ = session(tmp_path, monkeypatch)
    ids, _ = s.ids(7)
    full = s.tok("y")["input_ids"]
    assert ids[0].tolist() == full[START:START + s.T] and s.record(7, "x")["segment_tokens"] == [START, START + s.T]
    (tmp_path / "b").mkdir()
    s2, _ = session(tmp_path / "b", monkeypatch, start=11)                 # T + 11 > T + 10 tokens: too short
    with pytest.raises(SystemExit, match="too short"):
        s2.ids(7)


def test_masks_restore_exactly_and_checks_pass(tmp_path, monkeypatch):
    s, _ = session(tmp_path, monkeypatch)
    assert len(s.adapter_modules) == 4
    saved = [b.clone() for b in s.params[1::2]]
    s.set_condition("NOV2")
    assert not s.params[1].any() and all(torch.equal(s.params[2 * i + 1], saved[i]) for i in (1, 2, 3))
    assert P.LoRALinear.scale == 2.0
    s.set_condition("BASE")
    assert all(torch.equal(b, x) for b, x in zip(s.params[1::2], saved)) and P.LoRALinear.scale == 0.0
    ad = s.adapter_checks()
    assert ad["as_expected"], ad
    assert ad["modules_on"] == {"BASE": 4, "FULL": 4, "V2": 1, "NOV2": 3, "NOV4": 3, "NOG2": 3}
    assert s.logits_check()["as_expected"]


def test_phases_and_analysis_end_to_end(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    monkeypatch.setattr(run_c9, "PHASE_A_N", (3, 768))
    out = tmp_path / "run"; out.mkdir()
    with torch.no_grad():
        run_c9.phase_a(s, str(out))
    a = json.load(open(out / "phaseA.json"))
    assert a["stopped"] is None and [r["condition"] for r in a["rows"]] == list(run_c9.ORDER)
    with pytest.raises(SystemExit, match="No book files"):                     # Phase A only: nothing to look at
        analyze_c9.analyze(str(out), spec, partial=True)
    (out / "book_A_7.ATTEMPT.20260101T000000000000Z.json").write_text("{}")      # an earlier attempt
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c9.phase_b(s, str(out))
    with torch.no_grad():
        run_c9.phase_b(s, str(out), restart_reason="test retry")
    book = json.load(open(out / "book_A_7.json"))
    assert book["restart_reason"] == "test retry" and book["record"]["segment_tokens"] == [START, START + s.T]
    assert sorted(book["per_condition"]) == sorted(metrics_c9.CONDITIONS) and book["sham_differences_close"] == 0
    assert book["record"]["conditions"]["NOV2"]["modules_on"] == [M[3], M[4], M[5]] and P.LoRALinear.scale == 0.0
    r = analyze_c9.analyze(str(out), spec, "testseal", {"f": "h"})
    assert r["complete"] and r["C9"]["outcome"] in ("N1", "N1L", "N0", "R0", "U") and None not in r["tilts"].values()
    assert set(r["tilts_downstream"]) == set(metrics_c9.CONDITIONS)
    assert r["tilts"]["FULL"] != r["tilts"]["BASE"] and r["modules_on"]["NOV2"] == 3 and r["modules_on"]["V2"] == 1
    assert set(r["diagnostic_groups"]) == {"layer2", "layers3_51", "layers5_51", "layers6_51"}
    assert r["diagnostic_groups"]["layers6_51"] is None                       # the toy has no layer 6
    assert r["diagnostic_groups"]["layers5_51"]["tilts"] == r["tilts_downstream"]     # one definition of downstream
    assert r["reproduction"].startswith("not applicable")
    assert r["bootstrap_resamples_drawn"] == 1000 and r["bootstrap_resamples_used"]["full_change"] == 1000
    assert set(r["bootstrap_95"]) == {"retained_V2", "retained_NOV2", "retained_NOV4", "retained_NOG2", "full_change",
                                      "retained5_V2", "retained5_NOV2", "retained5_NOV4", "full_change5"}
    assert r["secondary_mean_of_book_shares"]["books"] == 1
    assert r["book_tilts"][0]["tilts_downstream"].keys() == set(metrics_c9.CONDITIONS)
    json.dumps(r)
    # partial: a spec expecting a second book gives no outcome, numbers kept apart
    spec2 = copy.deepcopy(spec); spec2["inputs"]["prompt_sets"]["A"][9] = "z"
    with pytest.raises(SystemExit, match="Incomplete"):
        analyze_c9.analyze(str(out), spec2)
    rpart = analyze_c9.analyze(str(out), spec2, partial=True)
    assert rpart["C9"]["outcome"] is None and "partial_diagnostic" in rpart["C9"] and "warning" in rpart
    # refusals: a condition's modules not as sealed (all_except and list forms), an unclean sham, a wrong seal
    spec3 = copy.deepcopy(spec); spec3["conditions"]["NOV4"]["modules"] = {"all_except": [M[5]]}
    with pytest.raises(SystemExit, match="NOV4 modules"):
        analyze_c9.analyze(str(out), spec3)
    spec4 = copy.deepcopy(spec); spec4["conditions"]["V2"]["modules"] = [M[3]]
    with pytest.raises(SystemExit, match="V2 modules"):
        analyze_c9.analyze(str(out), spec4)
    with pytest.raises(SystemExit, match="seal hash"):
        analyze_c9.analyze(str(out), spec, "otherseal")
    book["sham_differences"]["FULL"] = 3
    (out / "book_A_7.json").unlink(); (out / "book_A_7.json").write_text(json.dumps(book))
    with pytest.raises(SystemExit, match="sham"):
        analyze_c9.analyze(str(out), spec)


def test_phase_a_stops_by_rule_and_budget(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    monkeypatch.setattr(run_c9, "PHASE_A_N", (3, 768))
    out = tmp_path / "run"; out.mkdir()
    (out / "STARTED.json").write_text(json.dumps({"seal_sha256": "testseal", "utc": "2026"}))
    monkeypatch.setattr(run_c9.metrics, "aggregate_snr", lambda *a: 0.0)        # underpowered under every condition
    with torch.no_grad(), pytest.raises(SystemExit, match="underpowered"):
        run_c9.phase_a(s, str(out))
    assert json.load(open(out / "phaseA.json"))["stopped"]
    assert run_c9.run_state(str(out), spec) == "stopped"
    with pytest.raises(SystemExit, match="stopped"):
        run_c9.admit(str(tmp_path), str(tmp_path / "new"), "testseal", spec, restart_reason="x")
    with pytest.raises(SystemExit, match="Phase A check"):
        run_c9.phase_b(s, str(out))
    out2 = tmp_path / "run2"; out2.mkdir()
    s.ledger.cap_s = 1.0
    s.spec = dict(s.spec, budget={"max_gpu_hours": 1, "seconds_per_pass_estimate": 100})
    with pytest.raises(SystemExit, match="Budget cap"):
        run_c9.phase_a(s, str(out2))
    assert not (out2 / "phaseA.json").exists()


def test_setup_check_failure_writes_stopped_record(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    out = tmp_path / "run"; out.mkdir()
    (out / "startup.FAILED.20260101T000000000000Z.json").write_text("{}")
    monkeypatch.setattr(run_c9.Session, "adapter_checks", lambda self: {"as_expected": False})
    with pytest.raises(SystemExit, match="setup checks failed"):
        run_c9.phase_a(s, str(out))
    pa = json.load(open(out / "phaseA.json"))
    assert pa["stopped"] and pa["outcome_if_stopped"] == "U" and not list(out.glob("phaseA.FAILED.*"))
    assert pa["earlier_failures"] == ["startup.FAILED.20260101T000000000000Z.json"]


def test_distinctness_failure_stops_the_run(tmp_path, monkeypatch):
    s, spec = session(tmp_path, monkeypatch)
    real = run_c9.Session._reference_input
    monkeypatch.setattr(run_c9.Session, "_reference_input",
                        lambda self: real(self) if self.condition in ("BASE", None) else real(self) * 0 + 1)
    ad = s.adapter_checks()
    assert not ad["all_conditions_distinct"] and not ad["as_expected"]


def test_retry_reason_after_any_failure_and_stamped(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    spec = {"inputs": {"prompt_sets": {"A": {7: "y"}}}}
    run_c9.admit(str(raw), str(raw / "c9v1-a"), "seal", spec)
    (raw / "c9v1-a" / "startup.FAILED.20260101T000000000000Z.json").write_text("{}")
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c9.admit(str(raw), str(raw / "c9v1-a"), "seal", spec, new=False)
    run_c9.admit(str(raw), str(raw / "c9v1-a"), "seal", spec, restart_reason="pod lost the GPU", new=False)
    (tmp_path / "w").mkdir()
    s, _ = session(tmp_path / "w", monkeypatch)
    monkeypatch.setattr(run_c9, "PHASE_A_N", (3, 768))
    with torch.no_grad():
        run_c9.phase_a(s, str(raw / "c9v1-a"), restart_reason="pod lost the GPU")
    pa = json.load(open(raw / "c9v1-a" / "phaseA.json"))
    assert pa["restart_reason"] == "pod lost the GPU" and pa["earlier_failures"] == ["startup.FAILED.20260101T000000000000Z.json"]


def test_admission_and_continue(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    spec = {"inputs": {"prompt_sets": {"A": {7: "y"}}}}
    run_c9.admit(str(raw), str(raw / "c9v1-a"), "seal", spec)
    with pytest.raises(SystemExit, match="restart-reason"):
        run_c9.admit(str(raw), str(raw / "c9v1-b"), "seal", spec)
    run_c9.admit(str(raw), str(raw / "c9v1-b"), "seal", spec, restart_reason="pod crashed")
    with pytest.raises(SystemExit, match="newer"):
        run_c9.admit(str(raw), str(raw / "c9v1-a"), "seal", spec, new=False)
    run_c9.admit(str(raw), str(raw / "c9v1-b"), "seal", spec, new=False)
    with pytest.raises(SystemExit, match="STARTED"):
        run_c9.admit(str(raw), str(raw / "c9v1-zzz"), "seal", spec, new=False)
    (raw / "c9v1-b" / "book_A_7.json").write_text("{}")
    with pytest.raises(SystemExit, match="complete"):
        run_c9.admit(str(raw), str(raw / "c9v1-c"), "seal", spec, restart_reason="again")


def test_wrong_and_unfilled_hash_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="sha256"):
        session(tmp_path, monkeypatch, sha="0" * 64)
    (tmp_path / "b").mkdir(); (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit, match="sha256"):
        session(tmp_path / "b", monkeypatch, sha="FILL_AT_SEAL")
    s, _ = session(tmp_path / "c", monkeypatch, sha="FILL_AT_SEAL", allow_unfilled=True)   # the smoke path only
    assert len(s.adapter_sha) == 64


def test_the_draft_spec_itself():
    """The real draft: 15 sealed text hashes (glimmer.load_texts' count rule), conditions and scales as the code,
    no placeholder but the smoke result, every all_except module in the C5 adapter, segment inside every book."""
    import os
    import yaml
    root = os.path.join(os.path.dirname(__file__), "..")
    spec = yaml.safe_load(open(os.path.join(root, "criteria/drafts/c9-glimmer-necessity-v1.yaml")))
    assert len(spec["inputs"]["text_sha256"]) == 15
    assert set(int(g) for g in spec["inputs"]["prompt_sets"]["A"]) <= set(int(g) for g in spec["inputs"]["text_sha256"])
    assert run_c9.PHASE_A_BOOK in [int(g) for g in spec["inputs"]["prompt_sets"]["A"]]
    run_c9.check_condition_spec(spec)
    smoke = spec["budget"].pop("pre_seal_smoke")
    run_c9.check_no_placeholders(spec)
    assert smoke == "SMOKE_RESULT" or "SMOKE_RESULT" not in smoke
    names = [f"layers.{l}.{m}" for l in (2, 3, 4, 5) for m in
             ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    on = run_c9.condition_modules(spec, names)
    assert {c: len(v) for c, v in on.items()} == {"BASE": 32, "FULL": 32, "V2": 1, "NOV2": 31, "NOV4": 31, "NOG2": 31}
    assert spec["inputs"]["segment_start_token"] + spec["inputs"]["length_tokens"] <= 94774   # shortest of the 15 books
