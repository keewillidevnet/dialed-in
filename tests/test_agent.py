"""Tests for the Muse agent: fact sheet, number checking, narration retry and proposal checks. No network."""
import json
import os

import numpy as np

from dialedin.agent import facts, verify
from dialedin.agent import __main__ as agent

GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]


def write_results(repo):
    d = repo / "results" / "derived"
    curve = (np.array(GRID, float) / 4096) ** 0.14
    c1 = {"books": 15, "grid": GRID,
          "C1": {"outcome": "T", "tilt_direction": "LP",
                 "per_set": {s: {"valid_points": 23, "rho_gain_vs_N": 1.0, "min_normalized": 0.35} for s in "ABC"},
                 "shape_replication": {"A-B": {"rho": 1.0}}},
          "curves": {s: {"whole": curve.tolist()} for s in "ABC"}}
    (d / "c1v5-x").mkdir(parents=True); json.dump(c1, open(d / "c1v5-x" / "c1_summary.json", "w"))
    c2 = {"C2": {"outcome": "K0", "direction": "+",
                 "per_set": {"A": {"tilts": {"G0.5": 0.1348, "G1.0": 0.1371}, "delta_global": 0.006, "delta_local": 0.0137}}},
          "book_signs": {"books": 15, "global_up_steeper": 14},
          "diagnostic_G1_vs_C1v5": {"status": "compared", "cells_compared": 105, "max_relative_difference": 0.0}}
    (d / "c2v1-x").mkdir(); json.dump(c2, open(d / "c2v1-x" / "c2_summary.json", "w"))


def test_fact_sheet(tmp_path):
    write_results(tmp_path)
    sh = facts.build(str(tmp_path))
    labels = [f["label"] for f in sh.facts]
    assert any("C1 outcome" in l for l in labels) and any("C2 outcome" in l for l in labels)
    tilt = [f for f in sh.facts if "tilt (slope" in f["label"]][0]
    assert abs(tilt["value"] - 0.14) < 1e-9
    assert all(f["source"].startswith("results/derived/") for f in sh.facts)


def test_verify_numbers_and_citations(tmp_path):
    write_results(tmp_path)
    sh = facts.build(str(tmp_path))
    fid = next(f["id"] for f in sh.facts if f["value"] == 0.1371)
    ok = verify.check(f"The tilt was 0.137 [{fid}] in C1 and C2 on 15 books.", sh)
    assert ok["ok"], ok
    bad = verify.check(f"The tilt rose by 23.7% [{fid}] and 0.555 [F999].", sh)
    assert not bad["ok"] and f"23.7% [{fid}]" in bad["mismatched_citations"] and bad["unknown_citations"] == ["F999"]
    assert "0.555" in verify.check("It was 0.555 overall.", sh)["unverified_numbers"]
    assert verify.check("Gain at 0.35 of the maximum, 35%.", sh)["ok"]       # percent of a fact


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return self.replies.pop(0)


def test_narrate_retries_on_unverified_numbers(tmp_path):
    write_results(tmp_path)
    sh = facts.build(str(tmp_path))
    out = tmp_path / "out"; out.mkdir()
    c = FakeClient(["The tilt changed by 12.5%.", "The C2 outcome was K0, with 105 cells reproduced exactly."])
    agent.narrate(c, sh, str(out))
    assert len(c.calls) == 2 and "12.5%" in c.calls[1][0][-1]["content"]
    text = open(next(out.glob("narration-*.md"))).read()
    assert "every number matches" in text


def test_propose_checks_fields_and_budget(tmp_path):
    write_results(tmp_path)
    sh = facts.build(str(tmp_path))
    out = tmp_path / "out"; out.mkdir()
    prop = {k: "x" for k in agent.SCHEMA["required"]}
    prop.update(conditions=["window 1024", "window 4096"], risks=["off-distribution"], gpu_hours_estimate=1.5)
    agent.propose(FakeClient(["Here it is:\n```json\n" + json.dumps(prop) + "\n```"]), sh, str(out))   # fenced: tolerated
    saved = json.load(open(next(out.glob("proposal-*.json"))))
    assert saved["_checks"]["missing_fields"] == [] and saved["_checks"]["within_budget"]
    out2 = tmp_path / "out2"; out2.mkdir()
    agent.propose(FakeClient([json.dumps(dict(prop, gpu_hours_estimate=2.5))]), sh, str(out2))
    assert not json.load(open(next(out2.glob("proposal-*.json"))))["_checks"]["within_budget"]


def test_fact_sheet_marks_probes_exploratory():
    repo = os.path.join(os.path.dirname(__file__), "..")
    sh = facts.build(repo)
    probe = [f for f in sh.facts if "probe-" in f["source"] or "headmap" in f["source"]]
    assert all(f["label"].startswith("EXPLORATORY") for f in probe)
    assert all(not f["label"].startswith("EXPLORATORY") for f in sh.facts if f["label"].startswith(("C5", "C6", "C7")))


def test_no_anthropic_sdk_under_src():
    root = os.path.join(os.path.dirname(__file__), "..", "src")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                assert "anthropic" not in open(os.path.join(dirpath, f)).read().lower(), f


def test_citation_must_match_its_fact():
    class S:
        facts = [{"id": "F1", "label": "a", "value": 0.268, "source": "x"}, {"id": "F2", "label": "b", "value": 32, "source": "x"},
                 {"id": "F3", "label": "c", "value": 8, "source": "x"}]
        def numbers(self):
            return [f["value"] for f in self.facts]
    sh = S()
    assert verify.check("The dial flattened it by 27 percent [F1] using 32 [F2] modules.", sh)["ok"]
    assert verify.check("It changed by 0.268 [F1].", sh)["ok"]
    bad = verify.check("It used 32 [F3] modules.", sh)                      # 32 is on the sheet, but F3 holds 8
    assert not bad["ok"] and bad["mismatched_citations"] == ["32 [F3]"]
    assert not verify.check("Only layer 2 [F1] matters.", sh)["ok"]           # a label cited to a fact


def test_proposal_allows_design_numbers_but_checks_citations(tmp_path):
    write_results(tmp_path)
    sh = facts.build(str(tmp_path))
    fid = next(f["id"] for f in sh.facts if f["value"] == 0.1371)
    prop = {k: "x" for k in agent.SCHEMA["required"]}
    prop.update(conditions=["dial 0", "dial 2"], risks=["r"], gpu_hours_estimate=1.0,
                pass_rule=f"Pass if the share is at least 0.6 (a clear majority); base tilt was 0.137 [{fid}].")
    out = tmp_path / "p1"; out.mkdir()
    agent.propose(FakeClient([json.dumps(prop)]), sh, str(out))
    assert json.load(open(next(out.glob("proposal-*.json"))))["_checks"]["numbers"]["ok"]
    out2 = tmp_path / "p2"; out2.mkdir()
    agent.propose(FakeClient([json.dumps(dict(prop, pass_rule=f"Pass if share reaches 0.5066 [{fid}]."))]), sh, str(out2))
    assert not json.load(open(next(out2.glob("proposal-*.json"))))["_checks"]["numbers"]["ok"]
