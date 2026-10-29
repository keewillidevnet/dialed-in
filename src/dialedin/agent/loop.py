"""The lab loop: Muse proposes and reviews, a human seals, the GPU runs, Muse explains, Muse proposes again.

    PYTHONPATH=src python3 -m dialedin.agent status        # every experiment's trail, from the files (no API call)
    PYTHONPATH=src python3 -m dialedin.agent next          # one turn: narrate, propose, critique, revise; then stop

The loop never writes or seals criteria and never starts a GPU run. Those are human gates: a person drafts the spec
from the revised proposal, has it reviewed (independently and by Muse), seals it with scripts/seal.sh, and starts the
run. Every model call is logged in results/agent/calls.jsonl; every artifact the loop writes is in results/agent/.
"""
import glob
import json
import os
import re
import time

import yaml

from . import verify

# What review caught in the first proposals, written down once so the next proposal does not repeat it.
LAB_CONSTRAINTS = """Lab constraints (fixed facts about the setup, not results):
- Model: Muse Glimmer-30B, 52 decoder layers. Global layers (full attention, no position encoding): 3, 7, 11, ..., 51.
  All others are local (2,048-token sliding window, RoPE). A control in another layer should match the layer type.
- The tilt is measured over attention heads in layers 2 to 51. A change made in layer L directly changes layer L's own
  heads and can only change later layers. Layer 2's own heads carry a large part of the dial's change (C7 under the
  full dial: layer 2's heads -74 percent, layers 3 to 51 -13 percent), so a control placed in a later layer cannot
  reach them.
- A measurement uses the first 32,768 tokens of a book (or another 32,768-token segment), with an 8,192-token burn-in.
  The shortest of the 15 books has 94,774 tokens, so a 32,768-token segment must start at or before token 62,006.
  First 32,768 tokens already measured with the adapter: set A (C5), set B (C7), set C (C8). Exploratory probes used
  tokens 60,000 to 68,191 (set C; C6 used sets B and C there) and the band-mute probe 70,000 to 78,191 (sets A and B).
- Cost: one 32,768-token pass takes about 13.5 s; per book and condition, 9 passes (undriven, sham, 7 periods).
- Shares: say what the share is a share of. A share of layer 2 attention's change is not a share of the full dial.
- The C5 adapter has 32 modules in layers 2 to 5 (q, k, v, gate and o projections and the MLP's three), rank 8."""

CRITIQUE = ("You are Muse, acting as the lab's design critic. Below is a proposed experiment, the lab constraints and the "
            "fact sheet. Check it before a human drafts it into a sealed spec: does it test something the facts leave "
            "open; does every quoted number match the fact it cites and mean what the proposal says it means (a share "
            "of what?); is the control matched (layer type, size, position relative to the read point); does the text "
            "fit (segment lengths); does the pass rule have a baseline, a numeric bar with a reason, and a control; is "
            "the budget arithmetic right? List concrete problems with fixes, most serious first, under 400 words, then "
            "a one-line verdict: SOUND, FIXABLE or UNSOUND. Plain sentences; never use em dashes.")

REVISE = ("You are Muse, the lab's experiment designer. A critic reviewed your proposal. Revise it to fix every valid "
          "point; if you reject a point, say why in the field revision_notes. Reply with one JSON object only, with the "
          "same keys as before plus revision_notes (a list of strings). Design numbers are your own; results you quote "
          "must carry the id of the fact that holds that exact number. Plain sentences; never use em dashes.")


def _stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ---------------------------------------------------------------- status
def _outcome(summary_path, key):
    d = json.load(open(summary_path))
    o = d.get(key, {})
    if isinstance(o, dict) and "outcome" in o:
        return o["outcome"]
    if isinstance(o, dict):                                   # C4: one outcome per lever
        return ", ".join(f"{k} {v.get('outcome')}" for k, v in o.items() if isinstance(v, dict) and "outcome" in v)
    return None


def status(repo: str = ".") -> list:
    """One row per sealed spec, oldest first: id, seal time and hash, who proposed it, Muse reviews that read it, runs
    started under it, outcome and summary file. Built only from files in the repo."""
    rows = []
    seals_path = os.path.join(repo, "criteria", "SEALS.txt")
    seals = [l.split() for l in open(seals_path) if l.strip()] if os.path.exists(seals_path) else []
    reviews = {}
    for p in sorted(glob.glob(os.path.join(repo, "results", "agent", "review-*.md"))):
        first = open(p).readline()
        for f in first.replace("Reviewed files:", "").split(","):
            reviews.setdefault(os.path.basename(f.strip()).replace(".yaml", ""), []).append(os.path.relpath(p, repo))
    for utc, sha, path in seals:
        spec_path = os.path.join(repo, path)
        spec = yaml.safe_load(open(spec_path)) if os.path.exists(spec_path) else {}
        sid = spec.get("id") or os.path.basename(path).replace(".yaml", "")
        m = re.match(r"c(\d+)-", os.path.basename(path))
        tag = f"c{m.group(1)}v{spec.get('version', 1)}" if m else None
        runs = []
        for s in sorted(glob.glob(os.path.join(repo, "results", "raw", f"{tag}-*", "STARTED.json"))):
            if json.load(open(s)).get("seal_sha256") == sha:
                runs.append(os.path.basename(os.path.dirname(s)))
        outcome, summary = None, None
        for r in runs:
            for p in glob.glob(os.path.join(repo, "results", "derived", r, "*_summary.json")):
                outcome, summary = _outcome(p, f"C{m.group(1)}"), os.path.relpath(p, repo)
        stopped = []
        for r in runs:
            for f in glob.glob(os.path.join(repo, "results", "raw", r, "phase*.json")):
                if os.path.basename(f).count(".") == 1:
                    d = json.load(open(f))
                    if d.get("stopped"):
                        stopped.append(str(d["stopped"]))
        if not outcome and stopped:
            outcome = "U (stopped by its own rule)"
        draft_id = os.path.basename(path).replace(".yaml", "")
        rows.append({"id": sid, "sealed_utc": utc, "seal_sha256": sha[:12],
                     "proposed_by": spec.get("proposed_by", "the lab (Keenan, with Claude as tooling)"),
                     "muse_reviews": reviews.get(draft_id, []), "runs": runs, "outcome": outcome,
                     "summary": summary, "stop_reasons": stopped,
                     "state": ("stopped by rule" if stopped and not summary else "analyzed" if outcome else
                               "run started" if runs else "sealed, not run")})
    return rows


def print_status(rows):
    for r in rows:
        who = r["proposed_by"].split(" (")[0]
        rv = f"; Muse review: {len(r['muse_reviews'])}" if r["muse_reviews"] else ""
        print(f"{r['id']:42s} sealed {r['sealed_utc'][:16]}  {r['state']:16s} outcome {str(r['outcome']):24s} "
              f"proposed by {who}{rv}")


# ---------------------------------------------------------------- one turn
def next_turn(narrator, designer, critic, sheet, out_dir, propose_fn, narrate_fn, parse_json, schema_keys):
    """narrate (narrator) -> propose (designer) -> critique (critic) -> revise (designer). Writes every artifact and a
    turn record; returns the turn record. Stops before any spec, seal or run."""
    turn = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {"narrator": narrator.model if hasattr(narrator, "model") else "muse",
                       "designer": designer.model if hasattr(designer, "model") else "muse",
                       "critic": critic.model if hasattr(critic, "model") else "muse"}}
    turn["narration"] = narrate_fn(narrator, sheet, out_dir)
    prop, prop_path = propose_fn(designer, sheet, out_dir, extra=LAB_CONSTRAINTS)
    turn["proposal"] = prop_path
    ctx = "Lab constraints:\n" + LAB_CONSTRAINTS + "\n\nFact sheet:\n" + sheet.text()
    clean = {k: v for k, v in prop.items() if not k.startswith("_")}
    crit = critic.chat([{"role": "system", "content": CRITIQUE},
                        {"role": "user", "content": ctx + "\n\nProposal:\n" + json.dumps(clean, indent=1)}],
                       temperature=0.2, max_tokens=24000)
    crit_path = os.path.join(out_dir, f"critique-{_stamp()}-{turn['models']['critic']}.md")
    with open(crit_path, "w") as fh:
        fh.write(f"Critique of {os.path.basename(prop_path)}\n\n{crit.strip()}\n")
    turn["critique"] = crit_path
    raw = designer.chat([{"role": "system", "content": REVISE},
                         {"role": "user", "content": ctx + "\n\nYour proposal:\n" + json.dumps(clean, indent=1)
                          + "\n\nCritique:\n" + crit}], temperature=0.3, max_tokens=24000)
    rev = parse_json(raw)
    report = verify.check(json.dumps(rev), sheet)
    rev["_checks"] = {"missing_fields": [k for k in schema_keys if k not in rev],
                      "numbers_ok": not report["mismatched_citations"] and not report["unknown_citations"],
                      "mismatched_citations": report["mismatched_citations"], "unknown_citations": report["unknown_citations"]}
    rev["_revises"] = os.path.basename(prop_path)
    rev["_model"] = turn["models"]["designer"]
    rev_path = os.path.join(out_dir, f"proposal-{_stamp()}-{turn['models']['designer']}-revised.json")
    json.dump(rev, open(rev_path, "w"), indent=2)
    turn["revised_proposal"] = rev_path
    turn["human_gate"] = ("Draft a spec from the revised proposal, have it reviewed independently and by Muse "
                          "(agent review --files ...), seal it with scripts/seal.sh, then start the run on the pod.")
    turn_path = os.path.join(out_dir, f"turn-{_stamp()}.json")
    json.dump(turn, open(turn_path, "w"), indent=2)
    print(f"\n== critique ({turn['models']['critic']}):\n{crit.strip()}\n")
    print(json.dumps({k: v for k, v in rev.items()}, indent=2))
    print(f"\n== turn record: {turn_path}\n== HUMAN GATE: {turn['human_gate']}")
    return turn
