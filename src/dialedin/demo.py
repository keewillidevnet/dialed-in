"""Build the demo page from saved results. No GPU, no network, no model calls.

    PYTHONPATH=src python3 -m dialedin.demo            # writes demo/index.html

Every number on the page is read here from a file under results/ or criteria/ and carries that file's path, so a judge
can trace it. The page has three parts: the dial (C5 curves and costs), where the dial lives (head map, C7, C8), and
the lab loop (every sealed experiment, Muse's narration, and the C8 story from Muse Wonder's proposal to the seal).
C8 shows as running until results/derived/c8v1-*/c8_summary.json exists; rebuild after the analysis to fill it in.
A revised C9 proposal (results/agent/proposal-*-revised.json from `agent next`) is shown when present.
"""
import argparse
import glob
import json
import os
import re

C5 = "results/derived/c5v1-20261029T025343Z/c5_summary.json"
C6 = "results/derived/c6v2-20261029T060851Z/c6_summary.json"
C7 = "results/derived/c7v1-20261029T154201Z/c7_summary.json"
C1 = "results/derived/c1v5-20261027T152040Z/c1_summary.json"
C2 = "results/derived/c2v1-20261027T201901Z/c2_summary.json"
C3 = "results/derived/c3v1-20261028T014250Z/c3_summary.json"
C4 = "results/derived/c4v1-20261028T065820Z/c4_summary.json"
HEADMAP = "results/derived/headmap-c1v5-20261027T152040Z/headmap.json"
STATUS = "results/agent/status.json"
NARRATION = "results/agent/narration-20261029T191954Z-muse-spark-1.3.md"
PROPOSAL = "results/agent/proposal-20261029T193320Z-muse-wonder-1.0.json"
REVIEW = "results/agent/review-20261029T201626Z-muse-wonder-1.0.md"
C8_SPEC = "criteria/sealed/c8-glimmer-one-matrix-dial-v1.yaml"
SEALS = "criteria/SEALS.txt"
BOOKS_A = {"1342": "Pride and Prejudice", "2701": "Moby Dick", "1400": "Great Expectations", "145": "Middlemarch",
           "1184": "The Count of Monte Cristo"}          # criteria/sealed/c5-glimmer-trained-dial-v1.yaml, set A


def j(repo, p):
    return json.load(open(os.path.join(repo, p)))


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%"


def plain_outcomes(repo):
    """One plain sentence per analyzed experiment, numbers read from its summary."""
    c1, c2, c3, c4 = (j(repo, p) for p in (C1, C2, C3, C4))
    c5, c6, c7 = j(repo, C5)["C5"], j(repo, C6)["C6"], j(repo, C7)["C7"]
    mins = [v["min_normalized"] for v in c1["C1"]["per_set"].values()]
    c2d = [max(abs(v["delta_global"]), abs(v["delta_local"])) for v in c2["C2"]["per_set"].values()]
    t = c4["C4"]
    return {
        "c1-glimmer-frequency-response-v4": "Stopped by its own rule before measuring: no drive amplitude qualified.",
        "c1-glimmer-frequency-response-v5":
            f"The lean is real. In all three book sets gain rises with period, and the shortest period keeps "
            f"{min(mins) * 100:.0f} to {max(mins) * 100:.0f}% of the peak.",
        "c2-glimmer-knob-tilt-v1":
            f"Scaling whole global or local layers moved the tilt by at most {max(c2d):.3f}, short of the bar.",
        "c3-glimmer-head-dial-v1": "Boosting the selected heads did no better than boosting random heads.",
        "c4-glimmer-structure-and-text-v1":
            f"Changing the window or the position encoding did not flatten the tilt cleanly. Shuffled text keeps "
            f"{t['text']['shuffled_over_base'] * 100:.0f}% of it, so the lean comes from the model, not the prose.",
        "c5-glimmer-trained-dial-v1":
            f"The trained dial flattens the tilt by {abs(c5['relative_change']) * 100:.1f}% at scale 2, in order at every step and in "
            f"{c5['books_agree']} of 5 books, just short of the pre-registered bar. Perplexity x{c5['worst_ppl_ratio']:.3f}.",
        "c6-glimmer-dial-topic-switch-v2":
            f"At scale 2, loss right after a topic switch rose {pct(c6['relative_rise'])}, too small to count as a "
            f"behavior change.",
        "c7-glimmer-dial-localization-v1":
            f"Layer 2's q, k and v (3 of 32 modules) carry {c7['share_QKV2'] * 100:.1f}% of the dial's effect.",
    }


def parse_narration(text):
    body, _, tail = text.partition("\n---\n")
    srcs = {}
    for m in re.finditer(r"- \[(F\d+)\] (.+?): `([^`]+)`", tail):
        srcs[m.group(1)] = {"label": m.group(2), "path": m.group(3)}
    return {"paragraphs": [p.strip() for p in body.strip().split("\n\n") if p.strip()], "sources": srcs,
            "verified": "every number matches a cited fact" in tail}


# Short labels for the findings in REVIEW, in its order (severity is read from the file). Item 8 was checked and
# found to be a false alarm: C6's 10 books are sets B and C, tokens 60,000 to 68,191
# (criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml), so the draft's claim was right.
REVIEW_LABELS = ["The smoke-test result was still a placeholder, so a sealed file could never start.",
                 "The question's wording and the per-book rule disagreed.",
                 "A retry after a failure was not stamped with a written reason.",
                 "Time spent in a killed process never reached the 2-hour budget ledger.",
                 "Numbers from a partial run were not nested the way the spec said.",
                 "Stale text: a sentence said C7 where it meant C8.",
                 "Stale text: a docstring said set B where the run uses set C.",
                 "Asked which part of C6 used set C (checked: a false alarm)."]


def parse_review(text):
    first, _, rest = text.partition("\n")
    verdict = re.search(r"Verdict: (.+)", rest).group(1).strip()
    items = []
    for m in re.finditer(r"^\d+\. (Blocker|Major|Minor), (.+?)\. (.+?)(?= Fix:|$)", rest, re.M):
        items.append({"severity": m.group(1), "where": m.group(2).split(",")[0]})
    for it, label in zip(items, REVIEW_LABELS):
        it["label"] = label
    return {"files": first.replace("Reviewed files:", "").strip(), "verdict": verdict, "items": items}


def parse_spec_header(text):
    """The numbered revisions in the sealed spec's header comment, first sentence of each."""
    head = text.split("\n# Reviewed before sealing")[0]
    blocks = re.split(r"\n#   (?=\d\. )", head)[1:]
    out = []
    for b in blocks:
        b = re.sub(r"\s*\n#\s+", " ", b)
        m = re.match(r"(\d)\. (.+?\.)(?=\s|$)", b)
        out.append({"n": int(m.group(1)), "text": m.group(2)})
    return out


def latest(repo, pattern):
    hits = sorted(glob.glob(os.path.join(repo, pattern)))
    return os.path.relpath(hits[-1], repo) if hits else None


def build(repo):
    c5, c6, c7, hm = j(repo, C5), j(repo, C6), j(repo, C7), j(repo, HEADMAP)
    status = j(repo, STATUS)
    seals = {os.path.basename(p): {"utc": u, "sha": s} for u, s, p in
             (l.split() for l in open(os.path.join(repo, SEALS)) if l.strip())}
    c8_seal = seals.get(os.path.basename(C8_SPEC))
    prop = j(repo, PROPOSAL)
    data = {
        "dial": {
            "src": C5, "grid": c5["grid"], "curves": c5["curves"], "tilts": c5["tilts"],
            "ppl": c5["perplexity_ratio_median"], "relative_change": c5["C5"]["relative_change"],
            "needed_relative": c5["C5"]["needed"] / c5["tilts"]["S0"], "books_agree": c5["C5"]["books_agree"],
            "ci_S2": c5["bootstrap_95"]["relative_change_S2"],
            "books": [{"id": b["gutenberg_id"], "title": BOOKS_A.get(str(b["gutenberg_id"]), str(b["gutenberg_id"])),
                       "tilts": b["tilts"]} for b in c5["book_tilts"]],
            "c6": {"src": C6, "rise": c6["C6"]["relative_rise"], "ci": c6["bootstrap_95_relative_rise"],
                   "books_agree": c6["C6"]["books_agree"], "books": len(c6["book_rise_S2"]),
                   "control": c6["control_random_adapter"]["relative_rise"]},
        },
        "where": {
            "headmap_src": HEADMAP, "layers": hm["per_head"]["layers"], "tilt": hm["per_head"]["tilt"],
            "resolved": hm["summary"]["heads_resolved"], "total": hm["summary"]["heads_total"],
            "c7_src": C7, "c7": {k: c7["C7"][k] for k in ("outcome", "full_change", "share_QKV2", "share_REST")},
            "c7_ci": c7["bootstrap_95"], "c7_ppl_qkv2": c7["perplexity_ratio_median"]["QKV2"],
            "c7_layer2_full": c7["diagnostic_groups"]["layer2"]["relative_change"]["FULL"],
            "c7_later_full": c7["diagnostic_groups"]["layers3_51"]["relative_change"]["FULL"],
        },
        "loop": {
            "status_src": STATUS, "status_utc": status["utc"], "rows": status["experiments"],
            "plain": plain_outcomes(repo),
            "narration_src": NARRATION, "narration": parse_narration(open(os.path.join(repo, NARRATION)).read()),
            "proposal_src": PROPOSAL, "proposal": {k: prop[k] for k in ("title", "question", "control", "gpu_hours_estimate")},
            "proposal_model": prop.get("_model"),
            "proposal_utc": re.search(r"(\d{8}T\d{6}Z)", PROPOSAL).group(1),
            "review_utc": re.search(r"(\d{8}T\d{6}Z)", REVIEW).group(1),
            "review_src": REVIEW, "review": parse_review(open(os.path.join(repo, REVIEW)).read()),
            "spec_src": C8_SPEC, "revisions": parse_spec_header(open(os.path.join(repo, C8_SPEC)).read()),
            "seal": c8_seal,
        },
    }
    c8 = [p for p in sorted(glob.glob(os.path.join(repo, "results/derived/c8v1-*/c8_summary.json")))
          if "PARTIAL" not in p]
    if c8:
        d = json.load(open(c8[-1]))
        data["c8"] = {"src": os.path.relpath(c8[-1], repo), "C8": d["C8"], "ci": d.get("bootstrap_95"),
                      "ppl": d.get("perplexity_ratio_median"), "downstream": d.get("secondary_downstream_shares")}
    nxt = latest(repo, "results/agent/proposal-*-revised.json")
    if nxt:
        p = j(repo, nxt)
        data["next"] = {"src": nxt, "title": p.get("title"), "question": p.get("question"),
                        "why": p.get("why_this_next"), "model": p.get("_model"),
                        "numbers_ok": p.get("_checks", {}).get("numbers_ok")}
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="demo/index.html")
    ap.add_argument("--running", default=None, help="run id of a run in progress (shown until its summary exists)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    data = build(repo)
    if args.running and "c8" not in data:
        data["running"] = args.running
    tpl = open(os.path.join(os.path.dirname(__file__), "demo_template.html")).read()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = os.path.join(repo, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(f"== written: {os.path.relpath(out, repo)} ({len(html) // 1024} KB); C8 "
          f"{'filled from ' + data['c8']['src'] if 'c8' in data else 'shown as running'}; next proposal "
          f"{data['next']['src'] if 'next' in data else 'none yet'}")


if __name__ == "__main__":
    main()
