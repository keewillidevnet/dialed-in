"""Build the demo page from saved results. No GPU, no network, no model calls.

    PYTHONPATH=src python3 -m dialedin.demo            # writes demo/index.html

Every number on the page is read here from a file under results/ or criteria/ and carries that file's path, so a judge
can trace it. The page has three parts: the dial (C5 curves and costs, and the exploratory behavior probe), where the
dial acts (head map, C7, C8, C9), and the lab loop (every sealed experiment from the live status of the seals and
runs, Muse Spark's latest narration, the C9 story from Muse Wonder's proposal to the result, and the newest revised
proposal from `agent next`).
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
C9 = "results/derived/c9v1-20261030T023935Z/c9_summary.json"
BEHAVIOR = "results/raw/probe-behavior-20261030T022546Z/behavior.json"
TURN = "results/agent/turn-20261030T005705Z.json"              # the lab-loop turn that proposed C9
REVIEW = "results/agent/review-20261030T014157Z-muse-wonder-1.0.md"
C9_SPEC = "criteria/sealed/c9-glimmer-necessity-v1.yaml"
C6_SPEC = "criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml"
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
    c8 = j(repo, latest(repo, "results/derived/c8v1-*/c8_summary.json"))["C8"]
    c9 = j(repo, C9)["C9"]
    mins = [v["min_normalized"] for v in c1["C1"]["per_set"].values()]
    c2d = [max(abs(v["delta_global"]), abs(v["delta_local"])) for v in c2["C2"]["per_set"].values()]
    t = c4["C4"]
    return {
        "c1-glimmer-frequency-response-v4": "Stopped by its own rule before measuring: no drive amplitude qualified.",
        "c1-glimmer-frequency-response-v5":
            f"The lean is real for the probe signal. In all three book sets gain rises with period, and the shortest period keeps "
            f"{min(mins) * 100:.0f} to {max(mins) * 100:.0f}% of the peak.",
        "c2-glimmer-knob-tilt-v1":
            f"Scaling whole global or local layers moved the tilt by at most {max(c2d):.3f}, short of the bar.",
        "c3-glimmer-head-dial-v1": "Boosting the selected heads did no better than boosting random heads.",
        "c4-glimmer-structure-and-text-v1":
            f"Changing the window or the position encoding did not flatten the tilt cleanly. Shuffled text keeps "
            f"{t['text']['shuffled_over_base'] * 100:.0f}% of it, so the lean belongs to the model more than the prose.",
        "c5-glimmer-trained-dial-v1":
            f"The trained dial flattens the tilt by {abs(c5['relative_change']) * 100:.1f}% at scale 2, in order at every step and in "
            f"{c5['books_agree']} of 5 books, just short of the pre-registered bar. Perplexity x{c5['worst_ppl_ratio']:.3f}.",
        "c6-glimmer-dial-topic-switch-v2":
            f"At scale 2, loss right after a topic switch rose {pct(c6['relative_rise'])}, too small to count as a "
            f"behavior change.",
        "c7-glimmer-dial-localization-v1":
            f"Layer 2's q, k and v (3 of 32 modules) carry {c7['share_QKV2'] * 100:.1f}% of the dial's effect.",
        "c8-glimmer-one-matrix-dial-v1":
            f"One matrix, layer 2's v, gives {c8['share_V2'] * 100:.1f}% of the dial's effect alone; the same matrix in "
            f"layer 4 gives {c8['share_V4'] * 100:.1f}%.",
        "c9-glimmer-necessity-v1":
            f"Layer 2's v is needed for about half: without it the full dial keeps only {c9['retained_NOV2'] * 100:.1f}% of its effect; without layer "
            f"2's gate it keeps {c9['retained_NOG2'] * 100:.1f}%. The downstream control passed narrowly "
            f"({c9['downstream']['retained_NOV4']:.3f} against a bar of 0.80).",
    }


def parse_narration(text):
    body, _, tail = text.partition("\n---\n")
    srcs = {}
    for m in re.finditer(r"- \[(F\d+)\] (.+?): `([^`]+)`", tail):
        srcs[m.group(1)] = {"label": m.group(2), "path": m.group(3)}
    return {"paragraphs": [p.strip() for p in body.strip().split("\n\n") if p.strip()], "sources": srcs,
            "verified": "every number matches a cited fact" in tail}


# Short labels for the findings in REVIEW (Muse Wonder's review of the C9 draft), in its order; severity is read from
# the file. Item 1 was expected at review time: the smoke result is filled in after the pod smoke run, before sealing.
REVIEW_LABELS = ["The smoke-test result was still a placeholder (expected: filled in after the pod smoke run).",
                 "The partial-run folder was named differently from the spec.",
                 "A partial look at a run with no book files would have crashed.",
                 "The question's gates did not match the outcome rules.",
                 "The book rule left out that the full dial must flatten the book.",
                 "A stopped setup record did not list earlier failures."]


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


def parse_critique(text):
    """Spark's critique: the verdict and the first sentence of each numbered point."""
    verdict = re.search(r"Verdict: (\w+)", text)
    points = [m.group(1).split(". ")[0].rstrip(".") + "." for m in re.finditer(r"^\d+\. (.+)$", text, re.M)]
    problems = [p for p in points if not p.startswith("Quoted numbers match")]     # a confirmation, not a problem
    return {"verdict": verdict.group(1) if verdict else None, "points": problems,
            "confirmed_numbers": len(problems) < len(points)}


def latest(repo, pattern):
    hits = sorted(glob.glob(os.path.join(repo, pattern)))
    return os.path.relpath(hits[-1], repo) if hits else None


def build(repo):
    c5, c6, c7, hm = j(repo, C5), j(repo, C6), j(repo, C7), j(repo, HEADMAP)
    from .agent import loop
    import yaml
    rows = loop.status(repo)                               # live: seals, sealed specs, run records, summaries
    seals = {os.path.basename(p): {"utc": u, "sha": s} for u, s, p in
             (l.split() for l in open(os.path.join(repo, SEALS)) if l.strip())}
    turn = j(repo, TURN)
    rel = lambda p: os.path.relpath(os.path.join(repo, p), repo)
    prop, revp = j(repo, rel(turn["proposal"])), j(repo, rel(turn["revised_proposal"]))
    crit = parse_critique(open(os.path.join(repo, rel(turn["critique"]))).read())
    c9 = j(repo, C9)
    beh = j(repo, BEHAVIOR)
    c6spec = yaml.safe_load(open(os.path.join(repo, C6_SPEC)))
    titles = {int(g): t for st in c6spec["inputs"]["prompt_sets"].values() for g, t in st.items()}
    narr = latest(repo, "results/agent/narration-*.md")
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
            "behavior": behavior_panel(beh, titles),
        },
        "where": {
            "headmap_src": HEADMAP, "layers": hm["per_head"]["layers"], "tilt": hm["per_head"]["tilt"],
            "resolved": hm["summary"]["heads_resolved"], "total": hm["summary"]["heads_total"],
            "c7_src": C7, "c7": {k: c7["C7"][k] for k in ("outcome", "full_change", "share_QKV2", "share_REST")},
            "c7_ci": c7["bootstrap_95"], "c7_ppl_qkv2": c7["perplexity_ratio_median"]["QKV2"],
            "c7_layer2_full": c7["diagnostic_groups"]["layer2"]["relative_change"]["FULL"],
            "c7_later_full": c7["diagnostic_groups"]["layers3_51"]["relative_change"]["FULL"],
            "c9_src": C9, "c9": {k: v for k, v in c9["C9"].items() if k not in ("tilts", "tilts_downstream")},
            "c9_ci": c9["bootstrap_95"], "c9_ppl": c9["perplexity_ratio_median"],
        },
        "loop": {
            "rows": rows, "plain": plain_outcomes(repo),
            "narration_src": narr, "narration": parse_narration(open(os.path.join(repo, narr)).read()),
            "turn_src": TURN, "turn_utc": turn["started_utc"], "models": turn["models"],
            "proposal_src": rel(turn["proposal"]), "proposal": {k: prop[k] for k in ("title", "question")},
            "critique_src": rel(turn["critique"]), "critique": crit,
            "revised_src": rel(turn["revised_proposal"]), "revision_notes": revp.get("revision_notes", []),
            "review_utc": re.search(r"(\d{8}T\d{6}Z)", REVIEW).group(1),
            "review_src": REVIEW, "review": parse_review(open(os.path.join(repo, REVIEW)).read()),
            "spec_src": C9_SPEC, "revisions": parse_spec_header(open(os.path.join(repo, C9_SPEC)).read()),
            "seal": seals.get(os.path.basename(C9_SPEC)),
            "c9_src": C9, "c9_outcome": c9["C9"]["outcome"], "c9_retained_NOV2": c9["C9"]["retained_NOV2"],
            "c9_ci": c9["bootstrap_95"]["retained_NOV2"], "c9_retained_NOG2": c9["C9"]["retained_NOG2"],
        },
    }
    c8 = [p for p in sorted(glob.glob(os.path.join(repo, "results/derived/c8v1-*/c8_summary.json")))
          if "PARTIAL" not in p]
    if c8:
        d = json.load(open(c8[-1]))
        data["c8"] = {"src": os.path.relpath(c8[-1], repo), "C8": d["C8"], "ci": d.get("bootstrap_95"),
                      "ppl": d.get("perplexity_ratio_median"), "downstream": d.get("secondary_downstream_shares")}
    nxt = latest(repo, "results/agent/proposal-*-revised.json")
    if nxt and nxt != rel(turn["revised_proposal"]):           # a newer turn than C9's
        p = j(repo, nxt)
        data["next"] = {"src": nxt, "title": p.get("title"), "question": p.get("question"),
                        "why": p.get("why_this_next"), "model": p.get("_model"),
                        "numbers_ok": p.get("_checks", {}).get("numbers_ok"),
                        "revisions": len(p.get("revision_notes", []))}
        c8p = latest(repo, "results/derived/c8v1-*/c8_summary.json")
        if c8p and "q, k, v" in (p.get("title", "") + p.get("question", "")):
            g = j(repo, c8p)["diagnostic_groups"]["layers5_51"]["shares"]["QKV2"]
            data["next"]["lab_note"] = {"src": c8p + "#diagnostic_groups.layers5_51.shares.QKV2", "value": g}
    return data


def behavior_panel(beh, titles) -> dict:
    """The exploratory behavior probe: the rise in loss on the new book after a switch per condition (relative to the
    dial at 0), and the showcase pair at scale 4 with its three continuations, all read from the probe's file."""
    sm = beh["summary"]
    base = sm["S0"]["mean_excess_loss"]
    out = {"src": BEHAVIOR, "pairs": len(beh["pairs"]),
           "rise": {c: sm[c]["mean_excess_loss"] / base - 1 for c in ("S2", "S4", "R2", "R4")},
           "ppl": {c: sm[c]["median_perplexity_ratio"] for c in ("S2", "S4", "R2", "R4")},
           "old_words": sum(v["old_words_in_text"] for v in sm.values())}
    per = {c: sorted(p["conditions"][c]["excess_loss"] / p["conditions"]["S0"]["excess_loss"] - 1
                     for p in beh["pairs"].values()) for c in ("S2", "S4", "R2", "R4")}
    out["median"] = {c: (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2 if len(v) % 2 == 0 else v[len(v) // 2] for c, v in per.items()}
    out["up"] = {c: sum(x > 0 for x in v) for c, v in per.items()}
    k = beh["showcase"].get("S4")
    if k:
        pr = beh["pairs"][k]
        out["show"] = {"x": titles.get(pr["x"], str(pr["x"])), "y": titles.get(pr["y"], str(pr["y"])),
                       "texts": {c: pr["conditions"][c]["text"] for c in ("S0", "S4", "R4")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="demo/index.html")
    ap.add_argument("--body-only", default=None, help="also write the page body alone to this path")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    data = build(repo)
    tpl = open(os.path.join(os.path.dirname(__file__), "demo_template.html")).read()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
            "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
            + html + "\n</body>\n</html>\n")
    out = os.path.join(repo, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)                 # a complete page that opens straight from disk
    pages = os.path.join(repo, "docs", "index.html")              # the same page for GitHub Pages (main branch, /docs)
    os.makedirs(os.path.dirname(pages), exist_ok=True)
    open(pages, "w", encoding="utf-8").write(page)
    if args.body_only:                                            # the page body alone, for hosts that add their own <head>
        open(args.body_only, "w", encoding="utf-8").write(html)
    print(f"== written: {os.path.relpath(out, repo)} ({len(html) // 1024} KB); narration {data['loop']['narration_src']}; "
          f"next proposal {data['next']['src'] if 'next' in data else 'none yet'}")


if __name__ == "__main__":
    main()
