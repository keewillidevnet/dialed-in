"""Muse Spark lab agent.

    export META_API_KEY   (set it with: read -rs META_API_KEY; export META_API_KEY)
    PYTHONPATH=src python3 -m dialedin.agent facts      # print the fact sheet (no API call)
    PYTHONPATH=src python3 -m dialedin.agent narrate    # Muse explains the results; every number checked
    PYTHONPATH=src python3 -m dialedin.agent propose    # Muse drafts the next pre-registered experiment
    PYTHONPATH=src python3 -m dialedin.agent review --files <spec> <code> ...   # Muse reviews a draft before sealing
    PYTHONPATH=src python3 -m dialedin.agent status     # every experiment's trail, from the files (no API call)
    PYTHONPATH=src python3 -m dialedin.agent next       # one loop turn: Spark narrates, Wonder proposes, Spark
                                                        # critiques, Wonder revises; stops at the human gate
    add --model muse-wonder-1.0 to use Muse Wonder instead of Muse Spark (default muse-spark-1.3)

Outputs go to results/agent/. Muse never seals criteria, edits sealed files, or runs the GPU: a
proposal is a draft for Keenan, who reviews it, has it independently reviewed, and seals it himself.
"""
import argparse
import json
import os
import time

from . import facts, verify

STYLE = ("Write in plain, short sentences. Never use em dashes. Never use the construction 'not X, it's Y'. "
         "Use only numbers that appear in the fact sheet, rounded as written there or to fewer digits, or a fraction "
         "on the sheet written as a percentage (0.268 as 27 percent). Put the fact id in square brackets right after "
         "each number, and only the id of the fact that holds that exact number, like 0.137 [F12]. Never cite a fact "
         "for a word or a label. Do not compute new numbers. If a claim needs a number that is not on the sheet, say "
         "it qualitatively.")

NARRATE = ("You are Muse, the AI scientist of the Dialed In lab. The lab measures the frequency response of the "
           "Muse Glimmer-30B language model: it adds a small sine wave of period N tokens to the model's internal "
           "signal after layer 1 and measures how strongly each later attention head responds, for N from 2 to "
           "4,096 tokens. Every sealed experiment (C1 onward) was pre-registered and sealed by hash before it ran; facts "
           "labeled EXPLORATORY come from unsealed probes and must be called exploratory. Write the lab's findings "
           "for a technical judge, in 250 to 350 words, as a story with five short paragraphs: (1) what the lab "
           "measures and what it found, in plain words (the model favors slow, long-range patterns); (2) that no "
           "built-in setting could turn this down (C2 to C4); (3) the trained dial and what it costs and does (C5, "
           "C6), including the near miss and the small behavior effect; (4) where the dial acts (C7, and C8 if it is on "
           "the sheet: it was proposed by Muse Wonder); (5) what the "
           "exploratory head map suggests and what is still open. Lead each paragraph with its point and say what it "
           "means. Use at most 15 numbers in total, only the ones that carry the point; prefer percentages. Give each "
           "sealed outcome by its code. Be candid about null and failed results. " + STYLE)

PROPOSE = ("You are Muse, the AI scientist of the Dialed In lab. Using the results on the fact sheet, propose the "
           "single most informative next experiment the lab can run in under 2 GPU hours on one 96 GB GPU (BF16) "
           "with the existing harness. Available tools: the tilt measurement at 32,768 tokens (about 13 s per pass, "
           "7 periods, shams, 5 books per set); the frozen C5 adapter at any scale with any subset of its 32 modules "
           "switched on (layers 2 to 5: q, k, v, gate and o projections and MLP); muting chosen attention heads; "
           "per-token next-token loss on held-out book text; the drive direction, amplitude and injection layer; "
           "the local window size and RoPE base; shuffled or repeated-token text; and, as a separately labeled side "
           "check only, the same model loaded in 4-bit NF4. Book sets A, B and C hold 5 books each; text beyond "
           "token 70,000 has not been used. No new training. The proposal must test something the facts leave "
           "open, fix its pass rule before any data with a numeric threshold, and include a control. The conditions "
           "must include the unmodified model (dial at 0) as the baseline, and use dial scale 2, the sealed scale. The "
           "control must be an intervention of the same kind and size that should NOT produce the effect if the "
           "hypothesis is right (for example, the same modules in a different layer). Design numbers (thresholds, "
           "scales, counts, hours) are your own choices: write them plainly, without a citation, and justify each "
           "threshold in one clause. Cite a fact only when you quote a result. "
           "Reply with one JSON object only, no other text, with exactly these keys: " + ", ".join(
               ["title", "question", "prediction", "intervention", "conditions", "control", "books", "primary_metric",
                "pass_rule", "failure_meaning", "gpu_hours_estimate", "risks", "why_this_next"]) +
           ". conditions and risks are lists of strings; gpu_hours_estimate is a number; the rest are strings. "
           "Write in plain, short sentences. Never use em dashes. Results you quote must carry the id of the fact that "
           "holds that exact number, like 0.797 [F113]; never cite a fact for a design number or a label.")

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["title", "question", "prediction", "intervention", "conditions", "control", "books",
                       "primary_metric", "pass_rule", "failure_meaning", "gpu_hours_estimate", "risks", "why_this_next"],
          "properties": {k: {"type": "string"} for k in
                         ["title", "question", "prediction", "intervention", "control", "books", "primary_metric",
                          "pass_rule", "failure_meaning", "why_this_next"]} |
                        {"conditions": {"type": "array", "items": {"type": "string"}},
                         "gpu_hours_estimate": {"type": "number"},
                         "risks": {"type": "array", "items": {"type": "string"}}}}


REVIEW = ("You are Muse, the AI scientist of the Dialed In lab, acting as an adversarial pre-registration reviewer. "
          "Below are a draft experiment (a criteria file that will be sealed by hash before the run) and its code, plus "
          "the lab's fact sheet of earlier results. Find problems BEFORE the experiment is sealed and run on a GPU: "
          "anything that could crash after expensive GPU work, make the outcome rule mislead, let a result be "
          "rescued after the fact, contradict between the spec and the code, or misstate an earlier result. Check the "
          "arithmetic in the budget and in any implied numbers. Do not edit anything. Reply with a verdict (READY TO "
          "SEAL, READY WITH MINOR FIXES, or NOT READY), then a numbered list of findings, each with severity (blocker, "
          "major, minor), the file and the line or field, what is wrong, and a concrete fix. Report only what you can "
          "point to in the text. Under 900 words. Plain sentences; never use em dashes.")


def stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def parse_json(raw: str) -> dict:
    """The reply should be one JSON object; tolerate a code fence or text around it."""
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < a:
        raise SystemExit(f"Muse did not return JSON: {raw[:300]!r}")
    return json.loads(raw[a:b + 1])


def narrate(client, sheet, out_dir):
    messages = [{"role": "system", "content": NARRATE},
                {"role": "user", "content": "Fact sheet:\n" + sheet.text()}]
    text = client.chat(messages, max_tokens=12000)
    report = verify.check(text, sheet)
    if not report["ok"]:
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Verification failed. Numbers not on the fact sheet: "
                      f"{report['unverified_numbers']}; numbers that do not match the fact they cite: "
                      f"{report['mismatched_citations']}; unknown citations: {report['unknown_citations']}. "
                      "Rewrite the explanation using only fact-sheet numbers, each followed by the id of the fact "
                      "that holds it, and no citation after words or labels."}]
        text = client.chat(messages, max_tokens=12000)
        report = verify.check(text, sheet)
    path = os.path.join(out_dir, f"narration-{stamp()}-{getattr(client, 'model', 'muse')}.md")
    sources = "\n".join(f"- [{f['id']}] {f['label']}: `{f['source']}`" for f in sheet.facts if f["id"] in report["citations_used"])
    with open(path, "w") as fh:
        fh.write(text.strip() + "\n\n---\nVerification: " + ("every number matches a cited fact." if report["ok"] else
                 f"UNVERIFIED numbers {report['unverified_numbers']}, mismatched citations {report['mismatched_citations']}, "
                 f"unknown citations {report['unknown_citations']}.")
                 + "\n\nSources:\n" + sources + "\n")
    print(text.strip()); print("\n== verification:", "PASSED" if report["ok"] else report); print("== wrote", path)
    return path


def propose(client, sheet, out_dir, extra=None):
    messages = [{"role": "system", "content": PROPOSE},
                {"role": "user", "content": ((extra + "\n\n") if extra else "") + "Fact sheet:\n" + sheet.text()}]
    raw = client.chat(messages, temperature=0.4, max_tokens=16000)
    prop = parse_json(raw)
    missing = [k for k in SCHEMA["required"] if k not in prop]
    report = verify.check(json.dumps(prop), sheet)
    report["ok"] = not report["mismatched_citations"] and not report["unknown_citations"]   # design numbers are free
    prop["_checks"] = {"missing_fields": missing, "numbers": report,
                       "within_budget": isinstance(prop.get("gpu_hours_estimate"), (int, float)) and prop["gpu_hours_estimate"] <= 2}
    prop["_model"] = getattr(client, "model", "unknown")
    path = os.path.join(out_dir, f"proposal-{stamp()}-{getattr(client, 'model', 'muse')}.json")
    json.dump(prop, open(path, "w"), indent=2)
    print(json.dumps(prop, indent=2)); print("== wrote", path)
    print("== this is a draft for review: it becomes an experiment only after independent review and a human seal")
    return prop, path


def review(client, sheet, out_dir, files, repo, max_tokens=64000):
    parts = []
    for f in files:
        text = open(os.path.join(repo, f)).read()
        numbered = "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(text.splitlines()))
        parts.append(f"===== FILE {f} =====\n{numbered}")
    messages = [{"role": "system", "content": REVIEW},
                {"role": "user", "content": "Fact sheet:\n" + sheet.text() + "\n\n" + "\n\n".join(parts)}]
    text = client.chat(messages, temperature=0.2, max_tokens=max_tokens)
    path = os.path.join(out_dir, f"review-{stamp()}-{getattr(client, 'model', 'muse')}.md")
    with open(path, "w") as fh:
        fh.write(f"Reviewed files: {', '.join(files)}\n\n" + text.strip() + "\n")
    print(text.strip()); print("== wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["facts", "narrate", "propose", "review", "status", "next"])
    ap.add_argument("--narrator", default="muse-spark-1.3", help="next: model that explains the results")
    ap.add_argument("--designer", default="muse-wonder-1.0", help="next: model that proposes and revises")
    ap.add_argument("--critic", default="muse-spark-1.3", help="next: model that critiques the proposal")
    ap.add_argument("--files", nargs="+", default=[], help="for review: the draft spec and code files, repo-relative")
    ap.add_argument("--max-tokens", type=int, default=64000, help="for review: reply limit, reasoning included")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default=None, help="muse-spark-1.3 (default) or muse-wonder-1.0")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one reply (reasoning can be slow)")
    args = ap.parse_args()
    if args.task == "status":
        from . import loop
        rows = loop.status(args.repo)
        loop.print_status(rows)
        out = os.path.join(args.repo, "results", "agent", "status.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"utc": stamp(), "experiments": rows}, open(out, "w"), indent=1)
        print("== wrote", out); return
    sheet = facts.build(args.repo)
    if not sheet.facts:
        raise SystemExit("no results found under results/derived/")
    if args.task == "facts":
        print(sheet.text()); return
    from .muse import MuseClient
    out_dir = os.path.join(args.repo, "results", "agent")
    os.makedirs(out_dir, exist_ok=True)
    client = MuseClient(log_path=os.path.join(out_dir, "calls.jsonl"), timeout=args.timeout,
                        **({"model": args.model} if args.model else {}))
    if args.task == "next":
        from . import loop
        mk = lambda m: MuseClient(log_path=os.path.join(out_dir, "calls.jsonl"), timeout=args.timeout, model=m)
        return loop.next_turn(mk(args.narrator), mk(args.designer), mk(args.critic), sheet, out_dir,
                              propose, narrate, parse_json, SCHEMA["required"])
    if args.task == "review":
        if not args.files:
            raise SystemExit("review needs --files")
        return review(client, sheet, out_dir, args.files, args.repo, args.max_tokens)
    (narrate if args.task == "narrate" else propose)(client, sheet, out_dir)


if __name__ == "__main__":
    main()
