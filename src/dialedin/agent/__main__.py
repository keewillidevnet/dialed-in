"""Muse Spark lab agent.

    export META_API_KEY   (set it with: read -rs META_API_KEY; export META_API_KEY)
    PYTHONPATH=src python3 -m dialedin.agent facts      # print the fact sheet (no API call)
    PYTHONPATH=src python3 -m dialedin.agent narrate    # Muse explains the results; every number checked
    PYTHONPATH=src python3 -m dialedin.agent propose    # Muse drafts the next pre-registered experiment
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
         "Use only numbers that appear in the fact sheet, rounded as written there or to fewer digits, and put "
         "the fact id in square brackets right after each number, like 0.137 [F12]. Do not compute new numbers. "
         "If a claim needs a number that is not on the sheet, say it qualitatively.")

NARRATE = ("You are Muse, the AI scientist of the Dialed In lab. The lab measures the frequency response of the "
           "Muse Glimmer-30B language model: it adds a small sine wave of period N tokens to the model's internal "
           "signal after layer 1 and measures how strongly each later attention head responds, for N from 2 to "
           "4,096 tokens. Every sealed experiment (C1 to C7) was pre-registered and sealed by hash before it ran; facts "
           "labeled EXPLORATORY come from unsealed probes and must be called exploratory. Explain the story of the "
           "results for a technical judge, in under 350 words: what the lab measured, which built-in settings failed "
           "to move it (C2 to C4), the trained dial (C5) and its near miss, what it does to behavior (C5, C6), where "
           "the dial acts (C7), and what the exploratory head map adds. Give each sealed outcome by its code. Be "
           "candid about null and failed results. " + STYLE)

PROPOSE = ("You are Muse, the AI scientist of the Dialed In lab. Using the results on the fact sheet, propose the "
           "single most informative next experiment the lab can run in under 2 GPU hours on one 96 GB GPU (BF16) "
           "with the existing harness. Available tools: the tilt measurement at 32,768 tokens (about 13 s per pass, "
           "7 periods, shams, 5 books per set); the frozen C5 adapter at any scale with any subset of its 32 modules "
           "switched on (layers 2 to 5: q, k, v, gate and o projections and MLP); muting chosen attention heads; "
           "per-token next-token loss on held-out book text; the drive direction, amplitude and injection layer; "
           "the local window size and RoPE base; shuffled or repeated-token text; and, as a separately labeled side "
           "check only, the same model loaded in 4-bit NF4. Book sets A, B and C hold 5 books each; text beyond "
           "token 70,000 has not been used. No new training. The proposal must test something the facts leave "
           "open, fix its pass rule before any data with a numeric threshold, and include a control. "
           "Reply with one JSON object only, no other text, with exactly these keys: " + ", ".join(
               ["title", "question", "prediction", "intervention", "conditions", "control", "books", "primary_metric",
                "pass_rule", "failure_meaning", "gpu_hours_estimate", "risks", "why_this_next"]) +
           ". conditions and risks are lists of strings; gpu_hours_estimate is a number; the rest are strings. " + STYLE)

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["title", "question", "prediction", "intervention", "conditions", "control", "books",
                       "primary_metric", "pass_rule", "failure_meaning", "gpu_hours_estimate", "risks", "why_this_next"],
          "properties": {k: {"type": "string"} for k in
                         ["title", "question", "prediction", "intervention", "control", "books", "primary_metric",
                          "pass_rule", "failure_meaning", "why_this_next"]} |
                        {"conditions": {"type": "array", "items": {"type": "string"}},
                         "gpu_hours_estimate": {"type": "number"},
                         "risks": {"type": "array", "items": {"type": "string"}}}}


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
                     {"role": "user", "content": "These numbers are not on the fact sheet or have no matching fact: "
                      f"{report['unverified_numbers']}; unknown citations: {report['unknown_citations']}. "
                      "Rewrite the explanation using only fact-sheet numbers, each with its citation."}]
        text = client.chat(messages, max_tokens=12000)
        report = verify.check(text, sheet)
    path = os.path.join(out_dir, f"narration-{stamp()}-{getattr(client, 'model', 'muse')}.md")
    sources = "\n".join(f"- [{f['id']}] {f['label']}: `{f['source']}`" for f in sheet.facts if f["id"] in report["citations_used"])
    with open(path, "w") as fh:
        fh.write(text.strip() + "\n\n---\nVerification: " + ("every number matches a cited fact." if report["ok"] else
                 f"UNVERIFIED numbers {report['unverified_numbers']}, unknown citations {report['unknown_citations']}.")
                 + "\n\nSources:\n" + sources + "\n")
    print(text.strip()); print("\n== verification:", "PASSED" if report["ok"] else report); print("== wrote", path)


def propose(client, sheet, out_dir):
    messages = [{"role": "system", "content": PROPOSE},
                {"role": "user", "content": "Fact sheet:\n" + sheet.text()}]
    raw = client.chat(messages, temperature=0.4, max_tokens=16000)
    prop = parse_json(raw)
    missing = [k for k in SCHEMA["required"] if k not in prop]
    report = verify.check(json.dumps(prop), sheet)
    prop["_checks"] = {"missing_fields": missing, "numbers": report,
                       "within_budget": isinstance(prop.get("gpu_hours_estimate"), (int, float)) and prop["gpu_hours_estimate"] <= 2}
    prop["_model"] = getattr(client, "model", "unknown")
    path = os.path.join(out_dir, f"proposal-{stamp()}-{getattr(client, 'model', 'muse')}.json")
    json.dump(prop, open(path, "w"), indent=2)
    print(json.dumps(prop, indent=2)); print("== wrote", path)
    print("== this is a draft for review: it becomes an experiment only after independent review and a human seal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["facts", "narrate", "propose"])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default=None, help="muse-spark-1.3 (default) or muse-wonder-1.0")
    args = ap.parse_args()
    sheet = facts.build(args.repo)
    if not sheet.facts:
        raise SystemExit("no results found under results/derived/")
    if args.task == "facts":
        print(sheet.text()); return
    from .muse import MuseClient
    out_dir = os.path.join(args.repo, "results", "agent")
    os.makedirs(out_dir, exist_ok=True)
    client = MuseClient(log_path=os.path.join(out_dir, "calls.jsonl"), **({"model": args.model} if args.model else {}))
    (narrate if args.task == "narrate" else propose)(client, sheet, out_dir)


if __name__ == "__main__":
    main()
