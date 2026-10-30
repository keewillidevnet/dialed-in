"""Exploratory: does the C5 dial change something you can see in the model's text? Not sealed; nothing here gates or
confirms anything. Every prompt and every output is saved, chosen or not.

    PYTHONPATH=src python3 -m dialedin.probe_behavior

Question. C5 (exploratory) and C6 (sealed, B0: +1.8 percent at scale 2, below its 5 percent bar, rising at every step,
8 of 10 books) found that after a switch from book x to book y, the dial raises next-token loss on y. One reading: the
dialed model holds on to the old book longer. This probe makes that visible and measures it against a control.

Inputs: the 10 books of sets B and C, tokens 80,000 onward (no earlier run used them; the band-mute probe ended at
78,191). Ten ordered pairs (x, y): each book is x once, the next book in the list is y.
  switch prompt: x's tokens [80,000, 86,144), then y's tokens [86,144, 86,656) (6,144 of x, then 512 of y)
  clean prompt:  y's tokens [80,000, 86,656)
Old-book words: token ids that start a capitalized word of 3+ characters, occur at least 3 times in x's 6,144 tokens,
and never occur in y's tokens [80,000, 90,000) (character names and places of x, at most 50, most frequent first).

Per pair and condition:
  pull: the probability the model gives to old-book words, summed over that set and averaged over the 512 positions
        after the seam (teacher-forced on y's real text). The same on the clean prompt is the floor (x never appears).
  excess loss: mean next-token loss on y's 512 tokens, switch prompt minus clean prompt (C6's measure, shorter).
  text: 48 tokens of greedy continuation of the switch prompt, decoded, and how many old-book words it uses.
  perplexity: exp of the mean loss on the clean prompt's last 512 tokens, as a ratio to scale 0.
Conditions: S0, S2 (the sealed scale), S4 (beyond any sealed run; labeled as such), then a random adapter matched to
the trained one module by module in norm (run_c6.random_control, seed 1000) at scales 2 and 4.

The showcase rule, fixed here before the run: among pairs where S2 raises pull over S0 and the random adapter at 2
moves it by less than half as much, the pair with the largest S2 rise; and the same for S4 against random at 4. If no
pair qualifies, there is no showcase and that is reported. Output: results/raw/probe-behavior-<utc>/behavior.json.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch

from . import glimmer
from . import pilot_c5 as P

START, CONTEXT, AFTER, GEN = 80000, 6144, 512, 48
VOCAB_REGION = 10000                  # y's tokens [START, START + VOCAB_REGION) define "never in y"
MIN_COUNT, MAX_WORDS = 3, 50
TRAINED = (("S0", 0.0), ("S2", 2.0), ("S4", 4.0))
RANDOM = (("R2", 2.0), ("R4", 4.0))
RANDOM_SEED = 1000
WARN_ONLY = [False]                   # set if generate needed warn_only determinism; recorded in the output


def old_book_words(x_ids, y_region_ids, decode) -> list:
    """Token ids that start a capitalized word of 3+ characters, occur at least MIN_COUNT times in x_ids and never in
    y_region_ids; most frequent first, at most MAX_WORDS."""
    y_set = set(y_region_ids)
    counts = Counter(i for i in x_ids if i not in y_set)
    out = []
    for i, n in counts.most_common():
        if n < MIN_COUNT or len(out) >= MAX_WORDS:
            break
        w = decode([i])
        if re.fullmatch(r" [A-Z][a-z]{2,}", w):
            out.append(int(i))
    return out


def pairs(books: list) -> list:
    return [(books[i], books[(i + 1) % len(books)]) for i in range(len(books))]


def showcase(rows: dict, trained: str, control: str):
    """rows: pair label -> {condition: {"pull": float}}. The pair with the largest rise in pull from S0 to `trained`
    among pairs where that rise is positive and the control's change is under half of it; None if none qualifies."""
    best, best_rise = None, 0.0
    for label, r in rows.items():
        rise = r[trained]["pull"] - r["S0"]["pull"]
        ctl = abs(r[control]["pull"] - r["S0"]["pull"])
        if rise > 0 and ctl < 0.5 * rise and rise > best_rise:
            best, best_rise = label, rise
    return best


def measure(s, sw_ids, cl_ids, words, tok):
    """Pull, excess loss and perplexity from one forward of each prompt, then a greedy continuation."""
    w = torch.tensor(words, device=s.device, dtype=torch.long)
    out = {}
    with torch.inference_mode():
        for name, ids in (("switch", sw_ids), ("clean", cl_ids)):
            h = s.decoder(input_ids=ids, use_cache=False).last_hidden_state[0]
            lp = s.logits(h[CONTEXT - 1:CONTEXT + AFTER - 1]).log_softmax(-1)      # predicts y's 512 tokens
            tgt = ids[0, CONTEXT:CONTEXT + AFTER]
            loss = -lp.gather(1, tgt[:, None])[:, 0]
            out[name] = {"pull": float(lp[:, w].exp().sum(-1).mean()) if len(words) else 0.0,
                         "loss": float(loss.mean())}
        try:
            gen = s.model.generate(input_ids=sw_ids, max_new_tokens=GEN, do_sample=False)
        except RuntimeError as e:                 # an op in generate with no deterministic kernel: warn instead, recorded
            if "deterministic" not in str(e) or WARN_ONLY[0]:
                raise
            WARN_ONLY[0] = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            gen = s.model.generate(input_ids=sw_ids, max_new_tokens=GEN, do_sample=False)
        gen = gen[0, sw_ids.shape[1]:].tolist()
    return {"pull": out["switch"]["pull"], "pull_clean": out["clean"]["pull"],
            "excess_loss": out["switch"]["loss"] - out["clean"]["loss"], "clean_loss": out["clean"]["loss"],
            "text": tok.decode(gen), "old_words_in_text": int(sum(1 for t in gen if t in set(words)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    import yaml
    from .run_c6 import NullLedger, Session, random_control, write_json
    repo = os.path.abspath(args.repo)
    spec = yaml.safe_load(open(os.path.join(repo, args.spec)))
    tj = json.load(open(os.path.join(repo, os.path.dirname(spec["adapter"]["file"]), "training.json")))
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t0 = time.time()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    s = Session(spec, texts, model, decoder, layers, tok, platform, repo, NullLedger())
    if s.adapter_sha != tj["adapter_sha256"]:
        raise SystemExit("the adapter file differs from the one C5 trained")
    books = [int(g) for st in ("B", "C") for g in spec["inputs"]["prompt_sets"][st]]
    toks = {g: tok(texts[g], add_special_tokens=False)["input_ids"] for g in books}
    short = [g for g in books if len(toks[g]) < START + VOCAB_REGION]
    if short:
        raise SystemExit(f"books too short for tokens {START} to {START + VOCAB_REGION}: {short}")
    print(f"== load {time.time() - t0:.0f} s; adapter {s.adapter_sha[:12]}; books {books}", flush=True)
    decode = lambda ids: tok.decode(ids)
    prompts = {}
    for x, y in pairs(books):
        sw = toks[x][START:START + CONTEXT] + toks[y][START + CONTEXT:START + CONTEXT + AFTER]
        cl = toks[y][START:START + CONTEXT + AFTER]
        words = old_book_words(toks[x][START:START + CONTEXT], toks[y][START:START + VOCAB_REGION], decode)
        prompts[f"{x}->{y}"] = {"x": x, "y": y, "words": words, "word_strings": [decode([i]) for i in words],
                                "sw": torch.tensor([sw], device=s.device), "cl": torch.tensor([cl], device=s.device),
                                "seam_text": decode(sw[CONTEXT - 40:CONTEXT + 40])}
    rows = {k: {} for k in prompts}

    def run(conds):
        for c, scale in conds:
            P.LoRALinear.scale = scale
            t = time.time()
            for k, p in prompts.items():
                rows[k][c] = measure(s, p["sw"], p["cl"], p["words"], tok)
            pull = np.mean([rows[k][c]["pull"] for k in rows])
            print(f"   {c}: mean pull {pull:.5f}, mean excess loss {np.mean([rows[k][c]['excess_loss'] for k in rows]):+.4f}"
                  f" ({time.time() - t:.0f} s)", flush=True)
        P.LoRALinear.scale = 0.0

    run(TRAINED)
    norms = random_control(s, RANDOM_SEED)                     # replaces the trained factors; trained runs are done
    run(RANDOM)
    for k in rows:
        base = np.exp(rows[k]["S0"]["clean_loss"])
        for c in rows[k]:
            rows[k][c]["perplexity_ratio"] = float(np.exp(rows[k][c]["clean_loss"]) / base)
    show = {"S2": showcase(rows, "S2", "R2"), "S4": showcase(rows, "S4", "R4")}
    summary = {c: {"mean_pull": float(np.mean([rows[k][c]["pull"] for k in rows])),
                   "pairs_pull_up_vs_S0": int(sum(rows[k][c]["pull"] > rows[k]["S0"]["pull"] for k in rows)),
                   "mean_excess_loss": float(np.mean([rows[k][c]["excess_loss"] for k in rows])),
                   "median_perplexity_ratio": float(np.median([rows[k][c]["perplexity_ratio"] for k in rows])),
                   "old_words_in_text": int(sum(rows[k][c]["old_words_in_text"] for k in rows))}
               for c, _ in TRAINED + RANDOM}
    run_id = "probe-behavior-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(repo, "results", "raw", run_id)
    os.makedirs(out_dir)
    write_json(os.path.join(out_dir, "behavior.json"),
               {"exploratory": True, "sealed": False, "what": __doc__.split("\n\n")[2].strip(),
                "design": __doc__.split("\n\n")[3:6], "generate_warn_only": WARN_ONLY[0], "start": START, "context": CONTEXT, "after": AFTER,
                "generated_tokens": GEN, "conditions": dict(TRAINED + RANDOM), "random_seed": RANDOM_SEED,
                "random_norms": norms, "adapter_sha256": s.adapter_sha, "platform": platform,
                "deterministic": "torch.use_deterministic_algorithms(True); greedy decoding",
                "summary": summary, "showcase": show,
                "pairs": {k: {"x": p["x"], "y": p["y"], "old_book_words": p["word_strings"], "seam_text": p["seam_text"],
                              "conditions": rows[k]} for k, p in prompts.items()}}, indent=1)
    print("== condition  mean pull   pairs up   excess loss   perplexity   old words in text")
    for c, v in summary.items():
        print(f"   {c:4s}      {v['mean_pull']:.5f}   {v['pairs_pull_up_vs_S0']:2d}/10     {v['mean_excess_loss']:+.4f}"
              f"      x{v['median_perplexity_ratio']:.4f}   {v['old_words_in_text']}")
    for c, k in show.items():
        if k is None:
            print(f"== showcase {c}: no pair qualifies")
            continue
        r = rows[k]
        print(f"== showcase {c}: pair {k}; pull S0 {r['S0']['pull']:.5f}, {c} {r[c]['pull']:.5f}, "
              f"random {r['R' + c[1:]]['pull']:.5f}")
        for cc in ("S0", c, "R" + c[1:]):
            print(f"   [{cc}] {r[cc]['text']!r}")
    print(f"== exploratory; written to {os.path.relpath(out_dir, repo)}/behavior.json")


if __name__ == "__main__":
    main()
