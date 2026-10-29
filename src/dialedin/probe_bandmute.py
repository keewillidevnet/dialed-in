"""Exploratory band-mute test: does a head's listening range predict what it does? Not sealed; nothing here gates.

    PYTHONPATH=src python3 -m dialedin.probe_bandmute

The head filter map (results/derived/headmap-c1v5-20261027T152040Z/headmap.json) gives each resolved head a listening
range ("corner", in tokens). This probe mutes bands of heads (their gated output, the o_proj input slice, set to zero
on every token) and measures next-token loss on three kinds of tokens:
  continuation  a word piece that continues a word (decodes without a leading space, alphabetic, after a token that
                ends in a letter): predictable from the last few tokens
  far_repeat    the target completes a trigram that occurred before, at least FAR tokens back, while its two-token
                prefix does not occur in the last FAR tokens: predictable only from far context
  other         every other scored token
Bands (equal size k, the size of the smaller of the short and long sets; heads sorted by corner, ties by layer and head):
  short  the k heads with the shortest corners (all at or below 16 tokens)
  mid    k heads around the median corner
  long   the k heads with the longest corners (all at or above 256 tokens)
Controls: for each band, CONTROL_SEEDS layer-matched random sets: each muted head is replaced by a random head in no
band from the same layer, or, where that layer has none left, from the nearest layer that has one (counted and recorded). Text: sets A and B, tokens 70,000 to 78,191 (8,192), which no earlier run has measured;
scored positions 1,024 to 8,191. SDPA, BF16, batch 1, no adapter.
Written before the run, as a prediction: muting short heads raises continuation loss more than far_repeat loss
(relative to base), muting long heads raises far_repeat loss more than continuation loss, and each band does more
damage to its own kind of token than its layer-matched controls do. Output: results/raw/probe-bandmute-<utc>/bandmute.json.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import glimmer

START, TOKENS, SCORE_FROM = 70000, 8192, 1024
CATS = ("continuation", "far_repeat", "other")
FAR = 256
SHORT_MAX, LONG_MIN = 16, 256
CONTROL_SEEDS = (1, 2, 3)
N_HEADS = 32
LOSS_CHUNK = 1024


# ---------------------------------------------------------------- token categories
def categories(ids, decoded, score_from=SCORE_FROM, far=FAR):
    """Per target position t (score_from .. len-1): 'continuation', 'far_repeat' or 'other'. decoded[i] is the string
    for token ids[i] alone."""
    n = len(ids)
    last_tri, last_bi = {}, {}
    out = {}
    for t in range(2, n):
        bi = (ids[t - 2], ids[t - 1])
        tri = bi + (ids[t],)
        if t >= score_from:
            s, prev = decoded[t], decoded[t - 1]
            cont = bool(s) and not s[0].isspace() and s.isalpha() and bool(prev) and prev[-1].isalpha()
            far_rep = tri in last_tri and t - last_tri[tri] >= far and (bi not in last_bi or t - last_bi[bi] > far)
            out[t] = "far_repeat" if far_rep else ("continuation" if cont else "other")
        last_tri[tri] = t
        last_bi[bi] = t
    return out


# ---------------------------------------------------------------- bands
def bands(corner, layers):
    """corner: [layers][heads] (None where unresolved). Returns k and the short, mid and long head lists [(layer, head)]."""
    heads = sorted(((c, l, h) for i, l in enumerate(layers) for h, c in enumerate(corner[i]) if c is not None))
    short = [(l, h) for c, l, h in heads if c <= SHORT_MAX]
    long_ = [(l, h) for c, l, h in heads if c >= LONG_MIN]
    k = min(len(short), len(long_))
    if k == 0:
        raise SystemExit("no heads in the short or long band")
    m0 = max(0, len(heads) // 2 - k // 2)
    return k, {"short": short[:k], "mid": [(l, h) for _, l, h in heads[m0:m0 + k]], "long": long_[-k:]}


def matched_control(band, in_any_band, seed, n_heads=N_HEADS, layers=range(2, 52)):
    """Each head replaced by a random head in no band (numpy default_rng(seed)), from the same layer if one is left,
    otherwise from the nearest layer that has one (ties: the earlier layer). Returns the heads and the number of heads
    taken from another layer."""
    rng = np.random.default_rng(seed)
    out, used, moved = [], set(), 0
    for l, _ in band:
        for dl in sorted(set(abs(x - l) for x in layers)):
            cand = [ll for ll in (l - dl, l + dl) if ll in layers]
            pool = [(ll, h) for ll in sorted(set(cand)) for h in range(n_heads)
                    if (ll, h) not in in_any_band and (ll, h) not in used]
            if pool:
                break
        else:
            raise SystemExit("no free head left for a control")
        pick = pool[int(rng.integers(len(pool)))]
        moved += pick[0] != l
        used.add(pick); out.append(pick)
    return out, int(moved)


class Muter:
    """Zeroes the o_proj input slice of the chosen heads (their gated outputs) while set."""

    def __init__(self, layers, n_heads=N_HEADS):
        self.layers, self.n_heads, self.mute = layers, n_heads, {}
        self.handles = [layers[i].self_attn.o_proj.register_forward_pre_hook(self._hook(i)) for i in range(len(layers))]

    def _hook(self, i):
        def hook(m, a):
            hs = self.mute.get(i)
            if not hs:
                return None
            x = a[0].clone()
            d = x.shape[-1] // self.n_heads
            for h in hs:
                x[..., h * d:(h + 1) * d] = 0
            return (x,) + tuple(a[1:])
        return hook

    def set(self, heads):
        self.mute = {}
        for l, h in heads:
            self.mute.setdefault(l, []).append(h)

    def remove(self):
        for h in self.handles:
            h.remove()


def token_losses(decoder, logits, ids, score_from=SCORE_FROM):
    x = torch.tensor([ids], device=next(decoder.parameters()).device)
    out = np.empty(len(ids) - score_from)
    with torch.inference_mode():
        h = decoder(input_ids=x, use_cache=False).last_hidden_state[0]
        for a in range(score_from, len(ids), LOSS_CHUNK):
            b = min(a + LOSS_CHUNK, len(ids))
            lp = logits(h[a - 1:b - 1]).log_softmax(-1)
            out[a - score_from:b - score_from] = (-lp.gather(1, x[0, a:b, None])[:, 0]).double().cpu().numpy()
    return out


def run(decoder, layers, logits, texts_ids, cats, conds, log=print):
    """conds: name -> list of (layer index, head). Returns per condition per category: mean loss, token count."""
    muter = Muter(layers)
    res = {}
    try:
        for name, heads in conds.items():
            t = time.time()
            muter.set(heads)
            acc = {c: [] for c in CATS + ("all",)}
            for ids, cat in zip(texts_ids, cats):
                loss = token_losses(decoder, logits, ids)
                for pos, c in cat.items():
                    acc.setdefault(c, []).append(loss[pos - SCORE_FROM])
                acc.setdefault("all", []).extend(loss.tolist())
            res[name] = {c: {"mean_loss": float(np.mean(v)) if v else None, "tokens": len(v)} for c, v in acc.items()}
            log(f"   {name}: {len(heads)} heads muted, all-token loss {res[name]['all']['mean_loss']:.4f} "
                f"({time.time() - t:.0f} s)")
    finally:
        muter.set([]); muter.remove()
    return res


def summarize(res, k):
    """Relative loss change from base per condition and category (None where a category has no tokens), each band
    against its controls, and the written prediction checked (None where a needed value is missing)."""
    base = res["base"]
    ratio = lambda a, b: None if a is None or b is None else a / b - 1
    rel = {n: {c: ratio(r[c]["mean_loss"], base[c]["mean_loss"]) for c in base} for n, r in res.items() if n != "base"}
    out = {"k": k, "relative_change": rel}
    for band in ("short", "mid", "long"):
        ctl = [n for n in rel if n.startswith(f"control_{band}_")]
        out[f"{band}_vs_controls"] = {}
        for c in base:
            vals = [rel[n][c] for n in ctl if rel[n][c] is not None]
            out[f"{band}_vs_controls"][c] = {"band": rel[band][c],
                                             "controls_mean": float(np.mean(vals)) if vals else None,
                                             "controls_max": float(np.max(vals)) if vals else None}
    gt = lambda a, b: None if a is None or b is None else bool(a > b)
    out["prediction_check"] = {
        "short_hurts_continuation_more_than_far_repeat": gt(rel["short"]["continuation"], rel["short"]["far_repeat"]),
        "long_hurts_far_repeat_more_than_continuation": gt(rel["long"]["far_repeat"], rel["long"]["continuation"]),
        "short_beats_its_controls_on_continuation": gt(rel["short"]["continuation"],
                                                       out["short_vs_controls"]["continuation"]["controls_max"]),
        "long_beats_its_controls_on_far_repeat": gt(rel["long"]["far_repeat"],
                                                    out["long_vs_controls"]["far_repeat"]["controls_max"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="criteria/sealed/c7-glimmer-dial-localization-v1.yaml")
    ap.add_argument("--headmap", default="results/derived/headmap-c1v5-20261027T152040Z/headmap.json")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    import yaml
    from .run_c7 import write_json
    repo = os.path.abspath(args.repo)
    c1 = yaml.safe_load(open(os.path.join(repo, "criteria/sealed/c1-glimmer-frequency-response-v5.yaml")))
    spec = yaml.safe_load(open(os.path.join(repo, args.spec)))
    hm = json.load(open(os.path.join(repo, args.headmap)))
    k, bnd = bands(hm["per_head"]["corner_tokens"], hm["per_head"]["layers"])
    in_any = set(x for v in bnd.values() for x in v)
    conds = {"base": []}
    for name, heads in bnd.items():
        conds[name] = heads
    fallbacks = {}
    for name, heads in bnd.items():
        for sd in CONTROL_SEEDS:
            conds[f"control_{name}_{sd}"], fallbacks[f"control_{name}_{sd}"] = matched_control(heads, in_any, sd)
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t0 = time.time()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    head = model.get_output_embeddings()
    mult = getattr(decoder.config, "output_multiplier", None)
    cap = getattr(decoder.config, "final_logit_softcapping", None)

    def logits(h):
        x = head(h)
        if mult is not None:
            x = x * mult
        if cap:
            x = torch.tanh(x / cap) * cap
        return x.float()

    books = [int(g) for s in ("A", "B") for g in c1["inputs"]["prompt_sets"][s]]
    ids, cats = [], []
    for g in books:
        all_ids = tok(texts[g], add_special_tokens=False)["input_ids"]
        x = all_ids[START:START + TOKENS]
        if len(x) != TOKENS:
            raise SystemExit(f"book {g} has only {len(all_ids)} tokens")
        ids.append(x)
        cats.append(categories(x, [tok.decode([i]) for i in x]))
    counts = {c: sum(1 for cat in cats for v in cat.values() if v == c) for c in ("continuation", "far_repeat", "other")}
    print(f"== load {time.time() - t0:.0f} s; books {books}; k = {k} heads per band; token counts {counts}", flush=True)
    for name, heads in bnd.items():
        cs = [hm["per_head"]["corner_tokens"][l - 2][h] for l, h in heads]
        print(f"   band {name}: corners {min(cs)} to {max(cs)} tokens, layers {sorted(set(l for l, _ in heads))}")
    print(f"   control heads taken from a neighboring layer: {fallbacks}")
    res = run(decoder, layers, logits, ids, cats, conds, log=lambda m: print(m, flush=True))
    summ = summarize(res, k)
    run_id = "probe-bandmute-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(repo, "results", "raw", run_id)
    os.makedirs(out_dir)
    write_json(os.path.join(out_dir, "bandmute.json"),
               {"exploratory": True, "sealed": False, "what": __doc__.split("\n\n")[2].strip(), "books": books,
                "start": START, "tokens": TOKENS, "score_from": SCORE_FROM, "far": FAR, "token_counts": counts,
                "headmap": args.headmap, "bands": {n: [list(x) for x in v] for n, v in bnd.items()},
                "conditions": {n: [list(x) for x in v] for n, v in conds.items()}, "control_fallbacks": fallbacks,
                "platform": platform,
                "results": res, "summary": summ}, indent=1)
    rel = summ["relative_change"]
    print("== loss change vs base (percent):        continuation   far_repeat     other       all")
    pc = lambda v, w: f"{'n/a':>{w}}" if v is None else f"{100 * v:+{w}.2f}"
    for n in rel:
        r = rel[n]
        print(f"   {n:22s} {pc(r['continuation'], 12)} {pc(r['far_repeat'], 12)} {pc(r['other'], 9)} {pc(r['all'], 9)}")
    print(f"== prediction check: {summ['prediction_check']}")
    print(f"== exploratory; written to {os.path.relpath(out_dir, repo)}/bandmute.json")


if __name__ == "__main__":
    main()
