"""Run C6 exactly as sealed in criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c6                     # the sealed run (one per seal)
    PYTHONPATH=src python3 -m dialedin.run_c6 --smoke --spec criteria/drafts/c6-glimmer-dial-topic-switch-v2.yaml

Forked from run_c5.py (phase R). C6 is a confirmatory test of one exploratory C5 finding: the trained adapter from run
c5v1-20261029T025343Z, frozen and hash-checked, raised the excess next-token loss after a topic switch at every step of
its scale (C5 behavior.json: 509.6, 517.9, 532.0, 547.2, 567.5 nats at scales 0 to 2, set A). C6 repeats that
measurement on text C5 never measured: sets B and C, tokens 60,000 to 68,191, all 90 ordered pairs of the 10 books.
Analysis: dialedin.analyze_c6.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from . import glimmer
from . import pilot_c5 as P

SEALED = "criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml"
TAG = "c6v2"
CONDITIONS = {"S0": 0.0, "S0.5": 0.5, "S1": 1.0, "S1.5": 1.5, "S2": 2.0}
ORDER = ("S0", "S0.5", "S1", "S1.5", "S2")
CHECK_BOOK = 1260                   # adapter and logits checks read the first 2,048 tokens of this book
CHECK_LAYER = 10
LOSS_CHUNK = 1024
LOGIT_TOL = 0.005


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str, obj, indent=None):
    """Atomic: a file either exists complete or not at all. Never overwrites."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; raw results are never overwritten.")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_failure(out_dir: str, name: str, err: BaseException):
    path = os.path.join(out_dir, f"{name}.FAILED.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
    write_json(path, {"utc": now(), "what": name, "error": repr(err), "traceback": traceback.format_exc()}, indent=1)
    return path


class Ledger:
    """Append-only wall-clock ledger under results/ledger/, one JSON line per step (as C1 to C5)."""

    def __init__(self, ledger_root: str, seal_hash: str, cap_s: float):
        os.makedirs(ledger_root, exist_ok=True)
        self.path = os.path.join(ledger_root, f"{TAG}_{seal_hash[:12]}.jsonl")
        self.cap_s = cap_s
        self.seconds = 0.0
        if os.path.exists(self.path):
            for line in open(self.path):
                if line.strip():
                    self.seconds += json.loads(line)["seconds"]
        self.mark = time.time()
        self.last_pass = 0.0

    def add(self, what: str):
        t = time.time()
        dt = t - self.mark
        self.mark = t
        self.seconds += dt
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"utc": now(), "what": what, "seconds": round(dt, 3)}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return dt

    def check(self, upcoming_s: float = None):
        upcoming = self.last_pass if upcoming_s is None else upcoming_s
        if self.seconds + upcoming > self.cap_s:
            raise SystemExit(f"Budget cap: {self.seconds / 3600:.2f} wall hours used of {self.cap_s / 3600:.2f}. "
                             f"Stopping as the sealed criteria require.")


class NullLedger:
    seconds, last_pass = 0.0, 0.0

    def add(self, what):
        return 0.0

    def check(self, upcoming_s=None):
        return None


class RunLock:
    """One runner process per seal at a time. A stale lock after a hard kill must be removed by hand."""

    def __init__(self, ledger_root: str, seal_hash: str):
        os.makedirs(ledger_root, exist_ok=True)
        self.path = os.path.join(ledger_root, f"{TAG}_{seal_hash[:12]}.lock")
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{self.path} exists: another runner is active, or one was killed. "
                             f"If no runner is active, delete that file and start again.")
        os.write(fd, f"{os.getpid()} {now()}\n".encode())
        os.close(fd)

    def release(self):
        if os.path.exists(self.path):
            os.remove(self.path)


class Session:
    """Glimmer with the frozen C5 adapter attached (hash checked), hooks off, inference only."""

    def __init__(self, spec, texts, model, decoder, layers, tok, platform, repo, ledger, allow_unfilled=False):
        self.spec, self.texts, self.model, self.decoder, self.layers = spec, texts, model, decoder, layers
        self.tok, self.platform, self.repo, self.ledger = tok, platform, repo, ledger
        self.device = next(model.parameters()).device
        self.unadapted_ref = self._reference_input()
        ad_spec = spec["adapter"]
        path = os.path.join(repo, ad_spec["file"])
        self.adapter_sha = sha_file(path)
        if allow_unfilled and ad_spec["sha256"] == "FILL_AT_SEAL":
            print(f"== smoke: the adapter's sha256 is {self.adapter_sha}; fill it into the spec before sealing")
        elif self.adapter_sha != ad_spec["sha256"]:
            raise SystemExit(f"adapter {ad_spec['file']} sha256 {self.adapter_sha} differs from the sealed {ad_spec['sha256']}")
        ad = torch.load(path, map_location="cpu")
        names, params = P.attach_lora(layers, ad_spec["layers"], ad_spec["rank"], ad_spec["alpha"], family=ad_spec["family"])
        if names != ad["modules"]:
            raise SystemExit("the adapter's modules differ from the sealed family and layers")
        with torch.no_grad():
            for (a, b), pa, pb in zip(zip(ad["A"], ad["B"]), params[0::2], params[1::2]):
                pa.copy_(a.to(pa.device)); pb.copy_(b.to(pb.device))
        for p in params:
            p.requires_grad_(False)
        self.adapter_modules = names
        self.params = params
        P.LoRALinear.scale = 0.0
        self.head = model.get_output_embeddings()
        self.softcap = getattr(decoder.config, "final_logit_softcapping", None)
        self.multiplier = getattr(decoder.config, "output_multiplier", None)
        full_text = getattr(getattr(model, "config", None), "text_config", None)
        if full_text is not None:
            if self.softcap is None or self.multiplier is None:
                raise SystemExit("final_logit_softcapping or output_multiplier missing from the text config")
            if (self.softcap, self.multiplier) != (full_text.final_logit_softcapping, full_text.output_multiplier):
                raise SystemExit("softcap or multiplier differs between the decoder config and the full model's text_config")

    def _check_ids(self):
        return torch.tensor([self.tok(self.texts[CHECK_BOOK], add_special_tokens=False)["input_ids"][:2048]],
                            device=self.device)

    def _reference_input(self) -> torch.Tensor:
        seen = []
        h = self.layers[CHECK_LAYER].self_attn.o_proj.register_forward_pre_hook(
            lambda m, a: seen.append(a[0][0].detach().clone()))
        try:
            with torch.inference_mode():
                self.decoder(input_ids=self._check_ids(), use_cache=False)
        finally:
            h.remove()
        return seen[0]

    def set_condition(self, cond):
        P.LoRALinear.scale = CONDITIONS[cond]

    def adapter_checks(self) -> dict:
        try:
            self.set_condition("S0"); s0 = self._reference_input()
            self.set_condition("S1"); s1 = self._reference_input()
            self.set_condition("S0"); back = self._reference_input()
        finally:
            self.set_condition("S0")
        out = {"scale0_equals_unadapted": bool(torch.equal(s0, self.unadapted_ref)),
               "scale1_changes": bool((s1 != s0).any()), "scale0_restored": bool(torch.equal(back, s0))}
        out["as_expected"] = all(out.values())
        return out

    def logits(self, h):
        x = self.head(h)
        if self.multiplier is not None:
            x = x * self.multiplier
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x.float()

    def logits_check(self) -> dict:
        self.set_condition("S0")
        x = self._check_ids()
        with torch.inference_mode():
            ref = self.model(input_ids=x, use_cache=False).logits[0].float()
            ours = self.logits(self.decoder(input_ids=x, use_cache=False).last_hidden_state[0])
        diff, scale = float((ours - ref).abs().max()), float(ref.abs().max())
        return {"max_abs_diff": diff, "max_abs_logit": scale, "tolerance": LOGIT_TOL * scale,
                "as_expected": bool(diff <= LOGIT_TOL * scale)}

    def token_losses(self, ids: list, start: int, end: int) -> np.ndarray:
        """Next-token loss for tokens start .. end-1 of ids (predicted from positions start-1 .. end-2), float64."""
        x = torch.tensor([ids], device=self.device)
        out = np.empty(end - start)
        with torch.inference_mode():
            h = self.decoder(input_ids=x, use_cache=False).last_hidden_state[0]
            for a in range(start, end, LOSS_CHUNK):
                b = min(a + LOSS_CHUNK, end)
                lp = self.logits(h[a - 1:b - 1]).log_softmax(-1)
                out[a - start:b - start] = (-lp.gather(1, x[0, a:b, None])[:, 0]).double().cpu().numpy()
        return out


def random_control(s: Session, seed: int) -> list:
    """Replaces the trained adapter, module by module, with random low-rank factors (torch.Generator seeded
    seed + module index, standard normal) scaled so that the Frobenius norm of B A equals the trained module's.
    Returns per module the trained and new norms. Used only after the trained measurement is complete."""
    def ba_norm(A, B):                       # ||B A||_F = sqrt(trace((B^T B)(A A^T))), computed at rank size
        return float(torch.sqrt(torch.trace((B.T @ B) @ (A @ A.T)).clamp_min(0)))
    rows = []
    with torch.no_grad():
        for i, (A, B) in enumerate(zip(s.params[0::2], s.params[1::2])):
            target = ba_norm(A.float(), B.float())
            g = torch.Generator().manual_seed(seed + i)
            A2 = torch.randn(tuple(A.shape), generator=g, dtype=torch.float32)
            B2 = torch.randn(tuple(B.shape), generator=g, dtype=torch.float32)
            k = (target / ba_norm(A2, B2)) ** 0.5 if target > 0 else 0.0
            A.copy_((A2 * k).to(A.device, A.dtype)); B.copy_((B2 * k).to(B.device, B.dtype))
            rows.append({"module": s.adapter_modules[i], "trained_norm": target, "control_norm": ba_norm(A.float(), B.float())})
    return rows


def switch_inputs(spec, tok, texts, gids=None, start=None) -> dict:
    """Clean: each book's tokens [start, start + context + after). Switch, for every ordered pair (x, y) of different
    books: x's first `context` of those tokens, then y's last `after`. Noise: y's clean tokens with the first
    `noise_tokens` after the context replaced by tokens drawn from y's own context tokens (default_rng(noise_seed))."""
    sw = dict(spec["switch"])
    if start is not None:
        sw["start"] = start
    L = sw["context"] + sw["after"]
    if gids is None:
        gids = [int(g) for books in spec["inputs"]["prompt_sets"].values() for g in books]
    clean = {}
    for g in gids:
        body = tok(texts[g], add_special_tokens=False)["input_ids"]
        seg = body[sw["start"]:sw["start"] + L]
        assert len(seg) == L, (g, len(body))
        clean[g] = list(seg)
    rng = np.random.default_rng(sw["noise_seed"])
    noise = {}
    for g in gids:
        junk = rng.choice(np.asarray(clean[g][:sw["context"]]), size=sw["noise_tokens"]).tolist()
        c = sw["context"]
        noise[g] = clean[g][:c] + [int(t) for t in junk] + clean[g][c + sw["noise_tokens"]:]
    pairs = [(x, y) for x in gids for y in gids if x != y]
    switch = {(x, y): clean[x][:sw["context"]] + clean[y][sw["context"]:] for x, y in pairs}
    return {"clean": clean, "noise": noise, "switch": switch, "pairs": pairs, "gids": gids}


def measure(s: Session, inputs: dict, conditions=ORDER) -> dict:
    """Per condition: clean per-token losses per book, switch excess per ordered pair, noise excess per book.
    Stores per-pair and per-book totals, mean curves, and the clean mean loss."""
    sw = s.spec["switch"]
    c, n = sw["context"], sw["noise_tokens"]
    L = c + sw["after"]
    out = {}
    for cond in conditions:
        s.set_condition(cond)
        clean = {}
        for g, ids in inputs["clean"].items():
            s.ledger.check()
            clean[g] = s.token_losses(ids, c, L)
            s.ledger.last_pass = s.ledger.add(f"clean {cond} {g}")
        pair_total, curves = {}, []
        for (x, y) in inputs["pairs"]:
            s.ledger.check()
            e = s.token_losses(inputs["switch"][(x, y)], c, L) - clean[y]
            s.ledger.last_pass = s.ledger.add(f"switch {cond} {x}->{y}")
            pair_total[f"{x}->{y}"] = float(e.sum())
            curves.append(e)
        noise_total, ncurves = {}, []
        for g, ids in inputs["noise"].items():
            s.ledger.check()
            e = s.token_losses(ids, c + n, L) - clean[g][n:]
            s.ledger.last_pass = s.ledger.add(f"noise {cond} {g}")
            noise_total[str(g)] = float(e.sum())
            ncurves.append(e)
        out[cond] = {"clean_mean_loss": {str(g): float(v.mean()) for g, v in clean.items()},
                     "switch_pair_total": pair_total, "switch_curve_mean": np.mean(curves, axis=0).astype(np.float32).tolist(),
                     "noise_book_total": noise_total, "noise_curve_mean": np.mean(ncurves, axis=0).astype(np.float32).tolist()}
        m = float(np.mean(list(pair_total.values())))
        print(f"   {cond}: switch excess {m:.1f} nats (mean over {len(pair_total)} pairs); "
              f"noise excess {np.mean(list(noise_total.values())):.1f} nats", flush=True)
    s.set_condition("S0")
    return out


def runs_under_seal(raw_root: str, seal_hash: str) -> list:
    runs = []
    for p in glob.glob(os.path.join(raw_root, "*", "STARTED.json")):
        d = json.load(open(p))
        if d.get("seal_sha256") == seal_hash:
            run_dir = os.path.dirname(p)
            runs.append({"run_id": os.path.basename(run_dir), "started_utc": d.get("utc"),
                         "complete": os.path.exists(os.path.join(run_dir, "switch.json")),
                         "restart_reason": d.get("restart_reason")})
    return sorted(runs, key=lambda r: (r["started_utc"] or "", r["run_id"]))


def admit(raw_root: str, out_dir: str, seal_hash: str, restart_reason: str = None) -> list:
    """One complete run per seal. A new run is refused if any run under the seal is complete, and otherwise needs a
    written infrastructure reason if any run started. Writes STARTED.json."""
    runs = runs_under_seal(raw_root, seal_hash) if os.path.isdir(raw_root) else []
    if [r for r in runs if r["complete"]]:
        raise SystemExit(f"A run under this seal is complete: {[r['run_id'] for r in runs if r['complete']]}. No new run.")
    if runs and not restart_reason:
        raise SystemExit(f"Runs already started under this seal: {runs}. A new run needs --restart-reason "
                         f"(an infrastructure failure); the old run is kept and reported.")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "STARTED.json"), {"utc": now(), "seal_sha256": seal_hash,
                                                      "restart_reason": restart_reason, "earlier_runs": runs}, indent=1)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--restart-reason")
    ap.add_argument("--smoke", action="store_true", help="pre-seal dry run: 2 books, S0 and S2, writes nothing to results/")
    ap.add_argument("--spec", help="draft spec for --smoke")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    if args.smoke:
        spec = yaml.safe_load(open(args.spec))
        texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
        glimmer.configure_determinism()
        t = time.time()
        model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
        print(f"== load {time.time() - t:.0f} s; preflight passed: {platform}")
        for g in [int(g) for books in spec["inputs"]["prompt_sets"].values() for g in books]:
            n = len(tok(texts[g], add_special_tokens=False)["input_ids"])
            print(f"   book {g}: {n} tokens (needs {spec['switch']['start'] + spec['switch']['context'] + spec['switch']['after']})")
        s = Session(spec, texts, model, decoder, layers, tok, platform, repo, NullLedger(), allow_unfilled=True)
        print(f"== adapter {spec['adapter']['file']}: {len(s.adapter_modules)} modules, sha256 {s.adapter_sha}")
        print(f"== adapter checks: {s.adapter_checks()}")
        lc = s.logits_check()
        print(f"== perplexity path: max |logit difference| {lc['max_abs_diff']:.4g} of max |logit| {lc['max_abs_logit']:.4g}; "
              f"as expected: {lc['as_expected']}")
        gids = [int(g) for books in spec["inputs"]["prompt_sets"].values() for g in books][:2]
        inputs = switch_inputs(spec, tok, texts, gids)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        m = measure(s, inputs, ("S0", "S2"))
        n_fwd = 2 * (len(inputs["clean"]) + len(inputs["pairs"]) + len(inputs["noise"]))
        print(f"== {n_fwd} forwards in {time.time() - t:.1f} s ({(time.time() - t) / n_fwd:.2f} s each), "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; all finite: "
              f"{all(np.isfinite(v) for c in m.values() for v in c['switch_pair_total'].values())}")
        print("== smoke test PASSED" if s.adapter_checks()["as_expected"] and lc["as_expected"] else "== smoke test FAILED")
        return

    raw_root = os.path.join(repo, "results", "raw")
    ledger_root = os.path.join(repo, "results", "ledger")
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    ad_path = os.path.join(repo, spec["adapter"]["file"])
    ad_sha = sha_file(ad_path)                         # before admit, so a bad fill costs no run
    c5_training = json.load(open(os.path.join(os.path.dirname(ad_path), "training.json")))
    if not (ad_sha == spec["adapter"]["sha256"] == c5_training["adapter_sha256"]):
        raise SystemExit(f"adapter sha256 {ad_sha}, sealed {spec['adapter']['sha256']}, C5 training.json "
                         f"{c5_training['adapter_sha256']}: they must all agree")
    run_id = datetime.now(timezone.utc).strftime(f"{TAG}-%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(raw_root, run_id)
    lock = RunLock(ledger_root, seal_hash)
    try:
        admit(raw_root, out_dir, seal_hash, args.restart_reason)
        print(f"== Run {run_id}")
        ledger = Ledger(ledger_root, seal_hash, spec["budget"]["max_gpu_hours"] * 3600)
        try:
            texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
            glimmer.configure_determinism()
            model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
            ledger.add("startup and model load")
            s = Session(spec, texts, model, decoder, layers, tok, platform, repo, ledger)
            checks = {"adapter": s.adapter_checks(), "logits": s.logits_check()}
            print(f"   checks: {checks}", flush=True)
            if not (checks["adapter"]["as_expected"] and checks["logits"]["as_expected"]):
                write_json(os.path.join(out_dir, "checks.json"), {"utc": now(), "checks": checks,
                                                                   "stopped": "setup checks failed", "outcome": "U"}, indent=1)
                raise SystemExit("C6 is not run, as sealed: setup checks failed. Outcome U.")
            inputs = switch_inputs(spec, tok, texts)
            rep_spec = spec["reproduction_of_c5"]
            rep_inputs = switch_inputs(spec, tok, texts, rep_spec["books"], start=rep_spec["start"])
            per_scale = len(inputs["clean"]) + len(inputs["pairs"]) + len(inputs["noise"])
            n_fwd = (len(ORDER) + 1) * per_scale + 2 * (len(rep_inputs["clean"]) + len(rep_inputs["pairs"]) + len(rep_inputs["noise"]))
            ledger.check(n_fwd * spec["budget"]["seconds_per_forward_estimate"])
            result = measure(s, inputs)
            print("== reproduction of C5 (set A, trained adapter, S0 and S2):", flush=True)
            reproduction = measure(s, rep_inputs, ("S0", "S2"))
            print("== control: random adapter matched in norm, at S2:", flush=True)
            norms = random_control(s, spec["control"]["seed"])
            control = measure(s, inputs, ("S2",))
            write_json(os.path.join(out_dir, "switch.json"),
                       {"finished_utc": now(), "conditions": result, "pairs": [f"{x}->{y}" for x, y in inputs["pairs"]],
                        "reproduction_of_c5": reproduction,
                        "control_random_adapter": {"S2": control["S2"], "norms": norms, "seed": spec["control"]["seed"]},
                        "books": inputs["gids"], "checks": checks,
                        "record": {"sealed_file": SEALED, "seal_sha256": seal_hash, "code_sha256": code,
                                   "platform": platform, "adapter_sha256": s.adapter_sha,
                                   "adapter_modules": s.adapter_modules, "softcap": s.softcap,
                                   "output_multiplier": s.multiplier, "switch": spec["switch"],
                                   "text_sha256": {str(g): hashlib.sha256(texts[g].encode("utf-8")).hexdigest()
                                                   for g in inputs["gids"]}}}, indent=1)
            print(f"== C6 measurement complete: {out_dir}/switch.json")
        except BaseException as e:
            if not isinstance(e, SystemExit) or "not run, as sealed" not in str(e):
                print(f"!! C6 failed; record: {write_failure(out_dir, 'run', e)}")
            raise
        finally:
            ledger.add("process exit")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
