"""Run C4 exactly as sealed in criteria/sealed/c4-glimmer-structure-and-text-v1.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c4 --phase A
    PYTHONPATH=src python3 -m dialedin.run_c4 --phase B --run-id <id printed by phase A>

Forked from run_c3.py. C4 asks two questions on set A with C1 v5's drive and measurement:
  1. Structure: does changing the local layers' sliding-window size (1024, 2048, 4096) or their RoPE
     base (0.5x, 1x, 2x) change the low-pass tilt, and at what cost in perplexity?
  2. Text: does the tilt survive when the book is replaced by its own tokens shuffled, or by one
     token repeated? If so it belongs to the model, not to the text.
Analysis: dialedin.analyze_c4.
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

from . import glimmer, metrics
from .hooks import HeadProbe, ResidualNormProbe
from .wave import WaveInjector, drive

SEALED = "criteria/sealed/c4-glimmer-structure-and-text-v1.yaml"
TAG = "c4v1"
PHASE_A_BOOK = 1342
PHASE_A_N = (2, 4096)
A = 0.1                             # the amplitude C1 v5's sealed Phase A chose; fixed here, checked in Phase A
PHASE_MAIN = 0.0
BASE_WINDOW = 2048
# name: (window, RoPE base factor, text)
CONDITIONS = {"BASE": (2048, 1.0, "book"), "W1024": (1024, 1.0, "book"), "W4096": (4096, 1.0, "book"),
              "R0.5": (2048, 0.5, "book"), "R2.0": (2048, 2.0, "book"),
              "SHUF": (2048, 1.0, "shuffled"), "CONST": (2048, 1.0, "constant")}
ORDER = ("BASE", "W1024", "W4096", "R0.5", "R2.0", "SHUF", "CONST")
PHASE_A_PASSES = 1 + len(ORDER) * (1 + 2 * len(PHASE_A_N))
PPL_FROM = metrics.WINDOW_START     # perplexity over predictions of tokens PPL_FROM+1 .. T-1
PPL_CHUNK = 1024
LOGIT_TOL = 0.005                   # perplexity-path check: max |logit difference| <= 0.5 percent of max |logit|
ROPE_THETA = 500000.0               # Glimmer's local-layer RoPE base, checked against the loaded model


def rope_theta(config) -> float:
    rp = config.rope_parameters
    return float(rp["rope_theta"] if "rope_theta" in rp else rp["sliding_attention"]["rope_theta"])


def text_ids(ids: np.ndarray, kind: str) -> np.ndarray:
    """book: unchanged. shuffled: the same tokens in an order from numpy default_rng(0).permutation.
    constant: the book's most frequent token (ties to the lowest id) at every position."""
    if kind == "book":
        return ids
    if kind == "shuffled":
        return ids[np.random.default_rng(0).permutation(len(ids))]
    if kind == "constant":
        vals, counts = np.unique(ids, return_counts=True)
        return np.full_like(ids, vals[np.argmax(counts)])
    raise ValueError(kind)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


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


def n_books(spec) -> int:
    return sum(len(v) for v in spec["inputs"]["prompt_sets"].values())


def passes_per_book(grid) -> int:
    return len(ORDER) * (2 + len(grid)) + 1       # per condition undriven, sham and one pass per N; closing sham


class Ledger:
    """Append-only wall-clock ledger: every second a process spends under one sealed file, from start to exit.

    Lives at results/ledger/, outside results/raw/, one JSON line per entry. Each entry charges
    the wall time since the previous entry, so loads, tokenization and CPU work all count.
    Pod time between processes is billed by RunPod but not counted here.
    """

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
            raise SystemExit(f"Budget cap: {self.seconds / 3600:.2f} wall hours used of {self.cap_s / 3600:.2f}, "
                             f"{upcoming / 3600:.2f} more needed. Stopping as the sealed criteria require.")


class Session:
    def __init__(self, args, ledger_root: str):
        self.repo = os.path.abspath(args.repo)
        self.seal_hash = glimmer.check_seal(self.repo, SEALED)
        self.spec = yaml.safe_load(open(os.path.join(self.repo, SEALED)))
        self.code = glimmer.check_code(self.repo, self.spec)
        self.ledger = Ledger(ledger_root, self.seal_hash, self.spec["budget"]["max_gpu_hours"] * 3600)
        self.ledger.check(0.0)
        try:
            self.texts = glimmer.load_texts(self.spec, args.manifest, os.path.dirname(args.manifest))
            glimmer.configure_determinism()
            self.model, self.decoder, self.layers, self.tok, self.platform = glimmer.load(self.spec, args.weights)
        except BaseException:
            self.ledger.add("startup failed")
            raise
        self.ledger.add("startup and model load")
        self._setup()

    def _setup(self):
        """Everything after the model load: window, grid, direction, hooks, and the structural originals."""
        self.T = self.spec["inputs"]["length_tokens"]
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["drive"]["periods_N"]
        self.layer_ids = list(range(2, len(self.layers)))      # layers 2 to 51 on Glimmer, as in C1 v5
        self.global_ids = self.platform["global_layers"]
        self.local_ids = [i for i in range(len(self.layers)) if self.layers[i].self_attn.is_local_attention]
        self.device = next(self.model.parameters()).device
        hidden = self.layers[1].self_attn.o_proj.out_features
        g = torch.Generator().manual_seed(0)
        d = torch.randn(hidden, generator=g, dtype=torch.float32)
        d = d / d.norm()
        self.direction_sha = sha_array(d.numpy())
        self.direction = d.to(self.device)
        self.injector = WaveInjector(self.layers[1], self.direction)
        self.norms = ResidualNormProbe(self.layers[1])
        self.probe = HeadProbe(self.layers, self.layer_ids, n_heads=32, t1=self.t1)
        self.rotary = self.decoder.rotary_emb
        self.inv_freq0 = self.rotary.inv_freq.detach().clone()
        self.theta0 = rope_theta(self.decoder.config)
        if self.theta0 != ROPE_THETA:
            raise SystemExit(f"RoPE base {self.theta0} differs from the sealed {ROPE_THETA}")
        dim = 2 * self.inv_freq0.numel()
        expect = 1.0 / (self.theta0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        if not torch.allclose(self.inv_freq0.float().cpu(), expect, rtol=1e-6, atol=0):
            raise SystemExit("the loaded inv_freq does not match 1 / base ** (arange(0, dim, 2) / dim)")
        assert self.decoder.config.sliding_window == BASE_WINDOW
        assert all(self.layers[i].self_attn.sliding_window == BASE_WINDOW for i in self.local_ids)
        self.head = self.model.get_output_embeddings()
        self.softcap = getattr(self.decoder.config, "final_logit_softcapping", None)
        self.multiplier = getattr(self.decoder.config, "output_multiplier", None)
        full_text = getattr(getattr(self.model, "config", None), "text_config", None)
        if full_text is not None:                          # the real model: both must be set and agree with it
            if self.softcap is None or self.multiplier is None:
                raise SystemExit("final_logit_softcapping or output_multiplier missing from the text config")
            if (self.softcap, self.multiplier) != (full_text.final_logit_softcapping, full_text.output_multiplier):
                raise SystemExit("softcap or multiplier differs between the decoder config and the full model's text_config")
        self.keep_hidden = False
        self.last_hidden = None

    def set_structure(self, window: int, rope_factor: float):
        """Sets the local layers' sliding window (config, read when masks are built, and each local layer's own
        copy) and the RoPE base (inv_freq recomputed from base x factor; the original tensor restored at 1x)."""
        self.decoder.config.sliding_window = window
        for i in self.local_ids:
            self.layers[i].self_attn.sliding_window = window
        if rope_factor == 1.0:
            self.rotary.inv_freq.copy_(self.inv_freq0)
        else:
            dim = 2 * self.inv_freq0.numel()
            base = self.theta0 * rope_factor
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            self.rotary.inv_freq.copy_(inv.to(self.rotary.inv_freq.device, self.rotary.inv_freq.dtype))

    def set_condition(self, cond):
        window, factor, _ = CONDITIONS[cond]
        self.set_structure(window, factor)

    def window_check(self, window: int) -> dict:
        """Confirms the window took effect: on a 4,096-token input, changes one token at position 100 and
        records whether layer 0's head outputs at positions 1,600 and 3,100 change. A causal window W lets a
        token reach positions up to W - 1 later: 1,500 apart needs W > 1500, 3,000 apart needs W > 3000."""
        self.probe.mode = None                            # setup checks never record or measure
        n, p = 4096, 100
        base = np.asarray(self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:n])
        alt = base.copy(); alt[p] = base[p] + 1 if base[p] < 999 else base[p] - 1     # a different ordinary token
        seen = []
        h = self.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0][0].detach().clone()))
        try:
            with torch.inference_mode():
                for x in (base, alt):
                    self.decoder(input_ids=torch.tensor([x], device=self.device), use_cache=False)
        finally:
            h.remove()
        a, b = seen
        reach = {d: bool((a[p + d] != b[p + d]).any()) for d in (1500, 3000)}
        want = {1500: window > 1500, 3000: window > 3000}
        return {"window": window, "reaches": {str(k): v for k, v in reach.items()},
                "as_expected": reach == want}

    def rope_check(self, factor: float) -> dict:
        """Confirms the RoPE change took effect: layer 0's head outputs on the first 2,048 tokens of book 1342
        differ between the factor and 1x, and 1x restores them bit for bit."""
        self.probe.mode = None                            # setup checks never record or measure
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:2048]],
                         device=self.device)
        seen = []
        h = self.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0][0].detach().clone()))
        try:
            with torch.inference_mode():
                for f in (1.0, factor, 1.0):
                    self.set_structure(BASE_WINDOW, f)
                    self.decoder(input_ids=x, use_cache=False)
        finally:
            h.remove()
            self.set_structure(BASE_WINDOW, 1.0)
        changed, restored = bool((seen[0] != seen[1]).any()), bool(torch.equal(seen[0], seen[2]))
        return {"factor": factor, "changed": changed, "restored": restored, "as_expected": changed and restored}

    def logits(self, h):
        """Glimmer's output path, as in MuseGlimmerForConditionalGeneration.forward: the head in the model's dtype,
        then times output_multiplier, then T * tanh(x / T) with T = final_logit_softcapping, all in the model's dtype; cast to float32 at the end."""
        x = self.head(h)
        if self.multiplier is not None:
            x = x * self.multiplier
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x.float()

    def logits_check(self) -> dict:
        """Confirms the perplexity path: on the first 2,048 tokens of book 1342, the head applied to the text
        model's last hidden state (multiplier, then softcap) must match the full model's own logits."""
        self.probe.mode = None                            # setup checks never record or measure
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:2048]],
                         device=self.device)
        with torch.inference_mode():
            ref = self.model(input_ids=x, use_cache=False).logits[0].float()
            h = self.decoder(input_ids=x, use_cache=False).last_hidden_state[0]
            ours = self.logits(h)
        diff = float((ours - ref).abs().max())
        scale = float(ref.abs().max())
        return {"max_abs_diff": diff, "max_abs_logit": scale, "tolerance": LOGIT_TOL * scale,
                "as_expected": bool(diff <= LOGIT_TOL * scale)}

    def ids(self, gid, kind="book"):
        raw = np.asarray(self.tok(self.texts[gid], add_special_tokens=False)["input_ids"][: self.T], dtype=np.int64)
        assert len(raw) == self.T
        arr = text_ids(raw, kind)
        return torch.tensor(arr[None, :], device=self.device), sha_array(arr.astype(np.int32))

    def forward(self, ids, what: str):
        assert ids.shape[0] == 1, "batch size must be 1"
        self.ledger.check()
        self.ledger.add(f"before {what}")
        with torch.inference_mode():
            out = self.decoder(input_ids=ids, use_cache=False)
            if self.keep_hidden:
                self.last_hidden = out.last_hidden_state[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ledger.last_pass = self.ledger.add(what)

    def perplexity(self, ids) -> float:
        """exp of the mean next-token loss over predictions of tokens PPL_FROM+1 .. T-1, from the last undriven pass."""
        h, tgt = self.last_hidden, ids[0]
        total, count = 0.0, 0
        with torch.inference_mode():
            for s in range(PPL_FROM, self.T - 1, PPL_CHUNK):
                e = min(s + PPL_CHUNK, self.T - 1)
                lp = self.logits(h[s:e]).log_softmax(-1)
                total -= float(lp.gather(1, tgt[s + 1:e + 1, None]).sum())
                count += e - s
        self.last_hidden = None
        return float(np.exp(total / count))

    def baseline(self, ids):
        """Undriven pass: records head outputs, the amplitude reference and the perplexity."""
        self.injector.disarm()
        self.probe.mode = "record"
        self.norms.armed = True
        self.keep_hidden = True
        try:
            self.forward(ids, "undriven")
        finally:
            self.norms.armed = False
            self.keep_hidden = False
        return float(np.median(self.norms.norms)), self.perplexity(ids)

    def measure(self, ids, N, amplitude, phase, what):
        """One driven pass. The input component uses the drive as stored by the model dtype."""
        cos_b, sin_b = self.probe.set_frequencies(N, self.T, self.device)
        w = drive(N, self.T, amplitude, phase)
        self.injector.arm(w)
        self.probe.mode = "measure"
        try:
            self.forward(ids, what)
        finally:
            self.injector.disarm()
        out = self.probe.collect()
        out["input_mag"] = float(metrics.component(self.injector.realized[self.t1:], cos_b[:, :1], sin_b[:, :1])[0])
        out["input_mag_nominal"] = float(metrics.component(w[self.t1:], cos_b[:, :1], sin_b[:, :1])[0])
        out["realized"] = dict(self.injector.stats, realized_over_nominal=out["input_mag"] / out["input_mag_nominal"])
        return out

    def sham_differences(self, ids) -> int:
        """Number of head-output values, over all tokens and layers, whose bits differ from the undriven run."""
        self.injector.arm(np.zeros(self.T))
        self.probe.mode = "zero_check"
        try:
            self.forward(ids, "sham")
        finally:
            self.injector.disarm()
        total = int(sum(self.probe.result.values()))
        self.probe.result = {}
        return total

    def record(self, gid, token_sha):
        return {"conditions": {k: {"window": v[0], "rope_factor": v[1], "text": v[2]} for k, v in CONDITIONS.items()},
                "rope_theta_base": self.theta0, "softcap": self.softcap, "output_multiplier": self.multiplier,
                "sealed_file": SEALED, "seal_sha256": self.seal_hash, "code_sha256": self.code,
                "platform": self.platform, "gutenberg_id": gid,
                "text_sha256": hashlib.sha256(self.texts[gid].encode("utf-8")).hexdigest(),
                "token_ids_sha256": token_sha, "tokenizer": type(self.tok).__name__,
                "direction_seed": 0, "direction_sha256": self.direction_sha,
                "drive": {"form": "amplitude * cos(2 pi t / N + phase) at every token t", "phase_main": PHASE_MAIN},
                "window": [self.t1, self.T], "T": self.T, "layers_measured": self.layer_ids}


def expected_books(spec) -> list:
    return [f"book_{set_name}_{int(gid)}.json" for set_name, books in spec["inputs"]["prompt_sets"].items()
            for gid in books]


def run_state(run_dir: str, spec) -> str:
    """complete: every book file present. stopped: Phase A ended C4 by rule (not linear or underpowered).
    failed: a failure record and not complete. open: anything else (waiting for Phase B, running, or killed)."""
    files = set(os.listdir(run_dir))
    if all(b in files for b in expected_books(spec)):
        return "complete"
    if "phaseA.json" in files:
        pa = json.load(open(os.path.join(run_dir, "phaseA.json")))
        if pa.get("stopped"):
            return "stopped"
    if any(".FAILED." in f for f in files):
        return "failed"
    return "open"


def runs_under_seal(raw_root: str, seal_hash: str, spec) -> list:
    """Every run started under this seal (found by STARTED.json, written before any GPU work), oldest first."""
    runs = []
    for p in glob.glob(os.path.join(raw_root, "*", "STARTED.json")):
        d = json.load(open(p))
        if d.get("seal_sha256") != seal_hash:
            continue
        run_dir = os.path.dirname(p)
        runs.append({"run_id": os.path.basename(run_dir), "started_utc": d.get("utc"),
                     "restart_reason": d.get("restart_reason"), "state": run_state(run_dir, spec)})
    return sorted(runs, key=lambda r: (r["started_utc"] or "", r["run_id"]))


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


def write_failure(out_dir: str, name: str, err: BaseException):
    path = os.path.join(out_dir, f"{name}.FAILED.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
    write_json(path, {"utc": now(), "what": name, "error": repr(err), "traceback": traceback.format_exc()}, indent=1)
    return path


def cell(out) -> dict:
    return {"realized_drive": out["realized"], "input_mag": out["input_mag"], "input_mag_nominal": out["input_mag_nominal"],
            "target": out["target"].astype(np.float32).tolist(), "floor": out["floor"].astype(np.float32).tolist(),
            "bin_power": out["bin_power"].tolist(), "n_below": out["n_below"]}


def phase_a(s: Session, out_dir: str):
    """Window checks, then linearity and power under every condition, on book 1342, at the fixed a, N = 2 and 4096."""
    remaining = (PHASE_A_PASSES + n_books(s.spec) * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    setup_failed = None
    try:
        checks = []
        for w in (1024, 2048, 4096):
            s.set_structure(w, 1.0)
            checks.append(s.window_check(w))
            print(f"   window {w}: reaches {checks[-1]['reaches']}  as expected: {checks[-1]['as_expected']}", flush=True)
        s.set_condition("BASE")
        rope_checks = [s.rope_check(f) for f in (0.5, 2.0)]
        for c in rope_checks:
            print(f"   RoPE x{c['factor']}: changed {c['changed']}, restored {c['restored']}", flush=True)
        logit_check = s.logits_check()
        print(f"   perplexity path: max |logit difference| {logit_check['max_abs_diff']:.4g} "
              f"(max |logit| {logit_check['max_abs_logit']:.4g}); as expected: {logit_check['as_expected']}", flush=True)
        failed_checks = ([f"window {c['window']}" for c in checks if not c["as_expected"]]
                         + [f"RoPE x{c['factor']}" for c in rope_checks if not c["as_expected"]]
                         + (["perplexity path"] if not logit_check["as_expected"] else []))
        if failed_checks:
            result = {"phase": "A", "finished_utc": now(), "window_checks": checks, "rope_checks": rope_checks,
                      "logits_check": logit_check, "stopped": f"setup checks failed: {failed_checks}",
                      "outcome_if_stopped": "U", "sham_differences": None, "rows": [], "chosen_a": None,
                      "record": s.record(PHASE_A_BOOK, None)}
            write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
            setup_failed = failed_checks
            raise SystemExit(f"C4 is not run, as sealed: setup checks failed: {failed_checks}. Outcome U (inconclusive).")
        rows, sham = [], None
        for cond in ORDER:
            s.set_condition(cond)
            ids, token_sha = s.ids(PHASE_A_BOOK, CONDITIONS[cond][2])
            if cond == "BASE":
                base_sha = token_sha
            m, ppl = s.baseline(ids)
            if sham is None:
                sham = s.sham_differences(ids)
                if sham != 0:
                    raise RuntimeError(f"Phase A sham differed in {sham} values. Determinism or hooks are broken.")
            row = {"condition": cond, "median_norm": m, "perplexity": ppl, "by_N": {}}
            for N in PHASE_A_N:
                resp, snr = [], None
                for scale in (1, 2):
                    out = s.measure(ids, N, A * scale * m, PHASE_MAIN, f"phaseA {cond} x{scale} N={N}")
                    resp.append(float(out["target"].mean()))
                    if scale == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   {cond:<6} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["linear"] = all(v["within_10pct"] for v in row["by_N"].values())
            row["powered"] = bool(row["by_N"][max(PHASE_A_N)]["snr_at_a"] >= 3)
            rows.append(row)
        s.set_condition("BASE")
    except BaseException as e:
        if setup_failed is None:                           # a stop by rule has its phaseA.json; no failure record
            print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    bad = [r["condition"] for r in rows if not (r["linear"] and r["powered"])]
    stopped = f"not linear or underpowered under {bad}" if bad else None
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    result = {"phase": "A", "finished_utc": now(), "wall_seconds_per_pass": wall_per_pass,
              "book": PHASE_A_BOOK, "a": A, "window_checks": checks, "rope_checks": rope_checks,
              "logits_check": logit_check, "stopped": stopped,
              "outcome_if_stopped": "U" if stopped else None, "sham_differences": sham, "rows": rows,
              "chosen_a": None if stopped else A,
              "seconds_per_pass_last": s.ledger.last_pass, "record": s.record(PHASE_A_BOOK, base_sha)}
    write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
    projected = n_books(s.spec) * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}")
    if stopped:
        raise SystemExit(f"C4 is not run, as sealed: {stopped}. Outcome U (inconclusive).")
    print(f"== Phase A passed under all {len(ORDER)} conditions at a = {A}; window, RoPE and perplexity checks as expected")


def check_phase_a(s: Session, out_dir: str) -> dict:
    """Phase B uses Phase A only if it ran under the same seal, code and platform with a clean sham."""
    pa = json.load(open(os.path.join(out_dir, "phaseA.json")))
    rec = pa["record"]
    problems = []
    if rec["seal_sha256"] != s.seal_hash:
        problems.append("seal hash differs")
    if rec["code_sha256"] != s.code:
        problems.append("code hashes differ")
    if rec["platform"] != s.platform:
        problems.append(f"platform differs: {rec['platform']} vs {s.platform}")
    if pa["sham_differences"] != 0:
        problems.append("Phase A sham was not clean")
    if pa.get("stopped"):
        problems.append(f"Phase A stopped C4 by rule ({pa['stopped']}), so C4 is not run, as sealed")
    if problems:
        raise SystemExit("Phase A check failed: " + "; ".join(problems))
    return pa


def phase_b(s: Session, out_dir: str, restart_reason: str = None):
    pa = check_phase_a(s, out_dir)
    a = pa["chosen_a"]
    todo = []
    for set_name, books in s.spec["inputs"]["prompt_sets"].items():
        for gid in books:
            path = os.path.join(out_dir, f"book_{set_name}_{int(gid)}.json")
            if os.path.exists(path):
                print(f"   skip {set_name} {gid} (done)")
                continue
            failed = (glob.glob(os.path.join(out_dir, f"book_{set_name}_{int(gid)}.FAILED.*.json"))
                      + glob.glob(os.path.join(out_dir, f"book_{set_name}_{int(gid)}.ATTEMPT.*.json")))
            if failed and not restart_reason:
                raise SystemExit(f"Book {gid} was attempted before ({failed}). Retrying it needs --restart-reason; "
                                 f"those records are kept and reported.")
            todo.append((set_name, int(gid), path, bool(failed)))
    per_pass = max(pa["wall_seconds_per_pass"], s.ledger.last_pass)
    s.ledger.check(len(todo) * passes_per_book(s.grid) * per_pass)
    for set_name, gid, path, retried in todo:
        t_book = time.time()
        write_json(os.path.join(out_dir, f"book_{set_name}_{gid}.ATTEMPT."
                                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"),
                   {"utc": now(), "restart_reason": restart_reason if retried else None})
        try:
            per_cond, shams, refs, ppls, token_shas = {}, {}, {}, {}, {}
            for cond in ORDER:
                s.set_condition(cond)
                ids, token_sha = s.ids(gid, CONDITIONS[cond][2])
                token_shas[cond] = token_sha
                m, ppl = s.baseline(ids)
                refs[cond], ppls[cond] = m, ppl
                sh = s.sham_differences(ids)
                if sh != 0:
                    raise RuntimeError(f"Sham under {cond} on book {gid} differed in {sh} values.")
                shams[cond] = sh
                per_cond[cond] = {N: cell(s.measure(ids, N, a * m, PHASE_MAIN, f"{gid} {cond} N={N}")) for N in s.grid}
            sham_close = s.sham_differences(ids)
            if sham_close != 0:
                raise RuntimeError(f"Closing sham on book {gid} differed in {sham_close} values; "
                                   f"results for this book are not written.")
            s.set_condition("BASE")
        except BaseException as e:
            print(f"!! Book {gid} failed; record: {write_failure(out_dir, f'book_{set_name}_{gid}', e)}")
            raise
        rec = {"set": set_name, "gutenberg_id": gid, "a": a, "amplitude_reference": refs, "perplexity": ppls,
               "token_ids_sha256": token_shas, "sham_differences": shams, "sham_differences_close": sham_close,
               "seconds": round(time.time() - t_book, 1), "finished_utc": now(),
               "restart_reason": restart_reason if retried else None, "global_layers": s.global_ids,
               "per_condition": per_cond, "record": s.record(gid, token_shas["BASE"])}
        write_json(path, rec)
        print(f"   done {set_name} {gid} in {rec['seconds']} s; {s.ledger.seconds / 3600:.2f} hours used", flush=True)


def admit(phase: str, raw_root: str, out_dir: str, seal_hash: str, spec, restart_reason: str = None) -> list:
    """Run-selection rules from the sealed spec. Raises SystemExit when a run may not proceed. Returns earlier runs.

    Phase A: refused if this run id already started, if any run under the seal is complete or was
    stopped by rule, or if any run exists and no written reason is given. Writes STARTED.json.
    Phase B: needs this run's STARTED.json under the same seal and its phaseA.json; refused if any
    other run under the seal is complete or if a newer run under the seal has started.
    """
    runs = runs_under_seal(raw_root, seal_hash, spec) if os.path.isdir(raw_root) else []
    me = os.path.basename(out_dir)
    others = [r for r in runs if r["run_id"] != me]
    if phase == "A":
        if os.path.exists(os.path.join(out_dir, "STARTED.json")):
            raise SystemExit("This run id already started; raw results are never overwritten.")
        final = [r["run_id"] for r in others if r["state"] in ("complete", "stopped")]
        if final:
            raise SystemExit(f"A run under this seal is complete or was stopped by rule: {final}. No new run.")
        if others and not restart_reason:
            raise SystemExit(f"Runs already started under this seal: {others}. A new run is allowed only "
                             f"after an infrastructure failure; pass --restart-reason and keep the old run.")
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, "STARTED.json"),
                   {"utc": now(), "seal_sha256": seal_hash, "restart_reason": restart_reason,
                    "earlier_runs": others}, indent=1)
        return others
    started = os.path.join(out_dir, "STARTED.json")
    if not os.path.exists(started) or json.load(open(started)).get("seal_sha256") != seal_hash:
        raise SystemExit(f"{out_dir} has no STARTED.json under this seal.")
    if not os.path.exists(os.path.join(out_dir, "phaseA.json")):
        raise SystemExit(f"No phaseA.json in {out_dir}.")
    done = [r["run_id"] for r in others if r["state"] == "complete"]
    if done:
        raise SystemExit(f"Another run under this seal is complete: {done}.")
    mine = next(r for r in runs if r["run_id"] == me)
    newer = [r["run_id"] for r in others if (r["started_utc"] or "", r["run_id"]) > (mine["started_utc"] or "", me)]
    if newer:
        raise SystemExit(f"A newer run under this seal has started: {newer}. Only the newest run may continue.")
    return others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--restart-reason", help="required for a new run under a seal that already has one, "
                                             "or to retry a book with an earlier attempt")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    raw_root = os.path.join(repo, "results", "raw")
    ledger_root = os.path.join(repo, "results", "ledger")
    if args.phase == "B" and not args.run_id:
        raise SystemExit("Phase B needs --run-id from phase A.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{TAG}-%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(raw_root, run_id)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    lock = RunLock(ledger_root, seal_hash)
    try:
        admit(args.phase, raw_root, out_dir, seal_hash, spec, args.restart_reason)
        print(f"== Run {run_id}, phase {args.phase}")
        try:
            s = Session(args, ledger_root)
        except BaseException as e:
            print(f"!! Startup failed; record: {write_failure(out_dir, 'startup', e)}")
            raise
        try:
            if args.phase == "A":
                phase_a(s, out_dir)
            else:
                phase_b(s, out_dir, args.restart_reason)
        finally:
            s.ledger.add("process exit")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
