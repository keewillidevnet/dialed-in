"""Run C8 exactly as sealed in criteria/sealed/c8-glimmer-one-matrix-dial-v1.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c8                                  # a new run: Phase A, then Phase B
    PYTHONPATH=src python3 -m dialedin.run_c8 --run-id <id> --restart-reason "<why>"   # continue an interrupted run
    PYTHONPATH=src python3 -m dialedin.run_c8 --smoke --spec criteria/drafts/c8-glimmer-one-matrix-dial-v1.yaml

Forked from run_c7.py. C8 was proposed by Muse Wonder (results/agent/proposal-20261029T193320Z-muse-wonder-1.0.json)
and revised in review. It asks whether one of the C5 adapter's 32 matrices, layer 2's v_proj, carries a majority of
the dial's tilt change, while the same matrix in layer 4 does not. The frozen C5 adapter is measured on set C, first
32,768 tokens, under six conditions read from the sealed spec (names and scales fixed here):
  BASE scale 0; FULL scale 2, all 32 modules; QKV2 layer 2 q, k, v; V2 layer 2 v; QK2 layer 2 q, k; V4 layer 4 v.
Modules not listed for a condition have B set to zero. Measurement as C5's and C7's Phase B. Analysis: analyze_c8.
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
from . import pilot_c5 as P
from .hooks import HeadProbe, ResidualNormProbe
from .wave import WaveInjector, drive

SEALED = "criteria/sealed/c8-glimmer-one-matrix-dial-v1.yaml"
TAG = "c8v1"
PHASE_A_BOOK = 74
PHASE_A_N = (2, 4096)
A = 0.1                             # C1 v5's sealed amplitude, as C5; checked in Phase A under every condition
PHASE_MAIN = 0.0
SCALE = 2.0
ORDER = ("BASE", "FULL", "QKV2", "V2", "QK2", "V4")
SCALES = {"BASE": 0.0, "FULL": SCALE, "QKV2": SCALE, "V2": SCALE, "QK2": SCALE, "V4": SCALE}
PHASE_A_PASSES = 1 + len(ORDER) * (1 + 2 * len(PHASE_A_N))
PPL_FROM = metrics.WINDOW_START
PPL_CHUNK = 1024
LOGIT_TOL = 0.005
CHECK_LAYER = 10


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def condition_modules(spec, names) -> dict:
    """Which adapter modules are on under each condition: "all", or a list from the sealed spec that must name
    adapter modules, without repeats. BASE runs at scale 0, so its set does not change the model."""
    out = {}
    for c in ORDER:
        mods = spec["conditions"][c]["modules"]
        if mods == "all":
            out[c] = list(names)
            continue
        missing = [m for m in mods if m not in names]
        if missing or not mods or len(set(mods)) != len(mods):
            raise SystemExit(f"{c} modules not all in the adapter (or empty or repeated): {missing}")
        out[c] = [n for n in names if n in mods]
        if out[c] != list(mods):            # the analysis compares records to the sealed list in order
            raise SystemExit(f"{c} modules must be listed in adapter order: {out[c]}")
    if len({tuple(v) for c, v in out.items() if c != "BASE"}) != len(ORDER) - 1:
        raise SystemExit("two conditions switch on the same modules")
    return out


PLACEHOLDERS = ("FILL_AT_SEAL", "SMOKE_RESULT")


def check_no_placeholders(obj, where="spec"):
    """A sealed spec must hold no placeholder anywhere (C6 v1 was sealed with one and could not run)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_placeholders(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_placeholders(v, f"{where}[{i}]")
    elif isinstance(obj, str) and any(p in obj for p in PLACEHOLDERS):
        raise SystemExit(f"{where} still holds a placeholder ({obj.strip()[:40]}); the sealed spec is not runnable")


def check_condition_spec(spec):
    named = sorted(k for k, v in spec["conditions"].items() if isinstance(v, dict) and "scale" in v)
    if named != sorted(ORDER):
        raise SystemExit(f"conditions in the spec {named} differ from the code {sorted(ORDER)}")
    sc = {c: float(spec["conditions"][c]["scale"]) for c in ORDER}
    if sc != SCALES:
        raise SystemExit(f"condition scales in the spec differ from the code: {sc}")


class Ledger:
    """Append-only wall-clock ledger under results/ledger/ (as C1 to C6)."""

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


class NullLedger:
    seconds, last_pass = 0.0, 0.0

    def add(self, what):
        return 0.0

    def check(self, upcoming_s=None):
        return None


class Session:
    """Glimmer with the frozen C5 adapter (hash checked) and C5's measurement hooks. Conditions switch the scale and
    which adapter modules are on (an off module has its B factor set to zero; switching back restores the saved B)."""

    def __init__(self, spec, seal_hash, code, ledger, args, allow_unfilled=False):
        self.repo = os.path.abspath(args.repo)
        self.spec, self.seal_hash, self.code, self.ledger = spec, seal_hash, code, ledger
        try:
            self.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
            glimmer.configure_determinism()
            self.model, self.decoder, self.layers, self.tok, self.platform = glimmer.load(spec, args.weights)
        except BaseException:
            self.ledger.add("startup failed")
            raise
        self.ledger.add("startup and model load")
        self._setup(allow_unfilled)

    def _reference_input(self) -> torch.Tensor:
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:2048]],
                         device=self.device)
        seen = []
        h = self.layers[CHECK_LAYER].self_attn.o_proj.register_forward_pre_hook(
            lambda m, a: seen.append(a[0][0].detach().clone()))
        try:
            with torch.inference_mode():
                self.decoder(input_ids=x, use_cache=False)
        finally:
            h.remove()
        return seen[0]

    def _setup(self, allow_unfilled=False):
        """Reference before the adapter, then the adapter (hash checked), then conditions, window, grid, hooks."""
        check_condition_spec(self.spec)
        self.T = self.spec["inputs"]["length_tokens"]
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["drive"]["periods_N"]
        self.layer_ids = list(range(2, len(self.layers)))
        self.global_ids = self.platform["global_layers"]
        self.device = next(self.model.parameters()).device
        self.unadapted_ref = self._reference_input()
        ad_spec = self.spec["adapter"]
        path = os.path.join(self.repo, ad_spec["file"])
        self.adapter_sha = sha_file(path)
        if allow_unfilled and ad_spec["sha256"] == "FILL_AT_SEAL":
            print(f"== smoke: the adapter's sha256 is {self.adapter_sha}; fill it into the spec before sealing")
        elif self.adapter_sha != ad_spec["sha256"]:
            raise SystemExit(f"adapter {ad_spec['file']} sha256 {self.adapter_sha} differs from the sealed {ad_spec['sha256']}")
        ad = torch.load(path, map_location="cpu")
        names, params = P.attach_lora(self.layers, ad_spec["layers"], ad_spec["rank"], ad_spec["alpha"],
                                      family=ad_spec["family"])
        if names != ad["modules"]:
            raise SystemExit("the adapter's modules differ from the sealed family and layers")
        with torch.no_grad():
            for (a, b), pa, pb in zip(zip(ad["A"], ad["B"]), params[0::2], params[1::2]):
                pa.copy_(a.to(pa.device)); pb.copy_(b.to(pb.device))
        for p in params:
            p.requires_grad_(False)
        self.adapter_modules = names
        self.params = params
        self.saved_B = [b.detach().clone() for b in params[1::2]]
        self.on = condition_modules(self.spec, names)
        self.condition = None
        P.LoRALinear.scale = 0.0
        hidden = self.decoder.config.hidden_size
        g = torch.Generator().manual_seed(0)
        d = torch.randn(hidden, generator=g, dtype=torch.float32)
        d = d / d.norm()
        self.direction_sha = sha_array(d.numpy())
        self.direction = d.to(self.device)
        self.injector = WaveInjector(self.layers[1], self.direction)
        self.norms = ResidualNormProbe(self.layers[1])
        self.probe = HeadProbe(self.layers, self.layer_ids, n_heads=32, t1=self.t1)
        self.head = self.model.get_output_embeddings()
        self.softcap = getattr(self.decoder.config, "final_logit_softcapping", None)
        self.multiplier = getattr(self.decoder.config, "output_multiplier", None)
        full_text = getattr(getattr(self.model, "config", None), "text_config", None)
        if full_text is not None:
            if self.softcap is None or self.multiplier is None:
                raise SystemExit("final_logit_softcapping or output_multiplier missing from the text config")
            if (self.softcap, self.multiplier) != (full_text.final_logit_softcapping, full_text.output_multiplier):
                raise SystemExit("softcap or multiplier differs between the decoder config and the full model's text_config")
        self.keep_hidden = False
        self.last_hidden = None

    def set_condition(self, cond):
        on = set(self.on[cond])
        with torch.no_grad():
            for n, b, s in zip(self.adapter_modules, self.params[1::2], self.saved_B):
                if n in on:
                    b.copy_(s)
                else:
                    b.zero_()
        P.LoRALinear.scale = float(self.spec["conditions"][cond]["scale"])
        self.condition = cond

    def adapter_checks(self) -> dict:
        """BASE must reproduce the unadapted model bit for bit (layer 10 o_proj input, 2,048 tokens of book 74); every
        other condition must change it and differ from every other condition; FULL again after the masked conditions
        must reproduce the first FULL bit for bit, and BASE again the first BASE."""
        self.probe.mode = None
        x = {}
        try:
            for cond in ORDER:
                self.set_condition(cond)
                x[cond] = self._reference_input()
            self.set_condition("FULL"); full_again = self._reference_input()
            self.set_condition("BASE"); base_again = self._reference_input()
        finally:
            self.set_condition("BASE")
        others = [c for c in ORDER if c != "BASE"]
        out = {"base_equals_unadapted": bool(torch.equal(x["BASE"], self.unadapted_ref))}
        out.update({f"{c.lower()}_changes": bool((x[c] != x["BASE"]).any()) for c in others})
        out["all_conditions_distinct"] = all(bool((x[a] != x[b]).any()) for i, a in enumerate(others) for b in others[i + 1:])
        out["full_restored"] = bool(torch.equal(full_again, x["FULL"]))
        out["base_restored"] = bool(torch.equal(base_again, x["BASE"]))
        out["as_expected"] = all(out.values())
        out["modules_on"] = {c: len(v) for c, v in self.on.items()}
        return out

    def logits(self, h):
        x = self.head(h)
        if self.multiplier is not None:
            x = x * self.multiplier
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x.float()

    def logits_check(self) -> dict:
        self.probe.mode = None
        self.set_condition("BASE")
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:2048]],
                         device=self.device)
        with torch.inference_mode():
            ref = self.model(input_ids=x, use_cache=False).logits[0].float()
            ours = self.logits(self.decoder(input_ids=x, use_cache=False).last_hidden_state[0])
        diff, scale = float((ours - ref).abs().max()), float(ref.abs().max())
        return {"max_abs_diff": diff, "max_abs_logit": scale, "tolerance": LOGIT_TOL * scale,
                "as_expected": bool(diff <= LOGIT_TOL * scale)}

    def ids(self, gid):
        raw = np.asarray(self.tok(self.texts[gid], add_special_tokens=False)["input_ids"][: self.T], dtype=np.int64)
        assert len(raw) == self.T
        return torch.tensor(raw[None, :], device=self.device), sha_array(raw.astype(np.int32))

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
        return {"conditions": {c: {"scale": float(self.spec["conditions"][c]["scale"]), "modules_on": self.on[c]}
                               for c in ORDER},
                "adapter_sha256": self.adapter_sha, "adapter_modules": self.adapter_modules,
                "softcap": self.softcap, "output_multiplier": self.multiplier, "sealed_file": SEALED,
                "seal_sha256": self.seal_hash, "code_sha256": self.code, "platform": self.platform,
                "gutenberg_id": gid,
                "text_sha256": hashlib.sha256(self.texts[gid].encode("utf-8")).hexdigest() if gid is not None else None,
                "token_ids_sha256": token_sha, "tokenizer": type(self.tok).__name__,
                "direction_seed": 0, "direction_sha256": self.direction_sha,
                "drive": {"form": "amplitude * cos(2 pi t / N + phase) at every token t", "phase_main": PHASE_MAIN},
                "window": [self.t1, self.T], "T": self.T, "layers_measured": self.layer_ids}


def expected_books(spec) -> list:
    return [f"book_{set_name}_{int(gid)}.json" for set_name, books in spec["inputs"]["prompt_sets"].items()
            for gid in books]


def run_state(run_dir: str, spec) -> str:
    files = set(os.listdir(run_dir))
    if all(b in files for b in expected_books(spec)):
        return "complete"
    if "phaseA.json" in files and json.load(open(os.path.join(run_dir, "phaseA.json"))).get("stopped"):
        return "stopped"
    if any(".FAILED." in f for f in files):
        return "failed"
    return "open"


def runs_under_seal(raw_root: str, seal_hash: str, spec) -> list:
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


def phase_a(s: Session, out_dir: str, restart_reason: str = None):
    """Setup checks, then linearity and power under every condition on book 74 at a = 0.1, N = 2 and 4096."""
    remaining = (PHASE_A_PASSES + n_books(s.spec) * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    setup_failed = None
    try:
        ad_check = s.adapter_checks()
        print(f"   adapter and condition checks: {ad_check}", flush=True)
        logit_check = s.logits_check()
        print(f"   perplexity path: max |logit difference| {logit_check['max_abs_diff']:.4g} "
              f"(max |logit| {logit_check['max_abs_logit']:.4g}); as expected: {logit_check['as_expected']}", flush=True)
        failed_checks = ((["adapter"] if not ad_check["as_expected"] else [])
                         + (["perplexity path"] if not logit_check["as_expected"] else []))
        if failed_checks:
            write_json(os.path.join(out_dir, "phaseA.json"),
                       {"phase": "A", "finished_utc": now(), "restart_reason": restart_reason,
                        "adapter_checks": ad_check, "logits_check": logit_check,
                        "stopped": f"setup checks failed: {failed_checks}", "outcome_if_stopped": "U",
                        "sham_differences": None, "rows": [], "chosen_a": None, "record": s.record(PHASE_A_BOOK, None)},
                       indent=2)
            setup_failed = failed_checks
            raise SystemExit(f"C8 is not run, as sealed: setup checks failed: {failed_checks}. Outcome U.")
        rows, sham = [], None
        ids, base_sha = s.ids(PHASE_A_BOOK)
        for cond in ORDER:
            s.set_condition(cond)
            m, ppl = s.baseline(ids)
            if sham is None:
                sham = s.sham_differences(ids)
                if sham != 0:
                    raise RuntimeError(f"Phase A sham differed in {sham} values. Determinism or hooks are broken.")
            row = {"condition": cond, "median_norm": m, "perplexity": ppl, "by_N": {}}
            for N in PHASE_A_N:
                resp, snr = [], None
                for k in (1, 2):
                    out = s.measure(ids, N, A * k * m, PHASE_MAIN, f"phaseA {cond} x{k} N={N}")
                    resp.append(float(out["target"].mean()))
                    if k == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   {cond:<5} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["linear"] = all(v["within_10pct"] for v in row["by_N"].values())
            row["powered"] = bool(row["by_N"][max(PHASE_A_N)]["snr_at_a"] >= 3)
            rows.append(row)
        s.set_condition("BASE")
    except BaseException as e:
        if setup_failed is None:
            print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    bad = [r["condition"] for r in rows if not (r["linear"] and r["powered"])]
    stopped = f"not linear or underpowered under {bad}" if bad else None
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    write_json(os.path.join(out_dir, "phaseA.json"),
               {"phase": "A", "finished_utc": now(), "restart_reason": restart_reason,
                "earlier_failures": sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.FAILED.*.json"))),
                "wall_seconds_per_pass": wall_per_pass, "book": PHASE_A_BOOK,
                "a": A, "adapter_checks": ad_check, "logits_check": logit_check, "stopped": stopped,
                "outcome_if_stopped": "U" if stopped else None, "sham_differences": sham, "rows": rows,
                "chosen_a": None if stopped else A, "seconds_per_pass_last": s.ledger.last_pass,
                "record": s.record(PHASE_A_BOOK, base_sha)}, indent=2)
    projected = n_books(s.spec) * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}", flush=True)
    if stopped:
        raise SystemExit(f"C8 is not run, as sealed: {stopped}. Outcome U.")
    print(f"== Phase A passed under all {len(ORDER)} conditions at a = {A}; checks as expected", flush=True)


def check_phase_a(s: Session, out_dir: str) -> dict:
    pa = json.load(open(os.path.join(out_dir, "phaseA.json")))
    rec = pa["record"]
    problems = []
    if rec["seal_sha256"] != s.seal_hash:
        problems.append("seal hash differs")
    if rec["code_sha256"] != s.code:
        problems.append("code hashes differ")
    if rec["platform"] != s.platform:
        problems.append(f"platform differs: {rec['platform']} vs {s.platform}")
    if rec["adapter_sha256"] != s.adapter_sha:
        problems.append("adapter differs")
    if pa["sham_differences"] != 0:
        problems.append("Phase A sham was not clean")
    if pa.get("stopped"):
        problems.append(f"Phase A stopped C8 by rule ({pa['stopped']}), so C8 is not run, as sealed")
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
            per_cond, shams, refs, ppls = {}, {}, {}, {}
            ids, token_sha = s.ids(gid)
            for cond in ORDER:
                s.set_condition(cond)
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
        write_json(path, {"set": set_name, "gutenberg_id": gid, "a": a, "amplitude_reference": refs,
                          "perplexity": ppls, "token_ids_sha256": token_sha, "sham_differences": shams,
                          "sham_differences_close": sham_close, "seconds": round(time.time() - t_book, 1),
                          "finished_utc": now(), "restart_reason": restart_reason if retried else None,
                          "global_layers": s.global_ids, "per_condition": per_cond, "record": s.record(gid, token_sha)})
        print(f"   done {set_name} {gid} in {round(time.time() - t_book, 1)} s; {s.ledger.seconds / 3600:.2f} hours used",
              flush=True)


def admit(raw_root: str, out_dir: str, seal_hash: str, spec, restart_reason: str = None, new: bool = True) -> list:
    """New run: refused if this run id exists, if any run under the seal is complete or was stopped by rule, or if any
    run exists and no written reason is given; writes STARTED.json. Continue: needs STARTED.json under this seal,
    must be the newest run, refused if another run is complete; a Phase A failure or a book attempt needs a reason."""
    runs = runs_under_seal(raw_root, seal_hash, spec) if os.path.isdir(raw_root) else []
    me = os.path.basename(out_dir)
    others = [r for r in runs if r["run_id"] != me]
    final = [r["run_id"] for r in others if r["state"] in ("complete", "stopped")]
    if new:
        if os.path.exists(os.path.join(out_dir, "STARTED.json")):
            raise SystemExit("This run id already started; raw results are never overwritten.")
        if final:
            raise SystemExit(f"A run under this seal is complete or was stopped by rule: {final}. No new run.")
        if others and not restart_reason:
            raise SystemExit(f"Runs already started under this seal: {others}. A new run is allowed only after an "
                             f"infrastructure failure; pass --restart-reason and keep the old run.")
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, "STARTED.json"),
                   {"utc": now(), "seal_sha256": seal_hash, "restart_reason": restart_reason, "earlier_runs": others},
                   indent=1)
        return others
    started = os.path.join(out_dir, "STARTED.json")
    if not os.path.exists(started) or json.load(open(started)).get("seal_sha256") != seal_hash:
        raise SystemExit(f"{out_dir} has no STARTED.json under this seal.")
    mine = next(r for r in runs if r["run_id"] == me)
    newer = [r["run_id"] for r in others if (r["started_utc"] or "", r["run_id"]) > (mine["started_utc"] or "", me)]
    if newer:
        raise SystemExit(f"A newer run under this seal has started: {newer}. Only the newest run may continue.")
    if final:
        raise SystemExit(f"Another run under this seal is complete or was stopped by rule: {final}.")
    if mine["state"] == "stopped":
        raise SystemExit("This run was stopped by rule in Phase A; C8 is not run, as sealed.")
    if not os.path.exists(os.path.join(out_dir, "phaseA.json")) and \
            glob.glob(os.path.join(out_dir, "*.FAILED.*.json")) and not restart_reason:
        raise SystemExit("This run has a failure record (startup or Phase A). Continuing retries Phase A and needs "
                         "--restart-reason; the record is kept and the reason is stamped into phaseA.json.")
    return others


def adapter_matches_c5(repo, spec):
    ad = spec["adapter"]
    file_sha = sha_file(os.path.join(repo, ad["file"]))
    tj = json.load(open(os.path.join(repo, os.path.dirname(ad["file"]), "training.json")))
    if not (file_sha == ad["sha256"] == tj["adapter_sha256"]):
        raise SystemExit(f"adapter sha256 {file_sha}, sealed {ad['sha256']}, C5 training.json {tj['adapter_sha256']}: "
                         f"they must all agree")


def smoke(args):
    """Pre-seal dry run on the pod: load, adapter and condition checks, perplexity path, then on book 74 under
    BASE and V2 an undriven pass, a sham and driven passes at N = 2 and 4096, and a closing sham. Prints times,
    memory, sham counts, finiteness and BASE perplexity only. Writes nothing to results/."""
    spec = yaml.safe_load(open(args.spec))
    t = time.time()
    s = Session(spec, "smoke", {}, NullLedger(), args, allow_unfilled=True)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    print(f"== adapter {spec['adapter']['file']}: {len(s.adapter_modules)} modules, sha256 {s.adapter_sha}")
    for set_name, books in spec["inputs"]["prompt_sets"].items():
        for g in books:
            n = len(s.tok(s.texts[int(g)], add_special_tokens=False)["input_ids"])
            print(f"   book {set_name} {g}: {n} tokens (needs {s.T})")
    ad = s.adapter_checks()
    print(f"== adapter and condition checks: {ad}")
    lc = s.logits_check()
    print(f"== perplexity path: max |logit difference| {lc['max_abs_diff']:.4g} of max |logit| {lc['max_abs_logit']:.4g}; "
          f"as expected: {lc['as_expected']}")
    ids, _ = s.ids(PHASE_A_BOOK)
    shams, finite = [], True
    torch.cuda.reset_peak_memory_stats()
    for cond in ("BASE", "V2"):
        s.set_condition(cond)
        t = time.time(); m, ppl = s.baseline(ids)
        print(f"   {cond} undriven + perplexity: {time.time() - t:.2f} s" + (f"; BASE perplexity {ppl:.3f}" if cond == "BASE" else ""))
        t = time.time(); shams.append(s.sham_differences(ids)); print(f"   {cond} sham: {time.time() - t:.2f} s")
        for N in PHASE_A_N:
            t = time.time()
            out = s.measure(ids, N, A * m, PHASE_MAIN, f"smoke {cond} N={N}")
            finite &= all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
            print(f"   {cond} driven N={N}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    shams.append(s.sham_differences(ids))
    s.set_condition("BASE")
    print(f"== sham differing values: {shams} (all must be 0); outputs finite: {finite}")
    ok = ad["as_expected"] and lc["as_expected"] and all(x == 0 for x in shams) and finite
    print("== smoke test PASSED" if ok else "== smoke test FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="continue this run (Phase A if it has none, then Phase B)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--restart-reason", help="required for a new run under a seal that already has one, to retry "
                                             "Phase A after a failure, or to retry a book with an earlier attempt")
    ap.add_argument("--smoke", action="store_true", help="pre-seal dry run; writes nothing to results/")
    ap.add_argument("--spec", help="draft spec for --smoke")
    args = ap.parse_args()
    if args.smoke:
        return smoke(args)
    repo = os.path.abspath(args.repo)
    raw_root = os.path.join(repo, "results", "raw")
    ledger_root = os.path.join(repo, "results", "ledger")
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)              # before admit, so a mismatch leaves no orphan run
    check_condition_spec(spec)
    check_no_placeholders(spec)
    adapter_matches_c5(repo, spec)
    glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))   # text hashes, before any run starts
    new = args.run_id is None
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{TAG}-%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(raw_root, run_id)
    lock = RunLock(ledger_root, seal_hash)
    try:
        admit(raw_root, out_dir, seal_hash, spec, args.restart_reason, new=new)
        print(f"== Run {run_id}", flush=True)
        ledger = Ledger(ledger_root, seal_hash, spec["budget"]["max_gpu_hours"] * 3600)
        ledger.check(0.0)
        try:
            s = Session(spec, seal_hash, code, ledger, args)
        except BaseException as e:
            print(f"!! Startup failed; record: {write_failure(out_dir, 'startup', e)}")
            raise
        try:
            if not os.path.exists(os.path.join(out_dir, "phaseA.json")):
                phase_a(s, out_dir, args.restart_reason)
            phase_b(s, out_dir, args.restart_reason)
            print(f"== C8 measurement complete: {os.path.relpath(out_dir, repo)}. Next: "
                  f"PYTHONPATH=src python3 -m dialedin.analyze_c8 --run-id {run_id}")
        finally:
            s.ledger.add("process exit")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
