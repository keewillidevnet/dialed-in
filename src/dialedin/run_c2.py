"""Run C2 exactly as sealed in criteria/sealed/c2-glimmer-knob-tilt-v1.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c2 --phase A
    PYTHONPATH=src python3 -m dialedin.run_c2 --phase B --run-id <id printed by phase A>

Forked from run_c1v5.py. C2 asks whether the knob (scaling post_attention_layernorm output) on the
13 global layers changes the low-pass tilt C1 v5 measured, with the same knob on 13 local layers as
the specificity control. Drive, window, operator and run rules are C1 v5's. Analysis:
dialedin.analyze_c2.
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
from .hooks import GlobalKnob, HeadProbe, ResidualNormProbe
from .wave import WaveInjector, drive

SEALED = "criteria/sealed/c2-glimmer-knob-tilt-v1.yaml"
TAG = "c2v1"
PHASE_A_BOOK = 1342
PHASE_A_N = (2, 4096)
A = 0.1                             # the amplitude C1 v5's sealed Phase A chose; fixed here, checked in Phase A
PHASE_MAIN = 0.0
GLOBAL_KNOB_LAYERS = list(range(3, 52, 4))    # every global layer
LOCAL_KNOB_LAYERS = list(range(2, 52, 4))     # 13 local layers, one per block, matched in count
CONDITIONS = (("G", 1.0), ("G", 0.5), ("G", 1.5), ("L", 0.5), ("L", 1.5))
PHASE_A_PASSES = 1 + len(CONDITIONS) * (1 + 2 * len(PHASE_A_N))   # sham, then per condition undriven + a and 2a per N


def cond_name(c) -> str:
    return f"{c[0]}{c[1]:.1f}"


class KnobNormProbe:
    """Descriptive: median per-token norm of post_attention_layernorm output in the knob layers, recorded on
    the G1.0 undriven pass only, so the size of what each knob scales can be compared. Not used by any rule."""

    def __init__(self, layers, global_ids, local_ids):
        self.armed = False
        self.values = {}
        self.groups = {i: "global" for i in global_ids} | {i: "local" for i in local_ids}
        self.handles = [layers[i].post_attention_layernorm.register_forward_hook(self._make(i)) for i in self.groups]

    def _make(self, i):
        def hook(module, args, output):
            if self.armed:
                self.values[i] = float(output[0].float().norm(dim=-1).quantile(0.5))
            return None
        return hook

    def summary(self) -> dict:
        out = {}
        for g in ("global", "local"):
            v = [x for i, x in self.values.items() if self.groups[i] == g]
            out[g] = float(np.mean(v)) if v else None
        out["per_layer"] = {str(i): x for i, x in sorted(self.values.items())}
        return out


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


def passes_per_book(grid) -> int:
    return len(CONDITIONS) * (2 + len(grid)) + 1  # per condition undriven, sham and one pass per N; closing sham


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
        """Everything after the model load: window, grid, direction and hooks. The smoke test calls this too."""
        self.T = self.spec["inputs"]["length_tokens"]
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["drive"]["periods_N"]
        self.layer_ids = list(range(2, 52))
        self.global_ids = self.platform["global_layers"]
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
        assert all(not self.layers[i].self_attn.is_local_attention for i in GLOBAL_KNOB_LAYERS)
        assert all(self.layers[i].self_attn.is_local_attention for i in LOCAL_KNOB_LAYERS)
        self.knob_global = GlobalKnob(self.layers, GLOBAL_KNOB_LAYERS)
        self.knob_local = GlobalKnob(self.layers, LOCAL_KNOB_LAYERS)
        self.knob_norms = KnobNormProbe(self.layers, GLOBAL_KNOB_LAYERS, LOCAL_KNOB_LAYERS)

    def set_condition(self, cond):
        """Sets both knobs. Exactly one of them can differ from 1."""
        kind, g = cond
        self.knob_global.g = g if kind == "G" else 1.0
        self.knob_local.g = g if kind == "L" else 1.0

    def ids(self, gid):
        ids = self.tok(self.texts[gid], add_special_tokens=False)["input_ids"][: self.T]
        assert len(ids) == self.T
        arr = np.asarray(ids, dtype=np.int32)
        return torch.tensor([ids], device=self.device), sha_array(arr)

    def forward(self, ids, what: str):
        assert ids.shape[0] == 1, "batch size must be 1"
        self.ledger.check()
        self.ledger.add(f"before {what}")
        with torch.inference_mode():
            self.decoder(input_ids=ids, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ledger.last_pass = self.ledger.add(what)

    def baseline(self, ids) -> float:
        self.injector.disarm()
        self.probe.mode = "record"
        self.norms.armed = True
        try:
            self.forward(ids, "undriven")
        finally:
            self.norms.armed = False
        return float(np.median(self.norms.norms))

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
        return {"knob": {"global_layers": GLOBAL_KNOB_LAYERS, "local_layers": LOCAL_KNOB_LAYERS,
                         "conditions": [cond_name(c) for c in CONDITIONS],
                         "point": "output of post_attention_layernorm, multiplied by g"},
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
    """complete: every book file present. stopped: Phase A ended C2 by rule (not linear or underpowered).
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
    """Linearity and power under every knob condition, on book 1342, at the fixed a, N = 2 and 4096."""
    remaining = (PHASE_A_PASSES + 15 * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    try:
        ids, token_sha = s.ids(PHASE_A_BOOK)
        rows, sham, m0 = [], None, None
        for cond in CONDITIONS:
            s.set_condition(cond)
            m = s.baseline(ids)
            m0 = m if m0 is None else m0
            if m != m0:
                raise RuntimeError(f"amplitude reference changed under {cond_name(cond)}: {m} vs {m0}; "
                                   f"the knob must not reach layer 1")
            if sham is None:
                sham = s.sham_differences(ids)
                if sham != 0:
                    raise RuntimeError(f"Phase A sham differed in {sham} values. Determinism or hooks are broken.")
            row = {"condition": cond_name(cond), "median_norm": m, "by_N": {}}
            for N in PHASE_A_N:
                resp, snr = [], None
                for scale in (1, 2):
                    out = s.measure(ids, N, A * scale * m, PHASE_MAIN, f"phaseA {cond_name(cond)} x{scale} N={N}")
                    resp.append(float(out["target"].mean()))
                    if scale == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   {cond_name(cond)} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["linear"] = all(v["within_10pct"] for v in row["by_N"].values())
            row["powered"] = bool(row["by_N"][max(PHASE_A_N)]["snr_at_a"] >= 3)     # the longest period, 4096
            rows.append(row)
        s.set_condition(("G", 1.0))
    except BaseException as e:
        print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    bad = [r["condition"] for r in rows if not (r["linear"] and r["powered"])]
    stopped = f"not linear or underpowered under {bad}" if bad else None
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    result = {"phase": "A", "finished_utc": now(), "wall_seconds_per_pass": wall_per_pass,
              "book": PHASE_A_BOOK, "a": A, "median_norm": m0, "stopped": stopped,
              "outcome_if_stopped": "U" if stopped else None, "sham_differences": sham, "rows": rows,
              "chosen_a": None if stopped else A,
              "seconds_per_pass_last": s.ledger.last_pass, "record": s.record(PHASE_A_BOOK, token_sha)}
    write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
    projected = 15 * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}")
    if stopped:
        raise SystemExit(f"C2 is not run, as sealed: {stopped}. Outcome U (inconclusive).")
    print(f"== Phase A passed under all {len(CONDITIONS)} conditions at a = {A}")


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
        problems.append(f"Phase A stopped C2 by rule ({pa['stopped']}), so C2 is not run, as sealed")
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
            ids, token_sha = s.ids(gid)
            per_cond, shams, m0, knob_norms = {}, {}, None, None
            for cond in CONDITIONS:
                s.set_condition(cond)
                s.knob_norms.armed = cond == ("G", 1.0)
                try:
                    m = s.baseline(ids)
                finally:
                    s.knob_norms.armed = False
                if cond == ("G", 1.0):
                    knob_norms = s.knob_norms.summary()
                m0 = m if m0 is None else m0
                if m != m0:
                    raise RuntimeError(f"amplitude reference changed under {cond_name(cond)} on book {gid}")
                sh = s.sham_differences(ids)
                if sh != 0:
                    raise RuntimeError(f"Sham under {cond_name(cond)} on book {gid} differed in {sh} values.")
                shams[cond_name(cond)] = sh
                per_cond[cond_name(cond)] = {N: cell(s.measure(ids, N, a * m0, PHASE_MAIN,
                                                               f"{gid} {cond_name(cond)} N={N}")) for N in s.grid}
            sham_close = s.sham_differences(ids)
            if sham_close != 0:
                raise RuntimeError(f"Closing sham on book {gid} differed in {sham_close} values; "
                                   f"results for this book are not written.")
            s.set_condition(("G", 1.0))
        except BaseException as e:
            print(f"!! Book {gid} failed; record: {write_failure(out_dir, f'book_{set_name}_{gid}', e)}")
            raise
        rec = {"set": set_name, "gutenberg_id": gid, "median_norm": m0, "amplitude": a * m0, "a": a,
               "sham_differences": shams, "sham_differences_close": sham_close,
               "seconds": round(time.time() - t_book, 1), "finished_utc": now(),
               "restart_reason": restart_reason if retried else None, "global_layers": s.global_ids,
               "per_condition": per_cond, "knob_layer_output_norms": knob_norms, "record": s.record(gid, token_sha)}
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
