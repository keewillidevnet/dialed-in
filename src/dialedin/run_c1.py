"""Run C1 exactly as sealed in criteria/sealed/c1-glimmer-frequency-response-v4.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c1 --phase A
    PYTHONPATH=src python3 -m dialedin.run_c1 --phase B --run-id <id printed by phase A>

Phase A picks the pulse amplitude. Phase B runs the 15-book sweep, one raw JSON per
book, and resumes by skipping books already written. Analysis: dialedin.analyze_c1.
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
from .hooks import HeadProbe, PulseInjector, ResidualNormProbe

SEALED = "criteria/sealed/c1-glimmer-frequency-response-v4.yaml"
PHASE_A_BOOK = 1342
PHASE_A_N = (2, 16, 128, 1024)
POWER_GATE_N = 128                  # whole-model SNR at the chosen a must reach 3 here, or C1 stops as underpowered
CANDIDATES = (0.01, 0.02, 0.05, 0.1)
PHASE_A_PASSES = 2 + 2 * len(CANDIDATES) * len(PHASE_A_N)      # unpulsed, sham, then a and 2a per candidate and N


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
    return 3 + 2 * len(grid)                      # unpulsed, two shams, periodic and aperiodic per N


class Ledger:
    """Append-only wall-clock ledger: every second a process spends under one sealed file, from start to exit.

    Lives at results/ledger/, outside results/raw/, one JSON line per entry. Each entry charges
    the wall time since the previous entry, so loads, tokenization and CPU work all count.
    Pod time between processes is billed by RunPod but not counted here.
    """

    def __init__(self, ledger_root: str, seal_hash: str, cap_s: float):
        os.makedirs(ledger_root, exist_ok=True)
        self.path = os.path.join(ledger_root, f"c1v4_{seal_hash[:12]}.jsonl")
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
        self.T = self.spec["inputs"]["length_tokens"]
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["pulse"]["interval_grid_N"]
        self.layer_ids = list(range(2, 52))
        self.global_ids = self.platform["global_layers"]
        self.device = next(self.model.parameters()).device
        hidden = self.layers[1].self_attn.o_proj.out_features
        g = torch.Generator().manual_seed(0)
        d = torch.randn(hidden, generator=g, dtype=torch.float32)
        d = d / d.norm()
        self.direction_sha = sha_array(d.numpy())
        self.direction = d.to(self.device)
        self.injector = PulseInjector(self.layers[1], self.direction)
        self.norms = ResidualNormProbe(self.layers[1])
        self.probe = HeadProbe(self.layers, self.layer_ids, n_heads=32, t1=self.t1)

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
            self.forward(ids, "unpulsed")
        finally:
            self.norms.armed = False
        return float(np.median(self.norms.norms))

    def measure(self, ids, positions, amplitude, N, what):
        cos_b, sin_b = self.probe.set_frequencies(N, self.T, self.device)
        self.injector.arm(positions, amplitude)
        self.probe.mode = "measure"
        try:
            self.forward(ids, what)
        finally:
            self.injector.disarm()
        out = self.probe.collect()
        out["input_mag"] = metrics.train_component(positions, amplitude, cos_b, sin_b, self.t1, self.T)
        out["realized"] = dict(self.injector.realized)
        return out

    def sham_differences(self, ids) -> int:
        """Number of head-output values, over all tokens and layers, whose bits differ from the unpulsed run."""
        self.injector.arm(metrics.periodic_positions(16, self.T), 0.0)
        self.probe.mode = "zero_check"
        try:
            self.forward(ids, "sham")
        finally:
            self.injector.disarm()
        total = int(sum(self.probe.result.values()))
        self.probe.result = {}
        return total

    def record(self, gid, token_sha):
        return {"sealed_file": SEALED, "seal_sha256": self.seal_hash, "code_sha256": self.code,
                "platform": self.platform, "gutenberg_id": gid,
                "text_sha256": hashlib.sha256(self.texts[gid].encode("utf-8")).hexdigest(),
                "token_ids_sha256": token_sha, "tokenizer": type(self.tok).__name__,
                "direction_seed": 0, "direction_sha256": self.direction_sha,
                "offsets": {N: int(metrics.periodic_positions(N, self.T)[0]) for N in self.grid},
                "window": [self.t1, self.T], "T": self.T, "layers_measured": self.layer_ids}


def expected_books(spec) -> list:
    return [f"book_{set_name}_{int(gid)}.json" for set_name, books in spec["inputs"]["prompt_sets"].items()
            for gid in books]


def run_state(run_dir: str, spec) -> str:
    """complete: every book file present. stopped: Phase A ended C1 by rule (no amplitude, or underpowered).
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
        self.path = os.path.join(ledger_root, f"c1v4_{seal_hash[:12]}.lock")
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


def phase_a(s: Session, out_dir: str):
    remaining = (PHASE_A_PASSES + 15 * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    try:
        ids, token_sha = s.ids(PHASE_A_BOOK)
        m = s.baseline(ids)
        sham = s.sham_differences(ids)
        if sham != 0:
            raise RuntimeError(f"Phase A sham differed in {sham} values. Determinism or hooks are broken.")
        rows = []
        for a in CANDIDATES:
            row = {"a": a, "by_N": {}}
            for N in PHASE_A_N:
                pos = metrics.periodic_positions(N, s.T)
                resp, snr = [], None
                for scale in (1, 2):
                    out = s.measure(ids, pos, a * scale * m, N, f"phaseA a={a}x{scale} N={N}")
                    resp.append(float(out["target"].mean()))
                    if scale == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   a={a:<5} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["qualifies"] = all(v["within_10pct"] for v in row["by_N"].values())
            rows.append(row)
    except BaseException as e:
        print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    ok = [r["a"] for r in rows if r["qualifies"]]
    chosen = max(ok) if ok else None
    gate_snr = next(r["by_N"][POWER_GATE_N]["snr_at_a"] for r in rows if r["a"] == chosen) if ok else None
    stopped = ("no amplitude qualified" if chosen is None else
               f"underpowered: whole-model SNR {gate_snr:.2f} < 3 at N = {POWER_GATE_N}" if not gate_snr >= 3 else None)
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    result = {"phase": "A", "finished_utc": now(), "wall_seconds_per_pass": wall_per_pass,
              "book": PHASE_A_BOOK, "median_norm": m, "power_gate": {"N": POWER_GATE_N, "snr": gate_snr},
              "stopped": stopped, "outcome_if_stopped": "U" if stopped else None,
              "sham_differences": sham, "rows": rows, "chosen_a": chosen,
              "seconds_per_pass_last": s.ledger.last_pass, "record": s.record(PHASE_A_BOOK, token_sha)}
    write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
    projected = 15 * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}")
    if stopped:
        raise SystemExit(f"C1 is not run, as sealed: {stopped}. Outcome U (inconclusive), not a kill.")
    print(f"== Phase A chose a = {result['chosen_a']}; power gate SNR {gate_snr:.1f} at N = {POWER_GATE_N}")


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
        problems.append(f"Phase A stopped C1 by rule ({pa['stopped']}), so C1 is not run, as sealed")
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
            m = s.baseline(ids)
            sham_open = s.sham_differences(ids)
            if sham_open != 0:
                raise RuntimeError(f"Opening sham on book {gid} differed in {sham_open} values.")
            amp = a * m
            per_N = {}
            for N in s.grid:
                pos = metrics.periodic_positions(N, s.T)
                op = s.measure(ids, pos, amp, N, f"{gid} N={N} periodic")
                ap = metrics.aperiodic_positions(len(pos), s.T)
                oa = s.measure(ids, ap, amp, N, f"{gid} N={N} aperiodic")
                per_N[N] = {"pulses_total": int(len(pos)), "realized_pulse": op["realized"],
                            "input_mag": op["input_mag"],
                            "target": op["target"].astype(np.float32).tolist(),
                            "floor": op["floor"].astype(np.float32).tolist(),
                            "bin_power": op["bin_power"].tolist(), "n_below": op["n_below"],
                            "aperiodic": {"input_mag": oa["input_mag"],
                                          "target": oa["target"].astype(np.float32).tolist(),
                                          "floor": oa["floor"].astype(np.float32).tolist(),
                                          "bin_power": oa["bin_power"].tolist(), "n_below": oa["n_below"]}}
            sham_close = s.sham_differences(ids)
            if sham_close != 0:
                raise RuntimeError(f"Closing sham on book {gid} differed in {sham_close} values; "
                                   f"results for this book are not written.")
        except BaseException as e:
            print(f"!! Book {gid} failed; record: {write_failure(out_dir, f'book_{set_name}_{gid}', e)}")
            raise
        rec = {"set": set_name, "gutenberg_id": gid, "median_norm": m, "amplitude": amp, "a": a,
               "sham_differences_open": sham_open, "sham_differences_close": sham_close,
               "seconds": round(time.time() - t_book, 1), "finished_utc": now(),
               "restart_reason": restart_reason if retried else None, "global_layers": s.global_ids, "per_N": per_N,
               "record": s.record(gid, token_sha)}
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
    run_id = args.run_id or datetime.now(timezone.utc).strftime("c1v4-%Y%m%dT%H%M%SZ")
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
