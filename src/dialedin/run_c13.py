"""Run C13 exactly as sealed in criteria/sealed/c13-llama-lean-path-v1.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c13                                  # a new run: Phase A, then Phase B
    PYTHONPATH=src python3 -m dialedin.run_c13 --run-id <id> --restart-reason "<why>"   # continue an interrupted run
    PYTHONPATH=src python3 -m dialedin.run_c13 --smoke --spec criteria/drafts/c13-llama-lean-path-v1.yaml

Forked from run_c11.py for Llama-3.1-8B-Instruct. C11 (outcome VX) found in Muse Glimmer-30B that holding every
attention pattern and gate at its undriven state, so the probe reaches only the values, nearly doubles the lean
(T_V 1.865). C13 asks whether that holds in a different model family whose attention has no output gate and no
sliding-window layers. Same books, segment, probe and measurement as C11, three conditions (names fixed here):
  BASE   the model as it is.
  VONLY  in layers 2 to 31, q_proj and k_proj receive the attention input recorded on the book's undriven pass, so
         every attention pattern is the undriven one and the probe reaches the values (and the MLPs) only.
  PONLY  in layers 2 to 31, v_proj receives the undriven input; queries and keys see the probe.
The model is loaded and checked here (load_llama); glimmer.py supplies only the seal, code, text and determinism
helpers. Analysis: analyze_c13.
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

SEALED = "criteria/sealed/c13-llama-lean-path-v1.yaml"
TAG = "c13v1"
PHASE_A_BOOK = 1342
PHASE_A_N = (2, 4096)
A = 0.1                             # C1 v5's sealed amplitude; checked in Phase A under every condition
PHASE_MAIN = 0.0
ORDER = ("BASE", "VONLY", "PONLY")
MODULES = ("q_proj", "k_proj", "v_proj")                    # under self_attn; Llama has no attention output gate
FROZEN = {"BASE": (), "VONLY": ("q_proj", "k_proj"), "PONLY": ("v_proj",)}
GATED = ("BASE", "VONLY")           # Phase A linearity and power gate these; PONLY is recorded, not gated
PHASE_A_PASSES = 1 + len(ORDER) * (1 + 2 * len(PHASE_A_N))  # one undriven, then per condition a sham and A, 2A per N
PPL_FROM = metrics.WINDOW_START
PPL_CHUNK = 1024
LOGIT_TOL = 0.005
CHECK_TOKENS = 2048
HOST_HEADROOM = 1.25                # host memory available must exceed the recorded inputs by this factor


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
    return 1 + len(ORDER) * (1 + len(grid)) + 1   # one undriven; per condition a sham and one pass per N; closing sham


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
    named = sorted(k for k, v in spec["conditions"].items() if isinstance(v, dict) and "frozen" in v)
    if named != sorted(ORDER):
        raise SystemExit(f"conditions in the spec {named} differ from the code {sorted(ORDER)}")
    got = {c: tuple(spec["conditions"][c]["frozen"]) for c in ORDER}
    if got != FROZEN:
        raise SystemExit(f"frozen modules in the spec differ from the code: {got}")
    if int(spec["inputs"]["segment_start_token"]) != 0:
        raise SystemExit("C13 measures C11's segment: segment_start_token must be 0")


def host_memory() -> dict:
    """Host memory available to this process: /proc/meminfo MemAvailable and the cgroup limit, if any."""
    out = {"mem_available": None, "cgroup_limit": None, "cgroup_used": None}
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                out["mem_available"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    for lim, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                      ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            v = open(lim).read().strip()
            if v != "max" and int(v) < 1 << 60:
                out["cgroup_limit"] = int(v)
                out["cgroup_used"] = int(open(used).read().strip())
            break
        except (OSError, ValueError):
            continue
    free = [x for x in (out["mem_available"],
                        None if out["cgroup_limit"] is None else out["cgroup_limit"] - out["cgroup_used"]) if x is not None]
    out["usable"] = min(free) if free else None
    return out


def load_llama(spec: dict, weights_dir: str, allow_unfilled: bool = False):
    """Llama-3.1-8B-Instruct in bfloat16 with sdpa on one GPU, and its preflight. Returns model, decoder, layers,
    tokenizer and the platform record (every layer is full attention, so all are listed as global)."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    plat = spec["platform"]
    model = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.bfloat16, attn_implementation=plat["attention"],
                                                 device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(weights_dir)
    decoder = model.get_submodule("model")
    layers = model.get_submodule("model.layers")
    cfg = model.config
    problems = []
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.bfloat16}:
        problems.append(f"dtype_bf16: found {dtypes}")
    if getattr(cfg, "quantization_config", None) is not None:
        problems.append("no_quantization: quantization_config present")
    attn = getattr(cfg, "_attn_implementation", None)
    if attn != plat["attention"]:
        problems.append(f"sdpa: attention is {attn}")
    log_path = os.path.join(os.path.dirname(weights_dir.rstrip("/")), "DOWNLOAD_LOG.txt")
    revision = None
    if os.path.exists(log_path):
        for line in open(log_path):
            parts = line.split()                     # <utc> <repo> <sha> <dest>
            if len(parts) >= 3 and parts[1] == plat["model"]:
                revision = parts[2]
    if allow_unfilled and plat["revision"] == "FILL_AT_SEAL":
        print(f"== smoke: the download log has revision {revision}; fill its first 12 characters into the spec")
    elif not revision or not revision.startswith(str(plat["revision"])):
        problems.append(f"revision_match: download log has {revision}")
    if transformers.__version__ != plat["transformers"]:
        problems.append(f"versions_match: transformers {transformers.__version__} != {plat['transformers']}")
    if torch.__version__ != plat["torch"]:
        problems.append(f"versions_match: torch {torch.__version__} != {plat['torch']}")
    gpu = torch.cuda.get_device_name(0)
    if gpu != plat["gpu"]:
        problems.append(f"gpu_match: {gpu}")
    shape = {"layers": len(layers), "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
             "kv_heads": cfg.num_key_value_heads, "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)}
    if shape != plat["shape"]:
        problems.append(f"layer layout: {shape} != {plat['shape']}")
    if getattr(cfg, "sliding_window", None) or getattr(cfg, "use_sliding_window", False):
        problems.append("layer layout: a sliding window is configured")
    if any(not hasattr(l.self_attn, "o_proj") or l.self_attn.o_proj.in_features != shape["heads"] * shape["head_dim"]
           for l in layers):
        problems.append("layer layout: o_proj input is not heads x head_dim")
    if problems:
        raise SystemExit("Preflight failed:\n  " + "\n  ".join(problems))
    record = {"dtype": "bfloat16", "attention": attn, "revision": revision, "transformers": transformers.__version__,
              "torch": torch.__version__, "gpu": gpu, "shape": shape, "global_layers": list(range(len(layers))),
              "batch_size": 1, "deterministic": True, "tf32": False}
    return model, decoder, layers, tok, record


class PathFreeze:
    """Forward pre-hooks on self_attn.q_proj, k_proj and v_proj of the measured layers. Batch size 1.

    record: on an undriven pass, copies each layer's q_proj input (the attention input; the setup check requires q,
            k and v to receive the same values) into a preallocated host buffer, keyed to the pass's token ids.
    frozen: each module named in self.frozen receives the recorded input instead of this pass's input. One GPU slot
            holds the current layer's recorded input; it is filled once per layer per pass.
    None:   does nothing.
    Every hook counts its firings per pass; verify() requires exactly one per active hook per layer."""

    def __init__(self, layers, layer_ids, T: int, hidden: int, device, dtype=torch.bfloat16):
        self.layer_ids, self.T, self.hidden = list(layer_ids), T, hidden
        self.mode, self.frozen, self.key = None, (), None
        missing = [f"layers.{i}.self_attn.{n}" for i in self.layer_ids for n in MODULES
                   if not isinstance(getattr(layers[i].self_attn, n, None), torch.nn.Linear)]
        if missing:
            raise SystemExit(f"modules missing or not nn.Linear: {missing}")
        self.store_bytes = len(self.layer_ids) * T * hidden * torch.empty((), dtype=dtype).element_size()
        mem = host_memory()
        if mem["usable"] is not None and mem["usable"] < HOST_HEADROOM * self.store_bytes:
            raise SystemExit(f"host memory: {mem['usable'] / 2**30:.1f} GiB usable, the recorded inputs need "
                             f"{self.store_bytes / 2**30:.1f} GiB x {HOST_HEADROOM}")
        self.host_memory_before = mem
        self.pinned = True
        try:
            self.store = {i: torch.empty((T, hidden), dtype=dtype, pin_memory=True) for i in self.layer_ids}
        except RuntimeError:
            self.pinned = False
            self.store = {i: torch.empty((T, hidden), dtype=dtype) for i in self.layer_ids}
        self.slot = torch.empty((1, T, hidden), dtype=dtype, device=device)
        self.slot_layer = None
        self.fired = {}
        self.handles = [getattr(layers[i].self_attn, n).register_forward_pre_hook(self._make(i, n))
                        for i in self.layer_ids for n in MODULES]

    def begin(self):
        self.fired = {}
        self.slot_layer = None

    def _make(self, i, name):
        def hook(module, args):
            if self.mode == "record":
                if name != "q_proj":
                    return None
                x = args[0]
                if tuple(x.shape) != (1, self.T, self.hidden):
                    raise RuntimeError(f"layer {i} attention input has shape {tuple(x.shape)}")
                self.store[i].copy_(x[0].detach(), non_blocking=self.pinned)
                self.fired[(i, name)] = self.fired.get((i, name), 0) + 1
                return None
            if self.mode == "frozen" and name in self.frozen:
                if self.key is None:
                    raise RuntimeError("frozen pass with no recorded undriven pass")
                if tuple(args[0].shape) != (1, self.T, self.hidden):
                    raise RuntimeError(f"layer {i} {name} input has shape {tuple(args[0].shape)}")
                if self.slot_layer != i:
                    self.slot[0].copy_(self.store[i], non_blocking=self.pinned)
                    self.slot_layer = i
                self.fired[(i, name)] = self.fired.get((i, name), 0) + 1
                return (self.slot,) + tuple(args[1:])
            return None
        return hook

    def verify(self):
        want = ({(i, "q_proj") for i in self.layer_ids} if self.mode == "record" else
                {(i, n) for i in self.layer_ids for n in self.frozen} if self.mode == "frozen" else set())
        if set(self.fired) != want or any(v != 1 for v in self.fired.values()):
            bad = sorted(k for k in want | set(self.fired) if self.fired.get(k) != 1)
            raise RuntimeError(f"freeze hooks fired unexpectedly in mode {self.mode}: {bad[:6]}")
        return len(self.fired)

    def remove(self):
        for h in self.handles:
            h.remove()


class Ledger:
    """Append-only wall-clock ledger under results/ledger/ (as C1 to C12)."""

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
    """Llama with C1 v5's measurement hooks plus the path freeze."""

    def __init__(self, spec, seal_hash, code, ledger, args):
        self.repo = os.path.abspath(args.repo)
        self.spec, self.seal_hash, self.code, self.ledger = spec, seal_hash, code, ledger
        try:
            self.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
            glimmer.configure_determinism()
            self.model, self.decoder, self.layers, self.tok, self.platform = load_llama(
                spec, args.weights, allow_unfilled=bool(getattr(args, "smoke", False)))
        except BaseException:
            self.ledger.add("startup failed")
            raise
        self.ledger.add("startup and model load")
        self._setup()

    def _setup(self):
        check_condition_spec(self.spec)
        self.T = self.spec["inputs"]["length_tokens"]
        self.start = int(self.spec["inputs"]["segment_start_token"])
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["drive"]["periods_N"]
        self.layer_ids = list(range(2, len(self.layers)))
        self.global_ids = self.platform["global_layers"]
        self.device = next(self.model.parameters()).device
        hidden = self.decoder.config.hidden_size
        g = torch.Generator().manual_seed(0)
        d = torch.randn(hidden, generator=g, dtype=torch.float32)
        d = d / d.norm()
        self.direction_sha = sha_array(d.numpy())
        self.direction = d.to(self.device)
        self.injector = WaveInjector(self.layers[1], self.direction)
        self.norms = ResidualNormProbe(self.layers[1])
        self.probe = HeadProbe(self.layers, self.layer_ids, n_heads=self.platform["shape"]["heads"], t1=self.t1)
        self.freeze = PathFreeze(self.layers, self.layer_ids, self.T, hidden, self.device)
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

    def input_identity_check(self) -> dict:
        """On the first 2,048 tokens of book 1342: in every measured layer, q_proj, k_proj and v_proj must receive the
        same values bit for bit, so one recorded tensor per layer stands for all three."""
        first, same = {}, {}             # one tensor per layer, kept only until its four modules have fired

        def make(i, n):
            def hook(m, a):
                if i not in first:
                    first[i] = a[0].detach().clone()
                    same[(i, n)] = True
                else:                       # a fifth firing in one layer finds None and counts as differing
                    same[(i, n)] = first[i] is not None and (i, n) not in same and bool(torch.equal(a[0], first[i]))
                if sum(1 for k in same if k[0] == i) == len(MODULES):
                    first[i] = None
            return hook
        hs = [getattr(self.layers[i].self_attn, n).register_forward_pre_hook(make(i, n))
              for i in self.layer_ids for n in MODULES]
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:CHECK_TOKENS]],
                         device=self.device)
        self.probe.mode, self.freeze.mode = None, None
        try:
            with torch.inference_mode():
                self.decoder(input_ids=x, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        differ = [f"layers.{i}.{n}" for i in self.layer_ids for n in MODULES if not same.get((i, n), False)]
        return {"layers": len(self.layer_ids), "modules": list(MODULES), "differ_from_q_input": differ,
                "as_expected": not differ}

    def logits(self, h):
        x = self.head(h)
        if self.multiplier is not None:
            x = x * self.multiplier
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x.float()

    def logits_check(self) -> dict:
        self.probe.mode, self.freeze.mode = None, None
        x = torch.tensor([self.tok(self.texts[PHASE_A_BOOK], add_special_tokens=False)["input_ids"][:CHECK_TOKENS]],
                         device=self.device)
        with torch.inference_mode():
            ref = self.model(input_ids=x, use_cache=False).logits[0].float()
            ours = self.logits(self.decoder(input_ids=x, use_cache=False).last_hidden_state[0])
        diff, scale = float((ours - ref).abs().max()), float(ref.abs().max())
        return {"max_abs_diff": diff, "max_abs_logit": scale, "tolerance": LOGIT_TOL * scale,
                "as_expected": bool(diff <= LOGIT_TOL * scale)}

    def ids(self, gid):
        """Token ids [start, start + T) of the stripped text, no BOS (the sealed segment)."""
        raw = np.asarray(self.tok(self.texts[gid], add_special_tokens=False)["input_ids"][self.start: self.start + self.T],
                         dtype=np.int64)
        if len(raw) != self.T:
            raise SystemExit(f"book {gid} is too short for tokens {self.start} to {self.start + self.T - 1}")
        return torch.tensor(raw[None, :], device=self.device), sha_array(raw.astype(np.int32))

    def forward(self, ids, what: str):
        assert ids.shape[0] == 1, "batch size must be 1"
        self.ledger.check()
        self.ledger.add(f"before {what}")
        self.freeze.begin()
        with torch.inference_mode():
            out = self.decoder(input_ids=ids, use_cache=False)
            if self.keep_hidden:
                self.last_hidden = out.last_hidden_state[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.freeze.verify()
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

    def baseline(self, ids, token_sha):
        """The book's undriven pass: head outputs for the probe, attention inputs for the freeze, residual norms
        for the amplitude, and perplexity. Every condition is compared with this pass."""
        self.injector.disarm()
        self.probe.mode = "record"
        self.freeze.mode, self.freeze.key = "record", None
        self.norms.armed = True
        self.keep_hidden = True
        try:
            self.forward(ids, "undriven")
            self.freeze.key = token_sha
        finally:
            self.norms.armed = False
            self.keep_hidden = False
            self.freeze.mode = None
        return float(np.median(self.norms.norms)), self.perplexity(ids)

    def _arm(self, cond, token_sha):
        if self.freeze.key != token_sha:
            raise RuntimeError("the recorded attention inputs are from another text")
        self.freeze.frozen = FROZEN[cond]
        self.freeze.mode = "frozen" if FROZEN[cond] else None

    def measure(self, ids, token_sha, cond, N, amplitude, phase, what):
        cos_b, sin_b = self.probe.set_frequencies(N, self.T, self.device)
        w = drive(N, self.T, amplitude, phase)
        self._arm(cond, token_sha)
        self.injector.arm(w)
        self.probe.mode = "measure"
        try:
            self.forward(ids, what)
        finally:
            self.injector.disarm()
            self.freeze.mode = None
        out = self.probe.collect()
        out["input_mag"] = float(metrics.component(self.injector.realized[self.t1:], cos_b[:, :1], sin_b[:, :1])[0])
        out["input_mag_nominal"] = float(metrics.component(w[self.t1:], cos_b[:, :1], sin_b[:, :1])[0])
        out["realized"] = dict(self.injector.stats, realized_over_nominal=out["input_mag"] / out["input_mag_nominal"])
        return out

    def sham_differences(self, ids, token_sha, cond) -> int:
        """Zero wave under cond: with the freeze armed, the substituted inputs equal this pass's own, so every head
        output must keep the undriven pass's bits."""
        self._arm(cond, token_sha)
        self.injector.arm(np.zeros(self.T))
        self.probe.mode = "zero_check"
        try:
            self.forward(ids, f"sham {cond}")
        finally:
            self.injector.disarm()
            self.freeze.mode = None
        total = int(sum(self.probe.result.values()))
        self.probe.result = {}
        return total

    def record(self, gid, token_sha):
        return {"conditions": {c: {"frozen": list(FROZEN[c])} for c in ORDER},
                "freeze": {"layers": self.freeze.layer_ids, "modules": list(MODULES), "pinned": self.freeze.pinned,
                           "store_bytes": self.freeze.store_bytes},
                "softcap": self.softcap, "output_multiplier": self.multiplier, "sealed_file": SEALED,
                "seal_sha256": self.seal_hash, "code_sha256": self.code, "platform": self.platform,
                "gutenberg_id": gid,
                "text_sha256": hashlib.sha256(self.texts[gid].encode("utf-8")).hexdigest() if gid is not None else None,
                "token_ids_sha256": token_sha, "tokenizer": type(self.tok).__name__,
                "direction_seed": 0, "direction_sha256": self.direction_sha,
                "drive": {"form": "amplitude * cos(2 pi t / N + phase) at every token t", "phase_main": PHASE_MAIN},
                "segment_tokens": [self.start, self.start + self.T], "window": [self.t1, self.T], "T": self.T,
                "layers_measured": self.layer_ids}


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


def earlier_failures(out_dir):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.FAILED.*.json")))


def phase_a(s: Session, out_dir: str, restart_reason: str = None):
    """Setup checks, then on the Phase A book's segment: one undriven pass, and under every condition a sham and
    A and 2A at N = 2 and 4096."""
    remaining = (PHASE_A_PASSES + n_books(s.spec) * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    setup_failed = None
    try:
        ident = s.input_identity_check()
        print(f"   attention input identity: {ident['as_expected']} ({len(ident['differ_from_q_input'])} differ)", flush=True)
        logit_check = s.logits_check()
        print(f"   perplexity path: max |logit difference| {logit_check['max_abs_diff']:.4g} "
              f"(max |logit| {logit_check['max_abs_logit']:.4g}); as expected: {logit_check['as_expected']}", flush=True)
        failed_checks = ((["attention input identity"] if not ident["as_expected"] else [])
                         + (["perplexity path"] if not logit_check["as_expected"] else []))
        if failed_checks:
            write_json(os.path.join(out_dir, "phaseA.json"),
                       {"phase": "A", "finished_utc": now(), "restart_reason": restart_reason,
                        "earlier_failures": earlier_failures(out_dir), "identity_check": ident,
                        "logits_check": logit_check, "stopped": f"setup checks failed: {failed_checks}",
                        "outcome_if_stopped": "U", "sham_differences": None, "rows": [], "chosen_a": None,
                        "record": s.record(PHASE_A_BOOK, None)}, indent=2)
            setup_failed = failed_checks
            raise SystemExit(f"C13 is not run, as sealed: setup checks failed: {failed_checks}. Outcome U.")
        ids, base_sha = s.ids(PHASE_A_BOOK)
        m, ppl = s.baseline(ids, base_sha)
        rows, shams, at_a = [], {}, {}
        for cond in ORDER:
            shams[cond] = s.sham_differences(ids, base_sha, cond)
            if shams[cond] != 0:
                raise RuntimeError(f"Phase A sham under {cond} differed in {shams[cond]} values. "
                                   f"Determinism, hooks or the freeze are broken.")
            row = {"condition": cond, "frozen": list(FROZEN[cond]), "by_N": {}}
            for N in PHASE_A_N:
                resp, snr = [], None
                for k in (1, 2):
                    out = s.measure(ids, base_sha, cond, N, A * k * m, PHASE_MAIN, f"phaseA {cond} x{k} N={N}")
                    resp.append(float(out["target"].mean()))
                    if k == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                        at_a[(cond, N)] = out["target"]
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   {cond:<5} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["linear"] = all(v["within_10pct"] for v in row["by_N"].values())
            row["powered"] = bool(row["by_N"][max(PHASE_A_N)]["snr_at_a"] >= 3)
            row["gated"] = cond in GATED
            rows.append(row)
        effect = {f"{c.lower()}_differs_from_base_N{N}": bool(not np.array_equal(at_a[(c, N)], at_a[("BASE", N)]))
                  for c in ORDER[1:] for N in PHASE_A_N}
        effect["vonly_differs_from_ponly"] = all(bool(not np.array_equal(at_a[("VONLY", N)], at_a[("PONLY", N)]))
                                                 for N in PHASE_A_N)
        effect["as_expected"] = all(effect.values())
    except BaseException as e:
        if setup_failed is None:
            print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    bad = [r["condition"] for r in rows if r["gated"] and not (r["linear"] and r["powered"])]
    stopped = (f"not linear or underpowered under {bad}" if bad else
               None if effect["as_expected"] else f"the freeze did not change the driven head outputs: {effect}")
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    write_json(os.path.join(out_dir, "phaseA.json"),
               {"phase": "A", "finished_utc": now(), "restart_reason": restart_reason,
                "earlier_failures": earlier_failures(out_dir), "wall_seconds_per_pass": wall_per_pass,
                "book": PHASE_A_BOOK, "a": A, "median_norm": m, "perplexity": ppl, "identity_check": ident,
                "logits_check": logit_check, "effect_check": effect, "stopped": stopped,
                "outcome_if_stopped": "U" if stopped else None, "sham_differences": shams, "rows": rows,
                "chosen_a": None if stopped else A, "seconds_per_pass_last": s.ledger.last_pass,
                "record": s.record(PHASE_A_BOOK, base_sha)}, indent=2)
    projected = n_books(s.spec) * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}", flush=True)
    if stopped:
        raise SystemExit(f"C13 is not run, as sealed: {stopped}. Outcome U.")
    print(f"== Phase A passed under {', '.join(GATED)} at a = {A}; PONLY recorded; checks as expected", flush=True)


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
    if not isinstance(pa["sham_differences"], dict) or any(v != 0 for v in pa["sham_differences"].values()):
        problems.append("Phase A shams were not clean")
    if pa.get("stopped"):
        problems.append(f"Phase A stopped C13 by rule ({pa['stopped']}), so C13 is not run, as sealed")
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
            per_cond, shams = {}, {}
            ids, token_sha = s.ids(gid)
            m, ppl = s.baseline(ids, token_sha)
            for cond in ORDER:
                sh = s.sham_differences(ids, token_sha, cond)
                if sh != 0:
                    raise RuntimeError(f"Sham under {cond} on book {gid} differed in {sh} values.")
                shams[cond] = sh
                per_cond[cond] = {N: cell(s.measure(ids, token_sha, cond, N, a * m, PHASE_MAIN, f"{gid} {cond} N={N}"))
                                  for N in s.grid}
            sham_close = s.sham_differences(ids, token_sha, "BASE")
            if sham_close != 0:
                raise RuntimeError(f"Closing sham on book {gid} differed in {sham_close} values; "
                                   f"results for this book are not written.")
        except BaseException as e:
            print(f"!! Book {gid} failed; record: {write_failure(out_dir, f'book_{set_name}_{gid}', e)}")
            raise
        write_json(path, {"set": set_name, "gutenberg_id": gid, "a": a, "amplitude_reference": m,
                          "perplexity": ppl, "token_ids_sha256": token_sha, "sham_differences": shams,
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
        raise SystemExit("This run was stopped by rule in Phase A; C13 is not run, as sealed.")
    if not os.path.exists(os.path.join(out_dir, "phaseA.json")) and \
            glob.glob(os.path.join(out_dir, "*.FAILED.*.json")) and not restart_reason:
        raise SystemExit("This run has a failure record (startup or Phase A). Continuing retries Phase A and needs "
                         "--restart-reason; the record is kept and the reason is stamped into phaseA.json.")
    return others


def smoke(args):
    """Pre-seal dry run on the pod: load (printing the weights' revision), host memory, attention input identity,
    perplexity path; then on the Phase A
    book's segment the undriven (recording) pass, a sham under every condition, driven passes at N = 2 and 4096 under
    every condition, and a closing sham. Prints times, memory, hook counts, sham counts, finiteness, whether each
    frozen condition changes the driven outputs, and BASE perplexity only. Writes nothing to results/."""
    spec = yaml.safe_load(open(args.spec))
    t = time.time()
    s = Session(spec, "smoke", {}, NullLedger(), args)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    hm = host_memory()
    print(f"== host memory before the store: {s.freeze.host_memory_before}; now usable "
          f"{(hm['usable'] or 0) / 2**30:.1f} GiB; store {s.freeze.store_bytes / 2**30:.1f} GiB, pinned {s.freeze.pinned}")
    for set_name, books in spec["inputs"]["prompt_sets"].items():
        for g in books:
            n = len(s.tok(s.texts[int(g)], add_special_tokens=False)["input_ids"])
            print(f"   book {set_name} {g}: {n} tokens (needs {s.start + s.T})")
    ident = s.input_identity_check()
    print(f"== attention input identity (q, k, v, {ident['layers']} layers): {ident['as_expected']}"
          + (f"; differ: {ident['differ_from_q_input'][:8]}" if not ident["as_expected"] else ""))
    lc = s.logits_check()
    print(f"== perplexity path: max |logit difference| {lc['max_abs_diff']:.4g} of max |logit| {lc['max_abs_logit']:.4g}; "
          f"as expected: {lc['as_expected']}")
    ids, sha = s.ids(PHASE_A_BOOK)
    shams, finite, at = [], True, {}
    torch.cuda.reset_peak_memory_stats()
    t = time.time(); m, ppl = s.baseline(ids, sha)
    print(f"   undriven (recording) + perplexity: {time.time() - t:.2f} s; BASE perplexity {ppl:.3f}; "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    for cond in ORDER:
        t = time.time(); shams.append(s.sham_differences(ids, sha, cond))
        print(f"   {cond} sham: {time.time() - t:.2f} s, {len(s.freeze.fired)} freeze hook firings")
        for N in PHASE_A_N:
            t = time.time()
            out = s.measure(ids, sha, cond, N, A * m, PHASE_MAIN, f"smoke {cond} N={N}")
            at[(cond, N)] = out["target"]
            finite &= all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
            print(f"   {cond} driven N={N}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    shams.append(s.sham_differences(ids, sha, "BASE"))
    changed = {f"{c} N={N}": bool(not np.array_equal(at[(c, N)], at[("BASE", N)])) for c in ORDER[1:] for N in PHASE_A_N}
    print(f"== sham differing values: {shams} (all must be 0); outputs finite: {finite}")
    print(f"== frozen conditions change the driven head outputs: {changed} (all must be true)")
    ok = ident["as_expected"] and lc["as_expected"] and all(x == 0 for x in shams) and finite and all(changed.values())
    print("== smoke test PASSED" if ok else "== smoke test FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="continue this run (Phase A if it has none, then Phase B)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Llama-3.1-8B-Instruct")
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
            print(f"== C13 measurement complete: {os.path.relpath(out_dir, repo)}. Next: "
                  f"PYTHONPATH=src python3 -m dialedin.analyze_c13 --run-id {run_id}")
        finally:
            s.ledger.add("process exit")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
