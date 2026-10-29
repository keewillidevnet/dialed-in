"""Run C5 exactly as sealed in criteria/sealed/c5-glimmer-trained-dial-v1.yaml.

    PYTHONPATH=src python3 -m dialedin.run_c5 --phase T
    PYTHONPATH=src python3 -m dialedin.run_c5 --phase A --run-id <id printed by phase T>
    PYTHONPATH=src python3 -m dialedin.run_c5 --phase B --run-id <id>
    PYTHONPATH=src python3 -m dialedin.run_c5 --phase R --run-id <id>

Forked from run_c4.py. C5 asks whether a low-rank adapter, trained once on text from sets B and C to flatten the
tilt in proportion to its scale, acts as a dial on Glimmer's frequency response on set A at C1 v5's full 32K
measurement, without damaging the model. Phases:
  T  train the adapter with the sealed recipe (the C5 pilot run 3 recipe) and write it with its hash;
  A  setup checks (perplexity path, scale 0 is the base model bit for bit, the adapter takes effect), then
     linearity and power under every scale on book 1342;
  B  per book of set A and per scale (0, 0.5, 1, 1.5, 2): undriven pass with perplexity, sham, 7 driven periods;
  R  exploratory behavior at every scale (reported, never gated): passkey recall, adaptation after a topic switch,
     and recovery after a short burst of noise.
Analysis: dialedin.analyze_c5.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import glob
import hashlib
import json
import math
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

SEALED = "criteria/sealed/c5-glimmer-trained-dial-v1.yaml"
TAG = "c5v1"
PHASE_A_BOOK = 1342
PHASE_A_N = (2, 4096)
A = 0.1                             # the amplitude C1 v5's sealed Phase A chose; fixed here, checked in Phase A
PHASE_MAIN = 0.0
CONDITIONS = {"S0": 0.0, "S0.5": 0.5, "S1": 1.0, "S1.5": 1.5, "S2": 2.0}     # name: adapter scale
ORDER = ("S0", "S0.5", "S1", "S1.5", "S2")
PHASE_A_PASSES = 1 + len(ORDER) * (1 + 2 * len(PHASE_A_N))
PPL_FROM = metrics.WINDOW_START     # perplexity over predictions of tokens PPL_FROM+1 .. T-1
PPL_CHUNK = 1024
LOGIT_TOL = 0.005                   # perplexity-path check: max |logit difference| <= 0.5 percent of max |logit|
CHECK_LAYER = 10                    # the scale-0 and adapter checks read this layer's o_proj input
RECALL = {"context_tokens": 16384, "filler_start": 40000, "depths": (0.1, 0.3, 0.6, 0.9), "key_seed": 5,
          "needle": " The secret pass key is {key}. Remember it.",
          "question": "\nWhat is the secret pass key? The secret pass key is", "answer": " {key}"}
SWITCH = {"start": 53856, "context": 6144, "after": 2048, "noise_tokens": 32, "noise_seed": 7,
          "bins": ((1, 16), (17, 64), (65, 256), (257, 1024), (1025, 2048))}


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
    the wall time since the previous entry, so loads, tokenization, training and CPU work all count.
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


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ phase T
def train(spec, seal_hash, code, ledger, args, out_dir, loaded=None):
    """Trains the adapter once with the sealed recipe; writes adapter.pt and training.json (with the adapter's
    sha256). Training uses only the sealed training regions (sets B and C, tokens 40,960 to 47,103).
    loaded: (texts, model, decoder, layers, tok, platform), for tests; None loads Glimmer."""
    tr = spec["training"]
    torch.manual_seed(tr["seed"])
    if loaded is None:
        texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
        glimmer.configure_determinism()
        torch.manual_seed(tr["seed"])
        model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    else:
        texts, model, decoder, layers, tok, platform = loaded
    torch.use_deterministic_algorithms(False)       # sealed: training backward has nondeterministic kernels
    ledger.add("training startup and model load")
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    periods = P.configure(tr["tokens"])
    assert list(periods) == list(tr["periods_N"]), "training periods must be the sealed training periods"
    dim = decoder.config.hidden_size
    rig = P.Rig(layers, range(2, len(layers)))
    logits = P.logits_fn(model.get_output_embeddings(), getattr(decoder.config, "output_multiplier", None),
                         getattr(decoder.config, "final_logit_softcapping", None))
    regions = []
    for reg in tr["regions"]:
        ids_all = tok(texts[int(reg["gutenberg_id"])], add_special_tokens=False)["input_ids"]
        assert len(ids_all) >= reg["start"] + P.T, (reg, len(ids_all))
        regions.append({"gid": int(reg["gutenberg_id"]), "start": reg["start"],
                        "ids": torch.tensor([ids_all[reg["start"]:reg["start"] + P.T]], device=dev)})
    # base-gain cache per region at scale 0 (no adapter yet), and the amplitude reference per region
    cache = []
    with torch.no_grad():
        for reg in regions:
            rig.B = None
            P.one_pass(decoder, rig, reg["ids"], None)
            reg["norm"] = rig.median_norm
            rows = P.evaluate(decoder, rig, reg["ids"], tr["base_seeds"], [0.0], reg["norm"], dim, periods=periods)
            reg["base_gain"] = {n: float(np.mean([r["gains"][k] for r in rows])) for k, n in enumerate(periods)}
            reg["base_tilt"] = float(P.slope(list(periods), torch.tensor([reg["base_gain"][n] for n in periods])))
            cache.append({k: reg[k] for k in ("gid", "start", "norm", "base_gain", "base_tilt")})
            print(f"== region {reg['gid']}@{reg['start']}: base tilt {reg['base_tilt']:.4f}", flush=True)
    ledger.add("base-gain cache")
    tilt0 = float(np.mean([r["base_tilt"] for r in regions]))
    names, params = P.attach_lora(layers, tr["layers"], tr["rank"], tr["alpha"], family=tr["family"])
    undo = P.checkpoint_layers(layers)
    opt = torch.optim.Adam(params, lr=tr["lr"])
    rng = np.random.default_rng(tr["seed"])
    snap = {"params": [p.detach().clone() for p in params], "opt": None, "step": -1}
    steps, events = [], []
    for step in range(tr["steps"]):
        ledger.check()
        reg = regions[step % len(regions)]
        ids = reg["ids"]
        t0 = time.time()
        per = sorted(rng.choice(periods, size=tr["periods_per_step"], replace=False).tolist())
        d = P.direction(tr["direction_seed_offset"] + step, dim, dev)
        w, _ = P.tones(per, P.TONE_A * reg["norm"], rng)
        with torch.no_grad():
            P.LoRALinear.scale, rig.B = 0.0, None
            h_base = P.one_pass(decoder, rig, ids, None)[0].last_hidden_state[0]
        sc = float(rng.uniform(*tr["scale_range"]))
        base_tilt = float(P.slope(per, torch.tensor([reg["base_gain"][n] for n in per])))
        target = base_tilt * (1.0 + sc * tr["shift"])
        P.LoRALinear.scale = sc
        rig.B = P.basis(per, dev)
        out0, p0 = P.one_pass(decoder, rig, ids, None)
        delta = torch.as_tensor(w, dtype=torch.float32, device=dev)[:, None] * d[None, :]
        _, p1 = P.one_pass(decoder, rig, ids, delta)
        g, nz = rig.curves(p0, p1, per, P.input_mags(w, per, P.basis(per, "cpu")))
        tilt = P.slope(per, g)
        tilt_loss = ((tilt - target) / tilt0) ** 2
        kl = P.kl_to_base(out0.last_hidden_state[0], h_base, logits)
        loss = tilt_loss + tr["beta"] * kl + tr["beta_excess"] * torch.relu(kl - tr["kl_budget"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, tr["clip"]))
        if math.isfinite(gn):
            opt.step()
        rig.proj, rig.B = {}, None
        row = {"step": step, "region": f"{reg['gid']}@{reg['start']}", "periods": per, "scale": sc,
               "base_tilt": base_tilt, "target": target, "tilt": float(tilt.detach()),
               "tilt_loss": float(tilt_loss.detach()), "kl": float(kl.detach()), "grad_norm": gn,
               "seconds": time.time() - t0, "min_snr": float((g / nz).min().detach())}
        bad = row["kl"] > tr["kl_rollback"] or not math.isfinite(gn)
        if bad:
            lr_now = opt.param_groups[0]["lr"]
            with torch.no_grad():
                for p, q in zip(params, snap["params"]):
                    p.copy_(q)
            if snap["opt"] is not None:
                opt.load_state_dict(snap["opt"])
            for gp in opt.param_groups:
                gp["lr"] = lr_now * 0.5
            events.append({"step": step, "kl": row["kl"], "restored_to_step": snap["step"], "new_lr": lr_now * 0.5})
            print(f"== rollback at step {step}: KL {row['kl']:.2f}; restored step {snap['step']}", flush=True)
        elif row["kl"] <= 2 * tr["kl_budget"] and step % 10 == 0:
            snap = {"params": [p.detach().clone() for p in params], "opt": copy.deepcopy(opt.state_dict()),
                    "step": step}
        row["rolled_back"] = bad
        steps.append(row)
        ledger.last_pass = ledger.add(f"training step {step}")
        print(f"   step {step:>3} {row['region']}: N {per} scale {sc:.2f} tilt {row['tilt']:.4f} base {base_tilt:.4f} "
              f"target {target:.4f}  kl {row['kl']:.2e}  grad {gn:.2e}  {row['seconds']:.1f} s", flush=True)
        del h_base, out0, g, nz, tilt, loss, kl
    undo()
    P.LoRALinear.scale = 0.0
    adapter_path = os.path.join(out_dir, "adapter.pt")
    torch.save({"family": tr["family"], "layers": tr["layers"], "rank": tr["rank"], "alpha": tr["alpha"],
                "modules": names, "A": [p.detach().cpu() for p in params[0::2]],
                "B": [p.detach().cpu() for p in params[1::2]]}, adapter_path)
    adapter_sha = sha_file(adapter_path)
    write_json(os.path.join(out_dir, "training.json"),
               {"finished_utc": now(), "adapter_file": "adapter.pt", "adapter_sha256": adapter_sha,
                "modules": names, "params": int(sum(p.numel() for p in params)), "tilt0": tilt0,
                "regions": cache, "steps": steps, "rollbacks": events,
                "record": {"sealed_file": SEALED, "seal_sha256": seal_hash, "code_sha256": code,
                           "platform": platform, "training_deterministic": False, "training": tr}}, indent=1)
    print(f"== adapter trained: {len(names)} modules, sha256 {adapter_sha[:16]}..., {len(events)} rollbacks")


# ------------------------------------------------------------------------------------------------ phases A, B, R
class Session:
    def __init__(self, args, ledger_root: str, out_dir: str):
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
        self._setup(out_dir)

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

    def _setup(self, out_dir: str):
        """Reference before the adapter, then the adapter (hash checked), then window, grid, direction and hooks."""
        self.T = self.spec["inputs"]["length_tokens"]
        self.t1 = metrics.WINDOW_START
        self.grid = self.spec["drive"]["periods_N"]
        self.layer_ids = list(range(2, len(self.layers)))      # layers 2 to 51 on Glimmer, as in C1 v5
        self.global_ids = self.platform["global_layers"]
        self.device = next(self.model.parameters()).device
        self.unadapted_ref = self._reference_input()           # layer 10 o_proj input before any wrapper exists
        tj = json.load(open(os.path.join(out_dir, "training.json")))
        if tj["record"]["seal_sha256"] != self.seal_hash:
            raise SystemExit("training.json was written under a different seal")
        path = os.path.join(out_dir, tj["adapter_file"])
        self.adapter_sha = sha_file(path)
        if self.adapter_sha != tj["adapter_sha256"]:
            raise SystemExit("adapter.pt does not match the sha256 recorded at training")
        ad = torch.load(path, map_location="cpu")
        tr = self.spec["training"]
        names, params = P.attach_lora(self.layers, tr["layers"], tr["rank"], tr["alpha"], family=tr["family"])
        if names != ad["modules"] or names != tj["modules"]:
            raise SystemExit("the adapter's modules differ from the sealed family and layers")
        with torch.no_grad():
            for (a, b), pa, pb in zip(zip(ad["A"], ad["B"]), params[0::2], params[1::2]):
                pa.copy_(a.to(pa.device)); pb.copy_(b.to(pb.device))
        for p in params:
            p.requires_grad_(False)
        self.adapter_modules = names
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
        P.LoRALinear.scale = CONDITIONS[cond]

    def adapter_checks(self) -> dict:
        """Scale 0 must reproduce the unadapted model bit for bit (layer 10 o_proj input, 2,048 tokens of book 1342);
        scale 1 must change it; returning to 0 must reproduce it again."""
        self.probe.mode = None
        try:
            self.set_condition("S0")
            s0 = self._reference_input()
            self.set_condition("S1")
            s1 = self._reference_input()
            self.set_condition("S0")
            back = self._reference_input()
        finally:
            self.set_condition("S0")
        out = {"scale0_equals_unadapted": bool(torch.equal(s0, self.unadapted_ref)),
               "scale1_changes": bool((s1 != s0).any()), "scale0_restored": bool(torch.equal(back, s0))}
        out["as_expected"] = all(out.values())
        return out

    def logits(self, h):
        """Glimmer's output path, as in MuseGlimmerForConditionalGeneration.forward: the head in the model's dtype,
        then times output_multiplier, then T * tanh(x / T) with T = final_logit_softcapping; float32 at the end."""
        x = self.head(h)
        if self.multiplier is not None:
            x = x * self.multiplier
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x.float()

    def logits_check(self) -> dict:
        """At scale 0: the head on the text model's last hidden state must match the full model's own logits."""
        self.probe.mode = None
        self.set_condition("S0")
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
        return {"conditions": dict(CONDITIONS), "adapter_sha256": self.adapter_sha,
                "adapter_modules": self.adapter_modules, "softcap": self.softcap,
                "output_multiplier": self.multiplier, "sealed_file": SEALED, "seal_sha256": self.seal_hash,
                "code_sha256": self.code, "platform": self.platform, "gutenberg_id": gid,
                "text_sha256": hashlib.sha256(self.texts[gid].encode("utf-8")).hexdigest() if gid is not None else None,
                "token_ids_sha256": token_sha, "tokenizer": type(self.tok).__name__,
                "direction_seed": 0, "direction_sha256": self.direction_sha,
                "drive": {"form": "amplitude * cos(2 pi t / N + phase) at every token t", "phase_main": PHASE_MAIN},
                "window": [self.t1, self.T], "T": self.T, "layers_measured": self.layer_ids}


def expected_books(spec) -> list:
    return [f"book_{set_name}_{int(gid)}.json" for set_name, books in spec["inputs"]["prompt_sets"].items()
            for gid in books]


def run_state(run_dir: str, spec) -> str:
    """complete: every book file present. stopped: Phase A ended C5 by rule. failed: a failure record and not
    complete. open: anything else (training done or running, waiting for a phase, or killed)."""
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
    """Setup checks, then linearity and power under every scale, on book 1342, at the fixed a, N = 2 and 4096."""
    remaining = (PHASE_A_PASSES + n_books(s.spec) * passes_per_book(s.grid)) * s.spec["budget"]["seconds_per_pass_estimate"]
    s.ledger.check(remaining)
    t_phase = time.time()
    setup_failed = None
    try:
        ad_check = s.adapter_checks()
        print(f"   adapter: {ad_check}", flush=True)
        logit_check = s.logits_check()
        print(f"   perplexity path: max |logit difference| {logit_check['max_abs_diff']:.4g} "
              f"(max |logit| {logit_check['max_abs_logit']:.4g}); as expected: {logit_check['as_expected']}", flush=True)
        failed_checks = ((["adapter"] if not ad_check["as_expected"] else [])
                         + (["perplexity path"] if not logit_check["as_expected"] else []))
        if failed_checks:
            result = {"phase": "A", "finished_utc": now(), "adapter_checks": ad_check, "logits_check": logit_check,
                      "stopped": f"setup checks failed: {failed_checks}", "outcome_if_stopped": "U",
                      "sham_differences": None, "rows": [], "chosen_a": None, "record": s.record(PHASE_A_BOOK, None)}
            write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
            setup_failed = failed_checks
            raise SystemExit(f"C5 is not run, as sealed: setup checks failed: {failed_checks}. Outcome U (inconclusive).")
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
                for scale in (1, 2):
                    out = s.measure(ids, N, A * scale * m, PHASE_MAIN, f"phaseA {cond} x{scale} N={N}")
                    resp.append(float(out["target"].mean()))
                    if scale == 1:
                        snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                ratio = resp[1] / resp[0] if resp[0] > 0 else float("nan")
                row["by_N"][N] = {"response_a": resp[0], "response_2a": resp[1], "ratio": ratio,
                                  "within_10pct": bool(abs(ratio / 2 - 1) <= 0.10), "snr_at_a": snr}
                print(f"   {cond:<5} N={N:<5} ratio={ratio:.4f}  ({s.ledger.last_pass:.1f} s/pass)", flush=True)
            row["linear"] = all(v["within_10pct"] for v in row["by_N"].values())
            row["powered"] = bool(row["by_N"][max(PHASE_A_N)]["snr_at_a"] >= 3)
            rows.append(row)
        s.set_condition("S0")
    except BaseException as e:
        if setup_failed is None:
            print(f"!! Phase A failed; record: {write_failure(out_dir, 'phaseA', e)}")
        raise
    bad = [r["condition"] for r in rows if not (r["linear"] and r["powered"])]
    stopped = f"not linear or underpowered under {bad}" if bad else None
    wall_per_pass = (time.time() - t_phase) / PHASE_A_PASSES
    result = {"phase": "A", "finished_utc": now(), "wall_seconds_per_pass": wall_per_pass,
              "book": PHASE_A_BOOK, "a": A, "adapter_checks": ad_check, "logits_check": logit_check,
              "stopped": stopped, "outcome_if_stopped": "U" if stopped else None, "sham_differences": sham,
              "rows": rows, "chosen_a": None if stopped else A,
              "seconds_per_pass_last": s.ledger.last_pass, "record": s.record(PHASE_A_BOOK, base_sha)}
    write_json(os.path.join(out_dir, "phaseA.json"), result, indent=2)
    projected = n_books(s.spec) * passes_per_book(s.grid) * wall_per_pass / 3600
    print(f"== Phase B projection: {projected:.2f} hours at {wall_per_pass:.1f} wall s per pass; "
          f"{s.ledger.seconds / 3600:.2f} used of {s.spec['budget']['max_gpu_hours']}")
    if stopped:
        raise SystemExit(f"C5 is not run, as sealed: {stopped}. Outcome U (inconclusive).")
    print(f"== Phase A passed under all {len(ORDER)} scales at a = {A}; adapter and perplexity checks as expected")


def check_phase_a(s: Session, out_dir: str) -> dict:
    """Phase B uses Phase A only if it ran under the same seal, code, platform and adapter with a clean sham."""
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
        problems.append(f"Phase A stopped C5 by rule ({pa['stopped']}), so C5 is not run, as sealed")
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
            s.set_condition("S0")
        except BaseException as e:
            print(f"!! Book {gid} failed; record: {write_failure(out_dir, f'book_{set_name}_{gid}', e)}")
            raise
        rec = {"set": set_name, "gutenberg_id": gid, "a": a, "amplitude_reference": refs, "perplexity": ppls,
               "token_ids_sha256": token_sha, "sham_differences": shams, "sham_differences_close": sham_close,
               "seconds": round(time.time() - t_book, 1), "finished_utc": now(),
               "restart_reason": restart_reason if retried else None, "global_layers": s.global_ids,
               "per_condition": per_cond, "record": s.record(gid, token_sha)}
        write_json(path, rec)
        print(f"   done {set_name} {gid} in {rec['seconds']} s; {s.ledger.seconds / 3600:.2f} hours used", flush=True)


def recall_trials(s: Session) -> list:
    """Exploratory passkey trials: filler from each set A book (tokens from RECALL['filler_start'], beyond the
    measurement window), a needle at each depth, then the question. Keys from default_rng(RECALL['key_seed'])."""
    rng = np.random.default_rng(RECALL["key_seed"])
    enc = lambda t: s.tok(t, add_special_tokens=False)["input_ids"]
    trials = []
    for set_name, books in s.spec["inputs"]["prompt_sets"].items():
        for gid in books:
            body = enc(s.texts[int(gid)])
            for depth in RECALL["depths"]:
                key = int(rng.integers(10000, 100000))
                needle, question = enc(RECALL["needle"].format(key=key)), enc(RECALL["question"])
                answer = enc(RECALL["answer"].format(key=key))
                n_fill = RECALL["context_tokens"] - len(needle) - len(question) - len(answer)
                fill = body[RECALL["filler_start"]:RECALL["filler_start"] + n_fill]
                assert len(fill) == n_fill, (gid, len(body))
                at = int(depth * n_fill)
                ids = fill[:at] + needle + fill[at:] + question + answer
                trials.append({"gutenberg_id": int(gid), "depth": depth, "key": key, "ids": ids,
                               "answer_from": len(ids) - len(answer), "answer_len": len(answer)})
    return trials


def token_losses(s: Session, ids: list, start: int, end: int) -> np.ndarray:
    """Next-token loss for tokens start .. end-1 of ids (predicted from positions start-1 .. end-2), float64."""
    x = torch.tensor([ids], device=s.device)
    out = np.empty(end - start)
    with torch.inference_mode():
        h = s.decoder(input_ids=x, use_cache=False).last_hidden_state[0]
        for a in range(start, end, PPL_CHUNK):
            b = min(a + PPL_CHUNK, end)
            lp = s.logits(h[a - 1:b - 1]).log_softmax(-1)
            out[a - start:b - start] = (-lp.gather(1, x[0, a:b, None])[:, 0]).double().cpu().numpy()
    return out


def switch_inputs(s: Session) -> dict:
    """Clean: each set A book's tokens [start, start + context + after). Switch, for every ordered pair (x, y) of
    different set A books: x's first `context` of those tokens, then y's last `after`. Noise: y's clean tokens with the
    first `noise_tokens` after the context replaced by tokens drawn from y's own context tokens, never from the scored
    part, so the noise cannot copy what is predicted later (default_rng(noise_seed))."""
    sw = SWITCH
    L = sw["context"] + sw["after"]
    gids = [int(g) for books in s.spec["inputs"]["prompt_sets"].values() for g in books]
    clean = {}
    for g in gids:
        body = s.tok(s.texts[g], add_special_tokens=False)["input_ids"]
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
    return {"clean": clean, "noise": noise, "switch": switch, "pairs": pairs}


def binned(curve: np.ndarray, offset: int = 0) -> dict:
    """Mean of a per-position curve over SWITCH bins; curve[0] is position k = offset + 1 after the seam. A bin the
    curve covers only in part is labeled with the positions it actually holds; a bin it misses is left out."""
    out = {}
    for lo, hi in SWITCH["bins"]:
        a, b = max(lo - 1 - offset, 0), min(hi - offset, len(curve))
        if b > a:
            out[f"{a + 1 + offset}-{b + offset}"] = float(curve[a:b].mean())
    return out


def phase_r(s: Session, out_dir: str):
    """Exploratory, reported, never gated. Per scale:
    recall: the teacher-forced log-probability of each trial's answer tokens and whether every one is the top choice;
    switch: excess next-token loss after the seam (switch minus clean, same tokens), mean over ordered book pairs;
    noise: excess loss after a burst of noise tokens (noise minus clean), mean over books."""
    trials = recall_trials(s)
    sw_in = switch_inputs(s)
    b = s.spec["budget"]
    s.ledger.check(len(trials) * len(ORDER) * b["recall_seconds_per_trial_estimate"]
                   + (len(sw_in["clean"]) * 2 + len(sw_in["pairs"])) * len(ORDER) * b["switch_seconds_per_pass_estimate"])
    recall_rows, switch_rows, noise_rows = [], [], []
    c, n = SWITCH["context"], SWITCH["noise_tokens"]
    L = c + SWITCH["after"]
    s.probe.mode = None
    s.injector.disarm()
    try:
        for cond in ORDER:
            s.set_condition(cond)
            for t in trials:
                x = torch.tensor([t["ids"]], device=s.device)
                s.ledger.check()
                with torch.inference_mode():
                    h = s.decoder(input_ids=x, use_cache=False).last_hidden_state[0]
                    pos = torch.arange(t["answer_from"] - 1, t["answer_from"] - 1 + t["answer_len"], device=s.device)
                    lp = s.logits(h[pos]).log_softmax(-1)
                    tgt = x[0, pos + 1]
                    logp = float(lp.gather(1, tgt[:, None]).sum())
                    top = bool((lp.argmax(-1) == tgt).all())
                s.ledger.last_pass = s.ledger.add(f"recall {cond} {t['gutenberg_id']} {t['depth']}")
                recall_rows.append({"condition": cond, "gutenberg_id": t["gutenberg_id"], "depth": t["depth"],
                                    "key": t["key"], "answer_logprob": logp, "all_top1": top})
            clean_loss = {}
            for g, ids in sw_in["clean"].items():
                s.ledger.check()
                clean_loss[g] = token_losses(s, ids, c, L)
                s.ledger.last_pass = s.ledger.add(f"clean {cond} {g}")
            excess_sw = []
            for (x, y) in sw_in["pairs"]:
                s.ledger.check()
                excess_sw.append(token_losses(s, sw_in["switch"][(x, y)], c, L) - clean_loss[y])
                s.ledger.last_pass = s.ledger.add(f"switch {cond} {x}->{y}")
            excess_nz = []
            for g, ids in sw_in["noise"].items():
                s.ledger.check()
                excess_nz.append(token_losses(s, ids, c + n, L) - clean_loss[g][n:])
                s.ledger.last_pass = s.ledger.add(f"noise {cond} {g}")
            m_sw, m_nz = np.mean(excess_sw, axis=0), np.mean(excess_nz, axis=0)
            switch_rows.append({"condition": cond, "clean_mean_loss": float(np.mean([v.mean() for v in clean_loss.values()])),
                                "excess_binned": binned(m_sw), "excess_total": float(m_sw.sum()),
                                "excess_curve": m_sw.astype(np.float32).tolist(),
                                "per_pair_total": {f"{x}->{y}": float(e.sum()) for (x, y), e in zip(sw_in["pairs"], excess_sw)}})
            noise_rows.append({"condition": cond, "excess_binned": binned(m_nz, offset=n),
                               "excess_total": float(m_nz.sum()), "excess_curve": m_nz.astype(np.float32).tolist()})
            done = [r for r in recall_rows if r["condition"] == cond]
            print(f"   {cond}: recall log-prob {np.mean([r['answer_logprob'] for r in done]):.3f} "
                  f"(top-1 {sum(r['all_top1'] for r in done)}/{len(done)}); switch excess {m_sw.sum():.1f} nats "
                  f"(first 64: {m_sw[:64].sum():.1f}); noise excess {m_nz.sum():.1f} nats", flush=True)
        s.set_condition("S0")
    except BaseException as e:
        print(f"!! Behavior phase failed; record: {write_failure(out_dir, 'behavior', e)}")
        raise
    write_json(os.path.join(out_dir, "behavior.json"),
               {"finished_utc": now(), "exploratory": True,
                "design": {"recall": dict(RECALL), "switch": {k: v for k, v in SWITCH.items()},
                           "pairs": [f"{x}->{y}" for x, y in sw_in["pairs"]]},
                "recall": recall_rows, "switch": switch_rows, "noise": noise_rows,
                "record": s.record(None, None)}, indent=1)


def admit(phase: str, raw_root: str, out_dir: str, seal_hash: str, spec, restart_reason: str = None) -> list:
    """Run-selection rules from the sealed spec. Raises SystemExit when a run may not proceed. Returns earlier runs.

    Phase T: refused if this run id already started, if any earlier run under the seal has a training.json (one
    training per seal: a trained adapter is never replaced), if any run is complete or was stopped by rule, or if
    any run exists and no written reason is given. Writes STARTED.json.
    Phase A: needs this run's STARTED.json and training.json and no phaseA.json; a retry after a Phase A failure
    record needs a written reason.
    Phase B: needs this run's STARTED.json under the same seal and its phaseA.json; refused if any other run under
    the seal is complete or if a newer run under the seal has started.
    Phase R: needs this run complete and no behavior.json.
    """
    runs = runs_under_seal(raw_root, seal_hash, spec) if os.path.isdir(raw_root) else []
    me = os.path.basename(out_dir)
    others = [r for r in runs if r["run_id"] != me]
    if phase == "T":
        if os.path.exists(os.path.join(out_dir, "STARTED.json")):
            raise SystemExit("This run id already started; raw results are never overwritten.")
        trained = [r["run_id"] for r in others
                   if os.path.exists(os.path.join(raw_root, r["run_id"], "training.json"))]
        if trained:
            raise SystemExit(f"An adapter was already trained under this seal: {trained}. One training per seal; "
                             f"continue that run instead.")
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
    if not os.path.exists(os.path.join(out_dir, "training.json")):
        raise SystemExit(f"No training.json in {out_dir}: phase T has not finished.")
    mine = next(r for r in runs if r["run_id"] == me)
    newer = [r["run_id"] for r in others if (r["started_utc"] or "", r["run_id"]) > (mine["started_utc"] or "", me)]
    if newer:
        raise SystemExit(f"A newer run under this seal has started: {newer}. Only the newest run may continue.")
    if phase == "A":
        if os.path.exists(os.path.join(out_dir, "phaseA.json")):
            raise SystemExit("Phase A already ran for this run.")
        if glob.glob(os.path.join(out_dir, "phaseA.FAILED.*.json")) and not restart_reason:
            raise SystemExit("Phase A failed before in this run. A retry needs --restart-reason (an infrastructure "
                             "failure); the failure record is kept and reported.")
        return others
    if phase == "R":
        if run_state(out_dir, spec) != "complete":
            raise SystemExit("Recall runs only after Phase B is complete.")
        if os.path.exists(os.path.join(out_dir, "behavior.json")):
            raise SystemExit("behavior.json exists.")
        return others
    if not os.path.exists(os.path.join(out_dir, "phaseA.json")):
        raise SystemExit(f"No phaseA.json in {out_dir}.")
    done = [r["run_id"] for r in others if r["state"] == "complete"]
    if done:
        raise SystemExit(f"Another run under this seal is complete: {done}.")
    return others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["T", "A", "B", "R"], required=True)
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
    if args.phase != "T" and not args.run_id:
        raise SystemExit(f"Phase {args.phase} needs --run-id from phase T.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{TAG}-%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(raw_root, run_id)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)              # before admit, so a mismatch leaves no orphan run
    lock = RunLock(ledger_root, seal_hash)
    try:
        admit(args.phase, raw_root, out_dir, seal_hash, spec, args.restart_reason)
        print(f"== Run {run_id}, phase {args.phase}")
        if args.phase == "T":
            ledger = Ledger(ledger_root, seal_hash, spec["budget"]["max_gpu_hours"] * 3600)
            ledger.check(spec["budget"]["training_hours_estimate"] * 3600)
            try:
                train(spec, seal_hash, code, ledger, args, out_dir)
            except BaseException as e:
                print(f"!! Training failed; record: {write_failure(out_dir, 'training', e)}")
                raise
            finally:
                ledger.add("process exit")
            return
        try:
            s = Session(args, ledger_root, out_dir)
        except BaseException as e:
            print(f"!! Startup failed; record: {write_failure(out_dir, 'startup', e)}")
            raise
        try:
            if args.phase == "A":
                phase_a(s, out_dir)
            elif args.phase == "B":
                phase_b(s, out_dir, args.restart_reason)
            else:
                phase_r(s, out_dir)
        finally:
            s.ledger.add("process exit")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
