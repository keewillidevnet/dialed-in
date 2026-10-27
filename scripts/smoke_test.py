#!/usr/bin/env python3
"""Pre-kickoff smoke and capacity test. Infrastructure only.

No pulses, no sweeps, no measurement of the effect under study. Checks
that both models load and run, that hooks attach to every layer, how
long an input fits in memory, and whether 4-bit loading works on this
GPU. Writes a report to /workspace/env/, never to results/.

    python3 scripts/smoke_test.py
"""
import gc
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

import torch

WEIGHTS = os.environ.get("WEIGHTS_DIR", "/workspace/weights")
OUT_DIR = "/workspace/env"
MODELS = ["Llama-3.1-8B-Instruct", "Llama-3.1-8B"]
EXPECT = {"num_hidden_layers": 32, "num_attention_heads": 32,
          "num_key_value_heads": 8, "hidden_size": 4096}
# Public-domain text (Gettysburg Address, 1863) for the forward-pass check.
TEXT = ("Four score and seven years ago our fathers brought forth on this continent, "
        "a new nation, conceived in Liberty, and dedicated to the proposition that all "
        "men are created equal.")
HOOK_LENGTHS = [512, 2048, 4096, 8192, 16384]
ATTN_LENGTHS = [512, 1024, 2048, 4096]
GIB = 1024 ** 3

report = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "purpose": "pre-kickoff infrastructure check, no experiment"}


def say(msg):
    print(msg, flush=True)


def versions():
    v = {"python": platform.python_version(), "torch": torch.__version__,
         "cuda_runtime": torch.version.cuda}
    for name in ["transformers", "peft", "bitsandbytes", "accelerate", "huggingface_hub"]:
        try:
            v[name] = __import__(name).__version__
        except Exception:
            v[name] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        v.update(gpu=p.name, gpu_mem_gib=round(p.total_memory / GIB, 1),
                 compute_capability=f"{p.major}.{p.minor}")
    return v


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def load(path, four_bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok, round(time.time() - t, 1)


def attach_hooks(model):
    """Pre-hook on each layer's o_proj: its input is the 32 query heads, concatenated."""
    calls, shapes, handles = [], {}, []
    for i, layer in enumerate(model.model.layers):
        def hook(mod, args, i=i):
            calls.append(i)
            shapes[i] = tuple(args[0].shape)
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook))
    return calls, shapes, handles


def peak_run(fn):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        fn()
        torch.cuda.synchronize()
        return {"ok": True, "sec": round(time.time() - t, 2),
                "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 1)}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ok": False, "error": "out of memory"}


def check_model(name):
    path = os.path.join(WEIGHTS, name)
    r = {"path": path}
    model, tok, r["load_sec"] = load(path)
    cfg = model.config
    r["config"] = {k: getattr(cfg, k, None) for k in EXPECT}
    r["config_ok"] = all(r["config"][k] == v for k, v in EXPECT.items())
    r["weights_gib"] = round(torch.cuda.memory_allocated() / GIB, 1)

    calls, shapes, handles = attach_hooks(model)
    ids = tok(TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**ids, output_attentions=True)
    r["tokens"] = ids["input_ids"].shape[1]
    r["logits_finite"] = bool(torch.isfinite(out.logits).all())
    r["hook_calls"] = len(calls)
    r["hooks_ok"] = sorted(set(calls)) == list(range(EXPECT["num_hidden_layers"]))
    r["head_input_shape"] = list(shapes.get(0, ()))
    att = out.attentions
    r["attentions"] = {"layers": len(att), "shape": list(att[0].shape)} if att else None
    r["attentions_ok"] = bool(att) and len(att) == 32 and att[0].shape[1] == 32
    with torch.no_grad():
        nxt = model.generate(**ids, max_new_tokens=8, do_sample=False)
    r["greedy_continuation"] = tok.decode(nxt[0, ids["input_ids"].shape[1]:])
    free(out, att)

    # Capacity: random token ids, timing and memory only.
    g = torch.Generator(device="cuda").manual_seed(0)
    vocab = cfg.vocab_size
    r["capacity_hooks"], r["capacity_attentions"] = {}, {}
    for n in HOOK_LENGTHS:
        x = torch.randint(0, vocab, (1, n), device="cuda", generator=g)
        def f():
            with torch.no_grad():
                model(input_ids=x)
        res = peak_run(f)
        r["capacity_hooks"][n] = res
        say(f"   hooks only, {n:>6} tokens: {res}")
        if not res["ok"]:
            break
    for n in ATTN_LENGTHS:
        x = torch.randint(0, vocab, (1, n), device="cuda", generator=g)
        def f():
            with torch.no_grad():
                model(input_ids=x, output_attentions=True)
        res = peak_run(f)
        r["capacity_attentions"][n] = res
        say(f"   with attentions, {n:>6} tokens: {res}")
        if not res["ok"]:
            break

    for h in handles:
        h.remove()
    free(model)
    return r


def check_nf4(name):
    path = os.path.join(WEIGHTS, name)
    r = {}
    try:
        model, tok, r["load_sec"] = load(path, four_bit=True)
        r["weights_gib"] = round(torch.cuda.memory_allocated() / GIB, 1)
        calls, _, handles = attach_hooks(model)
        ids = tok(TEXT, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids)
        r["logits_finite"] = bool(torch.isfinite(out.logits).all())
        r["hooks_ok"] = sorted(set(calls)) == list(range(32))
        r["ok"] = r["logits_finite"] and r["hooks_ok"]
        for h in handles:
            h.remove()
        free(out, model)
    except Exception as e:
        r["ok"] = False
        r["error"] = f"{type(e).__name__}: {e}"
        r["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    return r


def main():
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run this on the pod.")
    os.makedirs(OUT_DIR, exist_ok=True)
    report["versions"] = versions()
    say(f"== Environment: {json.dumps(report['versions'])}")

    report["models"] = {}
    for name in MODELS:
        say(f"\n== {name} (BF16, eager)")
        try:
            r = check_model(name)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}",
                 "trace_tail": traceback.format_exc().splitlines()[-3:]}
        report["models"][name] = r
        for k in ["load_sec", "weights_gib", "config_ok", "logits_finite",
                  "hook_calls", "hooks_ok", "attentions_ok", "greedy_continuation", "error"]:
            if k in r:
                say(f"   {k}: {r[k]}")

    say("\n== NF4 4-bit load (Instruct)")
    report["nf4"] = check_nf4(MODELS[0])
    say(f"   {report['nf4']}")

    passed = all(m.get("config_ok") and m.get("hooks_ok") and m.get("logits_finite")
                 and m.get("attentions_ok") for m in report["models"].values())
    report["passed"] = bool(passed)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUT_DIR, f"smoke_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    say(f"\n== {'PASS' if passed else 'FAIL'}: BF16 checks on both models. NF4: "
        f"{'works' if report['nf4'].get('ok') else 'failed, see report'}.")
    say(f"Report saved to {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
