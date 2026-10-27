#!/usr/bin/env python3
"""Inspect Muse Glimmer-30B before writing hooks. Infrastructure only.

Prints the text config, the module layout of one local and one global
layer, any gate parameters, and how long an input fits in memory.
Writes a report to /workspace/env/.

    python3 inspect_glimmer.py
"""
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import torch

PATH = os.environ.get("GLIMMER_DIR", "/workspace/weights/Muse-Glimmer-30B")
OUT_DIR = "/workspace/env"
LENGTHS = [8192, 16384, 32768, 65536]
GIB = 1024 ** 3
TEXT = ("Four score and seven years ago our fathers brought forth on this continent, "
        "a new nation, conceived in Liberty, and dedicated to the proposition that all "
        "men are created equal.")
CONFIG_KEYS = ["num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
               "hidden_size", "intermediate_size", "sliding_window", "layer_types",
               "max_position_embeddings", "vocab_size", "rope_theta", "model_type"]

report = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "purpose": "pre-build inspection, no experiment"}


def say(msg=""):
    print(msg, flush=True)


def find_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == 52:
            return name, mod
    return None, None


def describe(module, prefix="", depth=0, max_depth=3, out=None):
    out = [] if out is None else out
    for name, child in module.named_children():
        line = f"{'  ' * depth}{prefix}{name}: {type(child).__name__}"
        if isinstance(child, torch.nn.Linear):
            line += f" {tuple(child.weight.shape)}"
        extras = {k: getattr(child, k) for k in dir(child)
                  if any(w in k.lower() for w in ("window", "sliding", "layer_type", "is_local", "is_global"))
                  and not k.startswith("_") and not callable(getattr(child, k, None))}
        if extras:
            line += f" {extras}"
        out.append(line)
        if depth + 1 < max_depth:
            describe(child, "", depth + 1, max_depth, out)
    return out


def main():
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run this on the pod.")
    import transformers
    report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    say(f"== Versions: {report['versions']}")

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(PATH)
    text_cfg = getattr(cfg, "text_config", cfg)
    report["config"] = {k: getattr(text_cfg, k, None) for k in CONFIG_KEYS}
    say(f"== Text config:\n{json.dumps(report['config'], indent=2, default=str)}")

    loader = None
    for cls_name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            t = time.time()
            model = cls.from_pretrained(PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
                                        device_map="cuda").eval()
            loader = cls_name
            report["load_sec"] = round(time.time() - t, 1)
            break
        except Exception as e:
            say(f"   {cls_name} failed: {type(e).__name__}: {str(e)[:200]}")
    if loader is None:
        report["error"] = "no loader worked"
        return finish(False)
    report["loader"] = loader
    report["weights_gib"] = round(torch.cuda.memory_allocated() / GIB, 1)
    say(f"== Loaded with {loader} in {report['load_sec']} s, {report['weights_gib']} GiB")

    layers_name, layers = find_layers(model)
    report["layers_path"] = layers_name
    if layers is None:
        report["error"] = "could not find a 52-layer ModuleList"
        return finish(False)
    say(f"== Decoder layers at: {layers_name}")
    for idx in (0, 3):
        lines = describe(layers[idx])
        report[f"layer_{idx}"] = lines
        say(f"\n== Layer {idx} ({'expected local' if idx == 0 else 'expected global'}):")
        for ln in lines:
            say("   " + ln)

    gate_params = [n for n, _ in layers[0].named_parameters()
                   if "gate" in n.lower() and "mlp" not in n.lower()]
    report["attention_gate_params"] = gate_params
    say(f"\n== Attention gate parameters in layer 0: {gate_params or 'none found by name'}")

    tok = AutoTokenizer.from_pretrained(PATH)
    ids = tok(TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**ids)
        nxt = model.generate(**ids, max_new_tokens=8, do_sample=False)
    report["logits_finite"] = bool(torch.isfinite(out.logits).all())
    report["greedy_continuation"] = tok.decode(nxt[0, ids["input_ids"].shape[1]:])
    say(f"\n== Forward pass: finite={report['logits_finite']}, continuation: {report['greedy_continuation']!r}")
    del out, nxt
    gc.collect(); torch.cuda.empty_cache()

    # Capacity: run the decoder stack only, so no huge logits tensor.
    decoder = model.get_submodule(layers_name.rsplit(".", 1)[0]) if "." in layers_name else model
    report["decoder_class"] = type(decoder).__name__
    report["capacity"] = {}
    say(f"\n== Capacity, decoder only ({type(decoder).__name__}), random tokens:")
    g = torch.Generator(device="cuda").manual_seed(0)
    vocab = report["config"]["vocab_size"] or 200000
    for n in LENGTHS:
        x = torch.randint(0, vocab, (1, n), device="cuda", generator=g)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.no_grad():
                decoder(input_ids=x, use_cache=False)
            torch.cuda.synchronize()
            res = {"ok": True, "sec": round(time.time() - t, 2),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 1)}
        except torch.OutOfMemoryError:
            res = {"ok": False, "error": "out of memory"}
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        report["capacity"][n] = res
        say(f"   {n:>6} tokens: {res}")
        del x
        gc.collect(); torch.cuda.empty_cache()
        if not res["ok"]:
            break
    return finish(True)


def finish(ok):
    report["ok"] = ok
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUT_DIR, f"glimmer_inspect_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    say(f"\n== {'DONE' if ok else 'STOPPED'}. Report saved to {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        report["error"] = traceback.format_exc().splitlines()[-1]
        sys.exit(finish(False))
