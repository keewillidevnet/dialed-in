"""Connect the trained C5 dial to live/server.py, the same way run_c9.py attaches it.

server.py calls load(model) once, after loading the first layers of Glimmer. This reads the
adapter's file, layers, rank, alpha and family from the sealed C9 spec, checks the adapter's
sha256 against it, attaches pilot_c5's LoRA wrappers to the layers that were loaded, copies in the
trained A and B, and returns set_scale(s), which sets LoRALinear.scale (the dial) to s.

Run from the repo root:
    PYTHONPATH=src:. python3 live/server.py --model /workspace/weights/Muse-Glimmer-30B \
        --text /workspace/data/books/<a book>.txt --adapter module:live.dial_shim:load --amplitude 0.1
"""
import glob
import hashlib
import os

import torch
import yaml

from dialedin import pilot_c5 as P

REPO = os.environ.get("DIALEDIN_REPO", ".")


def _sealed_adapter_spec():
    for path in sorted(glob.glob(os.path.join(REPO, "criteria", "sealed", "*c9*"))):
        with open(path) as fh:
            spec = yaml.safe_load(fh)                      # also reads JSON
        if isinstance(spec, dict) and "adapter" in spec:
            print(f"adapter spec from {path}")
            return spec["adapter"]
    raise SystemExit("no sealed C9 spec with an adapter section under criteria/sealed/ (set DIALEDIN_REPO to the repo root)")


def load(model):
    from live.server import find_layers

    ad_spec = _sealed_adapter_spec()
    path = os.path.join(REPO, ad_spec["file"])
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != ad_spec["sha256"]:
        raise SystemExit(f"adapter {path} sha256 {sha} differs from the sealed {ad_spec['sha256']}")
    ad = torch.load(path, map_location="cpu")
    layers = find_layers(model)
    keep = [i for i in ad_spec["layers"] if i < len(layers)]          # only the layers this server loaded
    names, params = P.attach_lora(layers, keep, ad_spec["rank"], ad_spec["alpha"], family=ad_spec["family"])
    saved = dict(zip(ad["modules"], zip(ad["A"], ad["B"])))
    missing = [n for n in names if n not in saved]
    if missing:
        raise SystemExit(f"adapter file has no weights for {missing}")
    with torch.no_grad():
        for n, pa, pb in zip(names, params[0::2], params[1::2]):
            a, b = saved[n]
            pa.copy_(a.to(pa.device)); pb.copy_(b.to(pb.device))
    for p in params:
        p.requires_grad_(False)
    print(f"dial attached: {len(names)} of {len(ad['modules'])} adapter modules (layers {keep}), sha256 checked")

    def set_scale(s):
        P.LoRALinear.scale = float(s)

    set_scale(0.0)
    return set_scale
