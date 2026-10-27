"""Load Muse Glimmer-30B and run the preflight checks every sealed run requires."""
import hashlib
import json
import os

import torch

EXPECTED_GLOBAL = list(range(3, 52, 4))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_seal(repo: str, sealed_rel: str) -> str:
    """The sealed file's hash must equal its latest entry in criteria/SEALS.txt."""
    log = {}
    with open(os.path.join(repo, "criteria", "SEALS.txt")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                log[parts[2]] = parts[1]
    actual = sha256_file(os.path.join(repo, sealed_rel))
    if log.get(sealed_rel) != actual:
        raise SystemExit(f"Seal check failed for {sealed_rel}: logged {log.get(sealed_rel)}, file {actual}")
    return actual


def check_code(repo: str, spec: dict) -> dict:
    """Every file listed under analysis.code_sha256 must match the sealed hash. Returns {path: sha256}."""
    got = {}
    for rel, want in spec["analysis"]["code_sha256"].items():
        h = sha256_file(os.path.join(repo, rel))
        if h != want:
            raise SystemExit(f"{rel} differs from the sealed code ({h} != {want}); refusing to run.")
        got[rel] = h
    return got


def load_texts(spec: dict, manifest_path: str, books_dir: str) -> dict:
    """Returns {gutenberg_id: text}, refusing any text whose hash differs from the sealed value."""
    with open(manifest_path) as fh:
        manifest = {b["gutenberg_id"]: b for b in json.load(fh)["books"]}
    texts = {}
    for gid, want in spec["inputs"]["text_sha256"].items():
        path = os.path.join(books_dir, manifest[int(gid)]["file"])
        body = open(path, encoding="utf-8").read()
        got = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if got != want:
            raise SystemExit(f"Text hash mismatch for book {gid}: {got} != sealed {want}")
        texts[int(gid)] = body
    if len(texts) != 15:
        raise SystemExit(f"Expected 15 sealed books, found {len(texts)}")
    return texts


def configure_determinism():
    """Any op without a deterministic implementation raises instead of warning."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load(spec: dict, weights_dir: str):
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    plat = spec["platform"]
    model = AutoModelForMultimodalLM.from_pretrained(
        weights_dir, dtype=torch.bfloat16, attn_implementation=plat["attention"], device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(weights_dir)
    layers = model.get_submodule("model.language_model.layers")
    decoder = model.get_submodule("model.language_model")
    record = preflight(spec, model, layers, weights_dir, transformers.__version__)
    return model, decoder, layers, tok, record


def preflight(spec: dict, model, layers, weights_dir: str, tf_version: str) -> dict:
    plat = spec["platform"]
    problems = []
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.bfloat16}:
        problems.append(f"dtype_bf16: found {dtypes}")
    if getattr(model.config, "quantization_config", None) is not None:
        problems.append("no_quantization: quantization_config present")
    attn = getattr(model.config, "_attn_implementation", None)
    text_attn = getattr(getattr(model.config, "text_config", model.config), "_attn_implementation", attn)
    if text_attn != plat["attention"]:
        problems.append(f"sdpa: text attention is {text_attn}")
    log_path = os.path.join(os.path.dirname(weights_dir.rstrip("/")), "DOWNLOAD_LOG.txt")
    revision = None
    if os.path.exists(log_path):
        for line in open(log_path):
            parts = line.split()                     # <utc> <repo> <sha> <dest>
            if len(parts) >= 3 and parts[1] == plat["model"]:
                revision = parts[2]
    if not revision or not revision.startswith(plat["revision"]):
        problems.append(f"revision_match: download log has {revision}")
    if tf_version != plat["transformers"]:
        problems.append(f"versions_match: transformers {tf_version} != {plat['transformers']}")
    if torch.__version__ != plat["torch"]:
        problems.append(f"versions_match: torch {torch.__version__} != {plat['torch']}")
    gpu = torch.cuda.get_device_name(0)
    if gpu != plat["gpu"]:
        problems.append(f"gpu_match: {gpu}")
    global_ids = [i for i, l in enumerate(layers) if not l.self_attn.is_local_attention]
    if global_ids != EXPECTED_GLOBAL:
        problems.append(f"layer layout: global layers {global_ids}")
    if problems:
        raise SystemExit("Preflight failed:\n  " + "\n  ".join(problems))
    return {"dtype": "bfloat16", "attention": text_attn, "revision": revision, "transformers": tf_version,
            "torch": torch.__version__, "gpu": gpu, "global_layers": global_ids, "batch_size": 1,
            "deterministic": True, "tf32": False}
