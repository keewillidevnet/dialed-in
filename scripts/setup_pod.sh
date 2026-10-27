#!/usr/bin/env bash
# One-time pod setup: verify the Hugging Face token, then download
# Muse Glimmer-30B to the persistent volume. Safe to re-run: finished
# files are skipped. Never put the token in this file; it comes from the
# HF_TOKEN environment variable set from a RunPod secret.
#
#   bash scripts/setup_pod.sh
set -euo pipefail

WEIGHTS="${WEIGHTS_DIR:-/workspace/weights}"
MODELS=("meta-models/Muse-Glimmer-30B")
NEED_GB=70

say() { printf '\n== %s\n' "$*"; }

say "1/4 Checking HF_TOKEN"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. In RunPod, add the environment variable"
  echo "  HF_TOKEN = {{ RUNPOD_SECRET_hf_token }}"
  echo "to the pod, then restart it. For a quick one-off instead, run:"
  echo "  read -rs HF_TOKEN && export HF_TOKEN"
  echo "and paste the token (it will not echo), then re-run this script."
  exit 1
fi
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
python3 - <<'EOF'
from huggingface_hub import whoami
try:
    me = whoami()
except Exception as e:
    raise SystemExit(f"Token rejected by Hugging Face: {e}")
print(f"Signed in as {me['name']}")
EOF

say "2/4 Checking the volume"
mkdir -p "$WEIGHTS"
FREE_GB=$(df -BG --output=avail "$WEIGHTS" | tail -1 | tr -dc '0-9')
echo "$WEIGHTS has ${FREE_GB} GB free"
if [[ "$WEIGHTS" != /workspace/* ]]; then
  echo "Warning: $WEIGHTS is not under /workspace, so it may not survive a pod restart."
fi
(( FREE_GB >= NEED_GB )) || { echo "Need about ${NEED_GB} GB free. Enlarge the volume and retry."; exit 1; }

say "3/4 Downloading (about 60 GB, skips anything already done)"
WEIGHTS="$WEIGHTS" python3 - "${MODELS[@]}" <<'EOF'
import os, sys, datetime
from huggingface_hub import HfApi, snapshot_download

root = os.environ["WEIGHTS"]
api = HfApi()
for repo in sys.argv[1:]:
    name = repo.split("/")[1]
    dest = os.path.join(root, name)
    try:
        sha = api.model_info(repo).sha
    except Exception as e:
        raise SystemExit(f"Cannot reach {repo}. Is your access approved? {e}")
    print(f"-> {repo} @ {sha[:12]}")
    snapshot_download(repo_id=repo, revision=sha, local_dir=dest,
                      ignore_patterns=["original/*"])
    with open(os.path.join(root, "DOWNLOAD_LOG.txt"), "a") as fh:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fh.write(f"{stamp} {repo} {sha} {dest}\n")
EOF

say "4/4 Verifying"
ok=1
for repo in "${MODELS[@]}"; do
  dir="$WEIGHTS/${repo#*/}"
  n=$(ls "$dir"/*.safetensors 2>/dev/null | wc -l)
  size=$(du -sh "$dir" | cut -f1)
  if [[ -f "$dir/config.json" && "$n" -ge 1 ]]; then
    echo "OK   ${repo#*/}: $n safetensors shards, $size"
  else
    echo "FAIL ${repo#*/}: missing config.json or weight shards"; ok=0
  fi
done
(( ok )) || exit 1

echo
echo "Done. The exact revisions are in $WEIGHTS/DOWNLOAD_LOG.txt."
echo "Copy those two lines into prep-log.md, then stop the pod."
