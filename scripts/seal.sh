#!/usr/bin/env bash
# Seal a draft criteria file. Run this yourself, outside Claude.
#
#   scripts/seal.sh criteria/drafts/c1-resonance.yaml
#
# Copies the draft to criteria/sealed/, makes it read-only, and appends
# "<UTC time> <sha256> <path>" to criteria/SEALS.txt. A sealed name is
# never reused: to change a criterion, seal a new version (c1-resonance-v2).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAFT="${1:?usage: scripts/seal.sh criteria/drafts/<name>.yaml}"
NAME="$(basename "$DRAFT")"
DEST="$ROOT/criteria/sealed/$NAME"
LOG="$ROOT/criteria/SEALS.txt"

[[ -f "$DRAFT" ]] || { echo "No such draft: $DRAFT" >&2; exit 1; }
[[ -e "$DEST" ]] && { echo "Already sealed: criteria/sealed/$NAME. Seal a new version instead." >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  hash_of() { sha256sum "$1" | cut -d' ' -f1; }
else
  hash_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

mkdir -p "$ROOT/criteria/sealed"
cp "$DRAFT" "$DEST"
chmod 444 "$DEST"
HASH="$(hash_of "$DEST")"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

[[ -f "$LOG" ]] && chmod 644 "$LOG"
echo "$STAMP $HASH criteria/sealed/$NAME" >> "$LOG"
chmod 444 "$LOG"

echo "Sealed criteria/sealed/$NAME"
echo "  sha256 $HASH"
echo "  at     $STAMP"
echo "Commit both files now so the seal has a timestamp outside this machine:"
echo "  git add criteria/sealed/$NAME criteria/SEALS.txt && git commit -m \"Seal $NAME\""
