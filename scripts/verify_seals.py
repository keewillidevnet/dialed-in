#!/usr/bin/env python3
"""Check every sealed criteria file against the hash log.

Usage:  python3 scripts/verify_seals.py [--quiet]
Exit 0 when every seal matches, 1 when any file is missing, changed,
or sealed without a log entry. As a SessionStart hook, its stdout is
added to Claude's context, so a broken seal is surfaced at once.
"""
import hashlib
import os
import sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEALED = os.path.join(ROOT, "criteria", "sealed")
LOG = os.path.join(ROOT, "criteria", "SEALS.txt")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_log() -> dict:
    """Return {relative path: latest sha256}. Line format: <utc> <sha256> <path>"""
    entries = {}
    if not os.path.exists(LOG):
        return entries
    with open(LOG, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and not line.startswith("#"):
                entries[parts[2]] = parts[1]
    return entries


def main() -> int:
    quiet = "--quiet" in sys.argv
    log = read_log()
    problems, ok = [], 0

    for rel, digest in log.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            problems.append(f"MISSING  {rel} is in SEALS.txt but the file is gone")
        elif sha256(path) != digest:
            problems.append(f"CHANGED  {rel} no longer matches its sealed hash")
        else:
            ok += 1

    if os.path.isdir(SEALED):
        for name in sorted(os.listdir(SEALED)):
            rel = f"criteria/sealed/{name}"
            if not name.startswith(".") and rel not in log:
                problems.append(f"UNLOGGED {rel} is in sealed/ with no SEALS.txt entry")

    if problems:
        print("SEAL CHECK FAILED. Do not run or report any experiment until Keenan resolves this:")
        for p in problems:
            print(f"  {p}")
        return 1
    if not quiet:
        print(f"Seal check passed: {ok} sealed criteria file(s) match SEALS.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
