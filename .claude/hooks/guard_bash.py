#!/usr/bin/env python3
"""PreToolUse guard for Bash.

Blocks shell commands that would change a protected path, and blocks
any attempt by Claude to run the sealing script. Reads are allowed.
This is best-effort: chmod 444 on sealed files and the hash check in
verify_seals.py are the backstop.
"""
import json
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protected import load_patterns, path_fragments, pattern_matches, project_root  # noqa: E402

SEPARATORS = re.compile(r"&&|\|\||;|\||\n|\$\(|`|\)")
ALWAYS_MUTATING = {"rm", "rmdir", "unlink", "shred", "truncate", "touch", "chmod",
                   "chown", "chattr", "ln", "mv", "chflags"}
COPYING = {"cp", "rsync", "install", "scp"}
IN_PLACE = {"sed", "perl", "gsed"}
GIT_MUTATING = {"checkout", "restore", "rm", "mv", "clean", "apply", "stash"}
INTERPRETERS = {"python", "python3", "node", "ruby", "perl"}


def block(msg: str) -> int:
    print(f"Blocked: {msg}", file=sys.stderr)
    return 2


class Checker:
    def __init__(self, root: str, cwd: str, patterns: list):
        self.root, self.cwd, self.patterns = root, cwd, patterns
        self.frags = path_fragments(patterns)

    def is_protected(self, token: str) -> bool:
        token = token.strip("'\"")
        if not token or token.startswith("-"):
            return False
        if any(c in token for c in "*?["):
            return any(f in token for f in self.frags)
        path = token if os.path.isabs(token) else os.path.join(self.cwd, token)
        rel = os.path.relpath(os.path.realpath(path), self.root).replace("\\", "/")
        return any(pattern_matches(rel, p) for p in self.patterns)

    def check_segment(self, seg: str):
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        if not toks:
            return None

        # Redirections anywhere in the segment: > file, >> file, >file, 2>file
        for i, t in enumerate(toks):
            m = re.match(r"^\d*>>?(.*)$", t)
            if m:
                target = m.group(1) or (toks[i + 1] if i + 1 < len(toks) else "")
                if target and not target.startswith("&") and self.is_protected(target):
                    return f"redirect writes to protected path {target}"

        while toks and (re.match(r"^\w+=", toks[0]) or toks[0] in {"sudo", "env", "command", "nohup", "time"}):
            toks = toks[1:]
        if not toks:
            return None
        cmd, args = os.path.basename(toks[0]), toks[1:]
        plain = [a for a in args if not a.startswith("-")]

        if cmd == "seal.sh" or (cmd in {"bash", "sh", "zsh", "source", "."} and plain
                                and os.path.basename(plain[0]) == "seal.sh"):
            return "SEAL"

        if cmd == "tee":
            hit = next((a for a in plain if self.is_protected(a)), None)
            return f"tee writes to protected path {hit}" if hit else None
        if cmd in ALWAYS_MUTATING:
            hit = next((a for a in plain if self.is_protected(a)), None)
            return f"{cmd} would change protected path {hit}" if hit else None
        if cmd in COPYING and plain:
            dest = plain[-1]
            return f"{cmd} would overwrite protected path {dest}" if self.is_protected(dest) else None
        if cmd in IN_PLACE and any(a.startswith("-i") or a == "--in-place" for a in args):
            hit = next((a for a in plain if self.is_protected(a)), None)
            return f"{cmd} -i would edit protected path {hit}" if hit else None
        if cmd == "dd":
            hit = next((a[3:] for a in args if a.startswith("of=") and self.is_protected(a[3:])), None)
            return f"dd would write protected path {hit}" if hit else None
        if cmd == "git" and args:
            sub = args[0]
            if sub in GIT_MUTATING:
                hit = next((a for a in args[1:] if self.is_protected(a)), None)
                return f"git {sub} would change protected path {hit}" if hit else None
        if cmd in INTERPRETERS:
            body = " ".join(args)
            if any(f in body for f in self.frags) and re.search(r"open\(|write|unlink|remove|rename|chmod|truncate", body):
                return "inline script appears to write a protected path"
        return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return block("guard_bash could not read the hook input.")

    command = (data.get("tool_input") or {}).get("command") or ""
    root = project_root(data)
    cwd = os.path.realpath(data.get("cwd") or root)
    checker = Checker(root, cwd, load_patterns(root))
    for seg in SEPARATORS.split(command):
        reason = checker.check_segment(seg)
        if reason == "SEAL":
            return block("only Keenan runs scripts/seal.sh. Ask him to seal the draft in his own terminal.")
        if reason:
            return block(
                f"{reason}. Protected paths are read-only for Claude; "
                "propose the change in chat, or write a new draft in criteria/drafts/."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
