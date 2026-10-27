"""Shared helpers for the protection hooks."""
import fnmatch
import os

FALLBACK_PATTERNS = [
    "criteria/sealed/**",
    "criteria/SEALS.txt",
    "results/raw/**",
    "src/dialedin/metrics.py",
    "scripts/seal.sh",
    ".claude/hooks/**",
    ".claude/settings.json",
    ".claude/protected.txt",
]


def project_root(data: dict) -> str:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.realpath(root)


def load_patterns(root: str) -> list:
    path = os.path.join(root, ".claude", "protected.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
    except OSError:
        # A missing list must fail closed, never open.
        return list(FALLBACK_PATTERNS)
    patterns = [ln for ln in lines if ln and not ln.startswith("#")]
    return patterns or list(FALLBACK_PATTERNS)


def to_relative(target: str, root: str) -> str:
    if not os.path.isabs(target):
        target = os.path.join(root, target)
    real = os.path.realpath(target)
    try:
        rel = os.path.relpath(real, root)
    except ValueError:
        return real.replace("\\", "/")
    return rel.replace("\\", "/")


def pattern_matches(rel: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        base = pattern[:-3]
        return rel == base or rel.startswith(base + "/")
    return fnmatch.fnmatchcase(rel, pattern)


def match_protected(target: str, root: str, patterns: list):
    rel = to_relative(target, root)
    for pattern in patterns:
        if pattern_matches(rel, pattern):
            return rel, pattern
    return None


def path_fragments(patterns: list) -> list:
    """Literal path pieces used to spot protected paths inside shell commands."""
    frags = []
    for p in patterns:
        frag = p[:-3] if p.endswith("/**") else p
        if "*" not in frag and "?" not in frag:
            frags.append(frag)
    return frags
