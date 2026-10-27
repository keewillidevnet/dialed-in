#!/usr/bin/env python3
"""PreToolUse guard for Edit, MultiEdit, Write and NotebookEdit.

Blocks any edit whose target matches a pattern in .claude/protected.txt.
Exit 2 blocks the tool call and sends stderr to Claude as feedback.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protected import load_patterns, match_protected, project_root  # noqa: E402


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("Blocked: protect_files could not read the hook input.", file=sys.stderr)
        return 2

    tool_input = data.get("tool_input") or {}
    target = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not target:
        return 0

    root = project_root(data)
    hit = match_protected(target, root, load_patterns(root))
    if hit:
        print(
            f"Blocked: {hit[0]} is protected (pattern '{hit[1]}'). "
            "Sealed criteria, raw results, the metrics module and the hooks are read-only for Claude. "
            "Propose the change in chat for Keenan to make, or write a new draft in criteria/drafts/.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
