"""Checks every number in a Muse narration against the fact sheet, and every citation against fact ids.

A number followed by a citation must match the fact it cites. Any other number counts as verified if it matches a
fact's value at the precision it was written with
(for example 0.137 matches 0.13712), or as a percentage of a fact (3.4% matches 0.0342). Integers
below 10 and identifiers such as C1, N = 4096 or layer 2 are treated as labels when a fact with
that exact value exists, and are otherwise reported too.
"""
import re

NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)(?!\.\d)")
CITE = re.compile(r"\[F(\d+)\]")


def _matches(token: str, values: list) -> bool:
    t = token.replace("−", "-").replace(",", "")
    pct = t.endswith("%")
    t = t.rstrip("%")
    x = float(t)
    decimals = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-12
    for v in values:
        cand = v * 100 if pct else v
        if abs(abs(x) - abs(cand)) <= tol:
            return True
    return False


BOUND = re.compile(r"(?<![\w.])([-+−]?\d+(?:,\d{3})*(?:\.\d+)?%?)\s*(percent\s*)?\[F(\d+)\]")


def check(text: str, sheet) -> dict:
    """A number written right before a citation must match that cited fact (at its written precision, or as a
    percentage of it); every other number must match some fact on the sheet; every citation must exist."""
    by_id = {f["id"]: f for f in sheet.facts}
    mismatched = []
    for num, word, fid in BOUND.findall(text):
        if word and not num.endswith("%"):
            num += "%"                                  # "27 percent" is checked as 27%
        f = by_id.get(f"F{fid}")
        v = f["value"] if f else None
        if f is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not _matches(num, [v])):
            mismatched.append(f"{num} [F{fid}]")
    stripped = BOUND.sub(" ", text)
    stripped = CITE.sub(" ", stripped)
    stripped = re.sub(r"\b[CF]\d+\b|\bv\d+\b|\b\d+B\b", " ", stripped)   # experiment names, versions, model size
    values = sheet.numbers()
    unverified = [tok for tok in NUM.findall(stripped) if not _matches(tok, values)]
    ids = set(by_id)
    cited = {f"F{m}" for m in CITE.findall(text)}
    return {"unverified_numbers": unverified, "mismatched_citations": mismatched, "unknown_citations": sorted(cited - ids),
            "citations_used": sorted(cited & ids, key=lambda x: int(x[1:])),
            "ok": not unverified and not mismatched and not (cited - ids)}
