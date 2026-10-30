"""Tests for the exploratory behavior probe's pure parts: old-book words, pairs and the showcase rule."""
from dialedin import probe_behavior as B


def test_old_book_words():
    vocab = {1: " Heath", 2: " the", 3: " Al", 4: " Jane", 5: " Mr", 6: " Linton"}
    dec = lambda ids: vocab[ids[0]]
    x = [1] * 5 + [2] * 9 + [3] * 4 + [4] * 4 + [5] * 6 + [6] * 2
    y = [4, 2]
    assert B.old_book_words(x, y, dec) == [1]          # " the" lowercase; " Al" and " Mr" too short; " Jane" occurs in y
    vocab[5] = " Mrs"
    assert B.old_book_words(x, y, dec) == [5, 1]        # most frequent first; " Linton" below MIN_COUNT


def test_pairs_and_showcase():
    assert B.pairs([1, 2, 3]) == [(1, 2), (2, 3), (3, 1)]
    r = lambda s0, s2, r2: {"S0": {"pull": s0}, "S2": {"pull": s2}, "R2": {"pull": r2}}
    rows = {"a": r(0.01, 0.03, 0.011), "b": r(0.01, 0.05, 0.04), "c": r(0.02, 0.01, 0.02)}
    assert B.showcase(rows, "S2", "R2") == "a"          # b's control moves too much; c falls
    assert B.showcase({"c": rows["c"]}, "S2", "R2") is None
