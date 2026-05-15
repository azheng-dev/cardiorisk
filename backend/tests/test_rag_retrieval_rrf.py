"""Tests for RRF (Reciprocal Rank Fusion).

Closed-form sanity: given two identical rankings, RRF should agree
with each on the top item; given two disjoint rankings, RRF should
favour items that appear in both.
"""

from __future__ import annotations

import math

import pytest

from cardiorisk.rag.retrieval.rrf import DEFAULT_K, rrf_fuse


def test_default_k_is_60() -> None:
    assert DEFAULT_K == 60


def test_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a", "b"]], k=0)


def test_empty_rankings() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_single_ranker_preserves_order() -> None:
    out = rrf_fuse([["a", "b", "c"]])
    assert [chunk_id for chunk_id, _ in out] == ["a", "b", "c"]
    # Score for rank-1 = 1 / (60 + 1)
    assert math.isclose(out[0][1], 1.0 / 61.0, rel_tol=1e-9)


def test_two_identical_rankings_double_score() -> None:
    out = rrf_fuse([["a", "b"], ["a", "b"]])
    expected_a = 2.0 / 61.0
    expected_b = 2.0 / 62.0
    assert out[0][0] == "a"
    assert math.isclose(out[0][1], expected_a, rel_tol=1e-9)
    assert math.isclose(out[1][1], expected_b, rel_tol=1e-9)


def test_disjoint_rankings_keep_both() -> None:
    out = rrf_fuse([["a"], ["b"]])
    ids = {chunk_id for chunk_id, _ in out}
    assert ids == {"a", "b"}


def test_overlap_wins_over_disjoint() -> None:
    """If 'b' appears in both rankings (ranks 2 and 1), it should
    beat 'a' even though 'a' is rank 1 in one ranking."""
    out = rrf_fuse([["a", "b", "c"], ["b", "x", "y"]])
    # 'b' score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # 'a' score = 1/(60+1) = 1/61
    assert out[0][0] == "b"
    assert out[1][0] in {"a", "x"}  # both at rank 2


def test_tiebreak_alphabetical() -> None:
    """Two items at the same score should sort alphabetically for stability."""
    # 'a' and 'b' each appear at rank 1 in one ranker only -> equal score.
    out = rrf_fuse([["a"], ["b"]])
    assert [chunk_id for chunk_id, _ in out] == ["a", "b"]


def test_three_rankers_sum() -> None:
    out = rrf_fuse([["a"], ["a"], ["a"]])
    assert math.isclose(out[0][1], 3.0 / 61.0, rel_tol=1e-9)


def test_k_changes_score_but_not_order() -> None:
    a = rrf_fuse([["a", "b", "c"], ["c", "a", "b"]], k=10)
    b = rrf_fuse([["a", "b", "c"], ["c", "a", "b"]], k=200)
    # The ordering is invariant to k; the absolute scores aren't.
    assert [x[0] for x in a] == [x[0] for x in b]
