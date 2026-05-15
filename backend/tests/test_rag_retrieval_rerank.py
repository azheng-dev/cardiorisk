"""Tests for the cross-encoder reranker wrappers."""

from __future__ import annotations

import pytest

from cardiorisk.rag.retrieval.rerank import (
    BaseReranker,
    MockReranker,
    get_reranker,
)


def test_mock_reranker_satisfies_protocol() -> None:
    rr = MockReranker()
    assert isinstance(rr, BaseReranker)


def test_mock_reranker_empty_passages() -> None:
    rr = MockReranker()
    assert rr.score("query", []) == []


def test_mock_reranker_higher_score_for_more_overlap() -> None:
    rr = MockReranker()
    scores = rr.score(
        "aspirin therapy primary prevention",
        [
            "aspirin therapy primary prevention statement",
            "completely unrelated content",
            "aspirin alone",
        ],
    )
    # Doc 0 has the most overlap with the query.
    assert scores[0] > scores[1]
    assert scores[0] > 0.0


def test_mock_reranker_zero_overlap_returns_zero() -> None:
    rr = MockReranker()
    scores = rr.score("alpha beta gamma", ["delta epsilon"])
    assert scores == [0.0]


def test_get_reranker_factory() -> None:
    assert isinstance(get_reranker("mock"), MockReranker)
    with pytest.raises(ValueError):
        get_reranker("not-a-reranker")


def test_mock_reranker_handles_empty_passage_text() -> None:
    rr = MockReranker()
    scores = rr.score("query", ["", "valid passage"])
    assert scores[0] == 0.0
