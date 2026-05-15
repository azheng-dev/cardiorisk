"""Tests for the retrieval-eval scorer."""

from __future__ import annotations

import math

from cardiorisk.rag.eval_retrieval.loader import EvalQuestion
from cardiorisk.rag.eval_retrieval.scorer import (
    aggregate_scores,
    score_question,
)
from cardiorisk.rag.ingest.chunkers import Chunk, chunk_id_for
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


def _q(
    *,
    id_: str = "q001",
    doc_id: str = "doc",
    page_range: tuple[int, int] = (1, 1),
    keywords: tuple[str, ...] = ("foo",),
    tags: tuple[str, ...] = ("risk_assessment",),
    expected_no_hit: bool = False,
) -> EvalQuestion:
    return EvalQuestion(
        id=id_,
        question="Q",
        expected_doc_id=doc_id,
        expected_page_range=page_range,
        expected_span_keywords=keywords,
        rationale="r",
        source_phase="3.2",
        requires_full_corpus=False,
        expected_no_hit=expected_no_hit,
        tags=tags,
    )


def _chunk(
    text: str,
    *,
    doc_id: str = "doc",
    page_start: int = 1,
    page_end: int = 1,
    char_start: int = 0,
) -> RetrievedChunk:
    char_end = char_start + len(text)
    c = Chunk(
        chunk_id=chunk_id_for(
            doc_id=doc_id, strategy="t", char_start=char_start, char_end=char_end
        ),
        doc_id=doc_id,
        strategy="t",
        char_start=char_start,
        char_end=char_end,
        page_start=page_start,
        page_end=page_end,
        text=text,
        n_tokens=len(text.split()),
    )
    return RetrievedChunk(
        chunk=c, score=1.0, rrf_score=0.5, vector_rank=1, bm25_rank=1, rerank_score=None
    )


def test_perfect_top_1_hit() -> None:
    q = _q(keywords=("foo",))
    res = score_question(q, [_chunk("this contains foo word")], top_k=5)
    assert res.hit_at_1 is True
    assert res.hit_at_5 is True
    assert res.rank_of_first_hit == 1


def test_doc_id_mismatch_is_miss() -> None:
    q = _q(doc_id="doc-a")
    res = score_question(q, [_chunk("foo", doc_id="doc-b")], top_k=5)
    assert res.hit_at_5 is False


def test_page_range_outside_is_miss() -> None:
    q = _q(page_range=(5, 10))
    res = score_question(q, [_chunk("foo", page_start=1, page_end=2)], top_k=5)
    assert res.hit_at_5 is False


def test_missing_keyword_is_miss() -> None:
    q = _q(keywords=("foo", "bar"))
    res = score_question(q, [_chunk("foo only")], top_k=5)
    assert res.hit_at_5 is False


def test_hit_at_5_but_not_hit_at_1() -> None:
    q = _q(keywords=("foo",))
    retrieved = [
        _chunk("decoy without keyword"),
        _chunk("decoy without keyword either", char_start=100),
        _chunk("contains foo here", char_start=200),
    ]
    res = score_question(q, retrieved, top_k=5)
    assert res.hit_at_1 is False
    assert res.hit_at_5 is True
    assert res.rank_of_first_hit == 3


def test_negative_case_no_match_is_hit() -> None:
    """expected_no_hit=true: no top-k chunk contains keywords -> hit."""
    q = _q(expected_no_hit=True, keywords=("metformin",))
    retrieved = [_chunk("aspirin therapy"), _chunk("statin therapy", char_start=100)]
    res = score_question(q, retrieved, top_k=5)
    assert res.hit_at_1 is True
    assert res.hit_at_5 is True
    assert res.rank_of_first_hit == 1


def test_negative_case_with_match_is_miss() -> None:
    """expected_no_hit=true but a chunk DOES contain the keyword -> miss."""
    q = _q(expected_no_hit=True, keywords=("metformin",))
    retrieved = [_chunk("metformin therapy mentioned")]
    res = score_question(q, retrieved, top_k=5)
    assert res.hit_at_1 is False
    assert res.hit_at_5 is False


def test_aggregate_scores_perfect() -> None:
    q = _q()
    res = score_question(q, [_chunk("foo")], top_k=5)
    rep = aggregate_scores([res], n_resamples=200)
    assert rep.hit_at_1 == 1.0
    assert rep.hit_at_5 == 1.0
    assert rep.mrr == 1.0
    assert rep.n_questions == 1


def test_aggregate_scores_per_tag() -> None:
    q1 = _q(id_="q001", tags=("risk_assessment",))
    q2 = _q(id_="q002", tags=("pharmacotherapy",))
    res1 = score_question(q1, [_chunk("foo")], top_k=5)
    res2 = score_question(q2, [_chunk("no match here")], top_k=5)
    rep = aggregate_scores([res1, res2], n_resamples=200)
    assert rep.per_tag["risk_assessment"]["hit_at_5"] == 1.0
    assert rep.per_tag["pharmacotherapy"]["hit_at_5"] == 0.0


def test_aggregate_scores_bootstrap_ci_within_bounds() -> None:
    """All metrics ∈ [0, 1]; CIs likewise."""
    qs = [_q(id_=f"q{i:03d}") for i in range(10)]
    results = []
    for i, q in enumerate(qs):
        c = _chunk("foo") if i < 7 else _chunk("no match")
        results.append(score_question(q, [c], top_k=5))
    rep = aggregate_scores(results, n_resamples=500)
    assert 0.0 <= rep.hit_at_1 <= 1.0
    assert 0.0 <= rep.ci_hit_at_5.lower <= rep.hit_at_5 <= rep.ci_hit_at_5.upper <= 1.0


def test_aggregate_scores_empty_input() -> None:
    rep = aggregate_scores([], n_resamples=100)
    assert rep.n_questions == 0
    assert math.isnan(rep.hit_at_5)


def test_mrr_is_one_over_rank_of_first_hit() -> None:
    q = _q(keywords=("foo",))
    retrieved = [_chunk("decoy"), _chunk("foo here", char_start=100)]
    res = score_question(q, retrieved, top_k=5)
    rep = aggregate_scores([res], n_resamples=200)
    assert math.isclose(rep.mrr, 0.5, rel_tol=1e-9)


def test_aggregate_deterministic() -> None:
    q = _q()
    res = score_question(q, [_chunk("foo")], top_k=5)
    a = aggregate_scores([res], n_resamples=500, seed=42)
    b = aggregate_scores([res], n_resamples=500, seed=42)
    assert a.ci_hit_at_5.lower == b.ci_hit_at_5.lower
    assert a.ci_hit_at_5.upper == b.ci_hit_at_5.upper
