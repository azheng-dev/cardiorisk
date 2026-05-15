"""Tests for the generation eval scorer."""

from __future__ import annotations

from cardiorisk.rag.eval_generation.loader import EvalCase
from cardiorisk.rag.eval_generation.scorer import (
    aggregate_scores,
    score_case,
)
from cardiorisk.rag.generation.generator import (
    GeneratedAnswer,
    SuppressedClaim,
    VerifiedClaim,
)
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


def _retrieved(chunk_id: str, doc_id: str = "doc") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        strategy="token",
        char_start=0,
        char_end=10,
        page_start=1,
        page_end=1,
        text="ignored",
        n_tokens=2,
    )
    return RetrievedChunk(
        chunk=chunk,
        score=1.0,
        rrf_score=1.0,
        vector_rank=1,
        bm25_rank=1,
        rerank_score=None,
    )


def _case(
    *,
    id_: str = "g001",
    expected_doc_ids: tuple[str, ...] = ("docA",),
    expected_keywords: tuple[str, ...] = ("alpha",),
    should_refuse: bool = False,
    tags: tuple[str, ...] = ("risk_assessment",),
) -> EvalCase:
    return EvalCase(
        id=id_,
        question="Q?",
        expected_doc_ids=expected_doc_ids,
        expected_keywords=expected_keywords,
        rationale="r",
        source_phase="3.3",
        requires_full_corpus=False,
        should_refuse=should_refuse,
        tags=tags,
    )


def _answer(
    *,
    text: str = "alpha and beta. [a:1]",
    is_refusal: bool = False,
    verified: tuple[VerifiedClaim, ...] = (),
    suppressed: tuple[SuppressedClaim, ...] = (),
    retrieved: tuple[RetrievedChunk, ...] = (),
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query="Q?",
        raw_llm_text=text,
        is_refusal=is_refusal,
        verified_claims=verified,
        suppressed_claims=suppressed,
        retrieved=retrieved,
    )


def test_keyword_recall_full_match() -> None:
    answer = _answer(
        verified=(
            VerifiedClaim(
                text="The threshold is 10% with alpha context.",
                headline_chunk_id="a:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("a:1", doc_id="docA"),),
    )
    res = score_case(_case(expected_keywords=("alpha", "10%")), answer)
    assert res.keyword_recall == 1.0
    assert not res.hallucination


def test_keyword_recall_partial_match() -> None:
    answer = _answer(
        verified=(
            VerifiedClaim(
                text="alpha only here.",
                headline_chunk_id="a:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("a:1", doc_id="docA"),),
    )
    res = score_case(_case(expected_keywords=("alpha", "beta")), answer)
    assert res.keyword_recall == 0.5


def test_refusal_case_correct_refusal_full_recall() -> None:
    answer = _answer(is_refusal=True)
    res = score_case(
        _case(
            expected_doc_ids=(),
            expected_keywords=(),
            should_refuse=True,
            tags=("refusal",),
        ),
        answer,
    )
    assert res.keyword_recall == 1.0
    assert res.refused
    assert not res.hallucination


def test_refusal_case_failed_refusal_zero_recall() -> None:
    answer = _answer(
        is_refusal=False,
        verified=(
            VerifiedClaim(
                text="A confident fabrication.",
                headline_chunk_id="x:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("x:1", doc_id="docX"),),
    )
    res = score_case(
        _case(
            expected_doc_ids=(),
            expected_keywords=(),
            should_refuse=True,
            tags=("refusal",),
        ),
        answer,
    )
    assert res.keyword_recall == 0.0
    assert not res.refused


def test_hallucination_when_cited_doc_not_expected() -> None:
    answer = _answer(
        verified=(
            VerifiedClaim(
                text="alpha.",
                headline_chunk_id="x:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("x:1", doc_id="docOTHER"),),
    )
    res = score_case(_case(expected_doc_ids=("docA",)), answer)
    assert res.hallucination


def test_hallucination_false_when_cited_doc_in_expected_set() -> None:
    answer = _answer(
        verified=(
            VerifiedClaim(
                text="alpha.",
                headline_chunk_id="a:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("a:1", doc_id="docA"),),
    )
    res = score_case(_case(expected_doc_ids=("docA",)), answer)
    assert not res.hallucination


def test_aggregate_zero_cases_returns_nan() -> None:
    report = aggregate_scores([])
    assert report.n_cases == 0
    assert report.keyword_recall != report.keyword_recall  # NaN


def test_aggregate_per_tag_split() -> None:
    answer_pos = _answer(
        verified=(
            VerifiedClaim(
                text="alpha.",
                headline_chunk_id="a:1",
                headline_score=0.9,
                supporting_chunk_ids=(),
                supporting_scores=(),
            ),
        ),
        retrieved=(_retrieved("a:1", doc_id="docA"),),
    )
    answer_refuse = _answer(is_refusal=True)
    rows = [
        score_case(_case(id_="g001", tags=("risk_assessment",)), answer_pos),
        score_case(
            _case(
                id_="g002",
                expected_doc_ids=(),
                expected_keywords=(),
                should_refuse=True,
                tags=("refusal",),
            ),
            answer_refuse,
        ),
    ]
    report = aggregate_scores(rows, n_resamples=100)
    assert report.n_cases == 2
    assert report.n_positive == 1
    assert report.n_refusal == 1
    assert "risk_assessment" in report.per_tag
    assert "refusal" in report.per_tag
    assert report.per_tag["refusal"]["refusal_accuracy"] == 1.0


def test_aggregate_bootstrap_ci_is_deterministic() -> None:
    rows = [
        score_case(
            _case(id_=f"g{i:03d}"),
            _answer(
                verified=(
                    VerifiedClaim(
                        text="alpha.",
                        headline_chunk_id="a:1",
                        headline_score=0.9,
                        supporting_chunk_ids=(),
                        supporting_scores=(),
                    ),
                ),
                retrieved=(_retrieved("a:1", doc_id="docA"),),
            ),
        )
        for i in range(1, 11)
    ]
    a = aggregate_scores(rows, n_resamples=200)
    b = aggregate_scores(rows, n_resamples=200)
    assert a.ci_keyword_recall.lower == b.ci_keyword_recall.lower
    assert a.ci_keyword_recall.upper == b.ci_keyword_recall.upper
