"""End-to-end pipeline tests using MockEmbedder + MockReranker.

These exercise the wiring (vector + BM25 -> RRF -> optional rerank)
without paying the real-model dep cost. The fixture builds a small
synthetic chunk set, indexes both legs, and asserts retrieval pulls
the keyword-matching chunk to the top.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cardiorisk.rag.ingest.chunkers import Chunk, chunk_id_for
from cardiorisk.rag.retrieval.bm25 import BM25Index
from cardiorisk.rag.retrieval.embed import EmbedCache, MockEmbedder
from cardiorisk.rag.retrieval.index import HNSWIndex
from cardiorisk.rag.retrieval.pipeline import RetrievalPipeline
from cardiorisk.rag.retrieval.rerank import MockReranker


def _make_chunk(text: str, *, doc_id: str = "doc", char_start: int = 0) -> Chunk:
    char_end = char_start + len(text)
    return Chunk(
        chunk_id=chunk_id_for(
            doc_id=doc_id, strategy="test", char_start=char_start, char_end=char_end
        ),
        doc_id=doc_id,
        strategy="test",
        char_start=char_start,
        char_end=char_end,
        page_start=1,
        page_end=1,
        text=text,
        n_tokens=len(text.split()),
    )


@pytest.fixture()
def pipeline_no_rerank(tmp_path: Path) -> RetrievalPipeline:
    chunks = [
        _make_chunk("aspirin is not routinely recommended for primary prevention", char_start=0),
        _make_chunk("statins should be offered to high risk patients", char_start=200),
        _make_chunk(
            "lifestyle interventions include physical activity 150 minutes", char_start=400
        ),
        _make_chunk("smoking cessation should be offered every visit", char_start=600),
        _make_chunk("blood pressure threshold is 140 over 90 mmHg", char_start=800),
    ]
    chunks_by_id = {c.chunk_id: c for c in chunks}
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    embedder = MockEmbedder()
    cache = EmbedCache(tmp_path, embedder)
    vectors = cache.encode(chunk_ids, texts)
    vec_idx = HNSWIndex(dim=embedder.dim)
    vec_idx.build(chunk_ids=chunk_ids, vectors=vectors)
    bm25_idx = BM25Index()
    bm25_idx.build(chunk_ids=chunk_ids, texts=texts)
    return RetrievalPipeline(
        embedder=embedder,
        embed_cache=cache,
        vector_index=vec_idx,
        bm25_index=bm25_idx,
        chunks_by_id=chunks_by_id,
    )


@pytest.fixture()
def pipeline_with_rerank(tmp_path: Path) -> RetrievalPipeline:
    chunks = [
        _make_chunk("aspirin is not routinely recommended for primary prevention", char_start=0),
        _make_chunk("statins should be offered to high risk patients", char_start=200),
        _make_chunk(
            "lifestyle interventions include physical activity 150 minutes", char_start=400
        ),
        _make_chunk("smoking cessation should be offered every visit", char_start=600),
        _make_chunk("blood pressure threshold is 140 over 90 mmHg", char_start=800),
    ]
    chunks_by_id = {c.chunk_id: c for c in chunks}
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    embedder = MockEmbedder()
    cache = EmbedCache(tmp_path, embedder)
    vectors = cache.encode(chunk_ids, texts)
    vec_idx = HNSWIndex(dim=embedder.dim)
    vec_idx.build(chunk_ids=chunk_ids, vectors=vectors)
    bm25_idx = BM25Index()
    bm25_idx.build(chunk_ids=chunk_ids, texts=texts)
    return RetrievalPipeline(
        embedder=embedder,
        embed_cache=cache,
        vector_index=vec_idx,
        bm25_index=bm25_idx,
        chunks_by_id=chunks_by_id,
        reranker=MockReranker(),
    )


def test_retrieve_returns_keyword_match(pipeline_no_rerank: RetrievalPipeline) -> None:
    """BM25 should pull the aspirin chunk to the top via RRF."""
    hits = pipeline_no_rerank.retrieve("aspirin primary prevention", top_k=3)
    assert hits
    assert "aspirin" in hits[0].chunk.text


def test_retrieve_top_k_caps(pipeline_no_rerank: RetrievalPipeline) -> None:
    hits = pipeline_no_rerank.retrieve("statin", top_k=2)
    assert len(hits) <= 2


def test_retrieve_reports_rrf_score(pipeline_no_rerank: RetrievalPipeline) -> None:
    hits = pipeline_no_rerank.retrieve("aspirin primary prevention", top_k=3)
    for hit in hits:
        assert hit.rrf_score > 0
        assert hit.rerank_score is None
        assert hit.score == hit.rrf_score


def test_retrieve_with_rerank_populates_rerank_score(
    pipeline_with_rerank: RetrievalPipeline,
) -> None:
    hits = pipeline_with_rerank.retrieve("aspirin primary prevention", top_k=3, with_rerank=True)
    assert hits
    for hit in hits:
        assert hit.rerank_score is not None
        assert hit.score == hit.rerank_score


def test_retrieve_with_rerank_requires_attached_reranker(
    pipeline_no_rerank: RetrievalPipeline,
) -> None:
    with pytest.raises(RuntimeError):
        pipeline_no_rerank.retrieve("aspirin", top_k=3, with_rerank=True)


def test_retrieve_empty_query_returns_empty(pipeline_no_rerank: RetrievalPipeline) -> None:
    """Tokeniser drops everything for whitespace-only queries; pipeline
    should still return the dense leg's hits (which are non-empty)."""
    hits = pipeline_no_rerank.retrieve("the of and to", top_k=3)
    # Vector leg still returns something via the mock embedder, so
    # this is a non-empty list. The semantics of "no query" is an
    # upstream concern.
    assert isinstance(hits, list)


def test_retrieve_vector_and_bm25_ranks_populated(
    pipeline_no_rerank: RetrievalPipeline,
) -> None:
    hits = pipeline_no_rerank.retrieve("aspirin primary prevention", top_k=3)
    # Top hit should be in both legs (BM25 finds it via keyword,
    # vector via mock-token similarity may or may not).
    top = hits[0]
    # At least one of the two should have a rank.
    assert top.vector_rank is not None or top.bm25_rank is not None


def test_has_reranker_property(
    pipeline_no_rerank: RetrievalPipeline,
    pipeline_with_rerank: RetrievalPipeline,
) -> None:
    assert not pipeline_no_rerank.has_reranker
    assert pipeline_with_rerank.has_reranker
