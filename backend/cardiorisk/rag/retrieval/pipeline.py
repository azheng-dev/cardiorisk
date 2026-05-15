"""Hybrid retrieval pipeline: vector + BM25 -> RRF -> optional rerank.

The Phase 3.3 citation generator and the Phase 6 eval harness call
:meth:`RetrievalPipeline.retrieve` directly. The pipeline carries
the raw ``Chunk`` text alongside the score breakdown so the citation
layer can render the retrieved passage without re-loading the
chunks JSONL.

Default knobs (per ADR-016 §3, §4):

- Per-leg ``top_k = 50`` candidates → RRF fuse → final ``top_k``.
- ``with_rerank=False`` by default; pass ``True`` from agentic nodes
  that need precision-at-1 (Phase 3.3 citation generation flips this
  on once the eval picks the winning cell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..ingest.chunkers import Chunk
from .bm25 import BM25Index
from .embed import BaseEmbedder, EmbedCache
from .index import HNSWIndex
from .rerank import BaseReranker
from .rrf import DEFAULT_K, rrf_fuse

#: Per-leg fan-out: each leg returns this many candidates before RRF.
DEFAULT_PER_LEG_K: Final[int] = 50


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieval result with score breakdown.

    Attributes:
        chunk: The full :class:`Chunk` object (text + metadata).
        score: Final score (RRF-fused if no rerank; reranker score if reranked).
        rrf_score: RRF-fused score (always populated).
        vector_rank: 1-indexed rank in the dense leg, or ``None`` if missing.
        bm25_rank: 1-indexed rank in the sparse leg, or ``None`` if missing.
        rerank_score: Reranker score, or ``None`` if no rerank stage ran.
    """

    chunk: Chunk
    score: float
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None


class RetrievalPipeline:
    """Hybrid retrieval over a single chunker strategy.

    One pipeline instance per strategy; the eval orchestrator builds
    six (3 chunkers x {with, without rerank}) but they all share the
    embedder + reranker references so weights are loaded once.
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        embed_cache: EmbedCache,
        vector_index: HNSWIndex,
        bm25_index: BM25Index,
        chunks_by_id: dict[str, Chunk],
        reranker: BaseReranker | None = None,
        per_leg_k: int = DEFAULT_PER_LEG_K,
        rrf_k: int = DEFAULT_K,
    ) -> None:
        self._embedder = embedder
        self._embed_cache = embed_cache
        self._vector_index = vector_index
        self._bm25_index = bm25_index
        self._chunks_by_id = chunks_by_id
        self._reranker = reranker
        self._per_leg_k = per_leg_k
        self._rrf_k = rrf_k

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        with_rerank: bool = False,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve the top-``top_k`` chunks for ``query``.

        Args:
            query: Natural-language question.
            top_k: Final number of chunks to return after fusion + rerank.
            with_rerank: If true, run the cross-encoder over the RRF-fused
                candidate set and re-sort by reranker score.

        Returns:
            ``top_k`` (or fewer if the index is small) :class:`RetrievedChunk`
            ordered by final score descending.
        """
        if with_rerank and self._reranker is None:
            raise RuntimeError("with_rerank=True but no reranker is attached")

        # 1. Dense leg.
        q_vec = self._embed_cache.encode_query(query)
        vec_hits = self._vector_index.search(q_vec, top_k=self._per_leg_k)
        vec_rank: dict[str, int] = {h.chunk_id: i + 1 for i, h in enumerate(vec_hits)}

        # 2. Sparse leg.
        bm_hits = self._bm25_index.search(query, top_k=self._per_leg_k)
        bm_rank: dict[str, int] = {h.chunk_id: i + 1 for i, h in enumerate(bm_hits)}

        # 3. RRF fuse.
        fused = rrf_fuse(
            [
                [h.chunk_id for h in vec_hits],
                [h.chunk_id for h in bm_hits],
            ],
            k=self._rrf_k,
        )
        if not fused:
            return []
        rrf_scores: dict[str, float] = dict(fused)
        candidate_ids = [chunk_id for chunk_id, _ in fused]

        # 4. Optional rerank — runs over the full RRF candidate pool
        # (which is up to per_leg_k * 2 unique ids) and re-sorts.
        rerank_scores: dict[str, float] | None = None
        if with_rerank:
            if self._reranker is None:  # defensive; the head-of-method check
                raise RuntimeError("with_rerank=True but no reranker is attached")
            passages = [self._chunks_by_id[cid].text for cid in candidate_ids]
            scores = self._reranker.score(query, passages)
            rerank_scores = dict(zip(candidate_ids, scores, strict=True))
            candidate_ids = sorted(
                candidate_ids,
                key=lambda cid: (-(rerank_scores[cid] if rerank_scores else 0.0), cid),
            )

        # 5. Materialise top_k.
        out: list[RetrievedChunk] = []
        for cid in candidate_ids[:top_k]:
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                continue
            final_score = rerank_scores[cid] if rerank_scores is not None else rrf_scores[cid]
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(final_score),
                    rrf_score=float(rrf_scores[cid]),
                    vector_rank=vec_rank.get(cid),
                    bm25_rank=bm_rank.get(cid),
                    rerank_score=rerank_scores[cid] if rerank_scores is not None else None,
                )
            )
        return out
