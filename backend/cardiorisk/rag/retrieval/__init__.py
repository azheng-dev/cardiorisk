"""Phase 3.2 hybrid retrieval layer (binding decision: ADR-016).

Builds the dense + sparse hybrid retriever on top of the Phase 3.1
manifest contract. The :class:`RetrievalPipeline` orchestrates a
vector leg (HNSW over BGE-M3 embeddings), a sparse leg (rank_bm25),
RRF fusion, and an optional cross-encoder reranker. Phase 3.3's
citation-mandatory generator and Phase 6's end-to-end harness both
consume :meth:`RetrievalPipeline.retrieve`.

Module map:

- :mod:`.embed` — :class:`BaseEmbedder` Protocol + concrete
  :class:`BGEM3Embedder`, :class:`MiniLMEmbedder` (CI-smoke
  stand-in), :class:`MockEmbedder` (deterministic, dep-free, used by
  unit tests). Disk-backed cache keyed by ``(model_name, chunk_id)``.
- :mod:`.index` — :class:`HNSWIndex` thin wrapper around
  :mod:`hnswlib` (cosine, ``M=16``, ``ef_construction=200``). Save /
  load round-trip mirrors ADR-010's local-artefact contract. Phase 4
  swaps this for a ``PgVectorIndex`` behind the same interface.
- :mod:`.bm25` — :class:`BM25Index` thin wrapper around
  :mod:`rank_bm25` (``BM25Okapi``). Whitespace + lowercase + stopword
  filter via a small vendored English list (no NLTK runtime download).
- :mod:`.rrf` — pure-Python :func:`rrf_fuse` (Reciprocal Rank Fusion,
  Cormack et al. 2009) with the published ``k=60`` default.
- :mod:`.rerank` — :class:`BGEReranker` wrapper around
  :class:`FlagEmbedding.FlagReranker` plus :class:`MockReranker`
  for tests.
- :mod:`.pipeline` — :class:`RetrievalPipeline` orchestrator.

The :data:`DEFAULT_CHUNKER` constant is set at the end of Phase 3.2
once the retrieval eval picks a winner; until then it stays
``"hybrid"`` as the most plausible default for downstream-code
defaults that don't want to wait for the eval.
"""

from .bm25 import BM25Index
from .embed import (
    BaseEmbedder,
    BGEM3Embedder,
    MiniLMEmbedder,
    MockEmbedder,
)
from .index import HNSWIndex
from .pipeline import RetrievalPipeline, RetrievedChunk
from .rerank import BaseReranker, BGEReranker, MockReranker
from .rrf import rrf_fuse

#: Default chunker name used by downstream code that doesn't want to
#: wait for the Phase-3.2 retrieval eval to commit a winner. Updated
#: to the empirical winner once ``reports/v1/retrieval/aggregate.json``
#: exists.
DEFAULT_CHUNKER: str = "hybrid"

__all__ = [
    "DEFAULT_CHUNKER",
    "BGEM3Embedder",
    "BGEReranker",
    "BM25Index",
    "BaseEmbedder",
    "BaseReranker",
    "HNSWIndex",
    "MiniLMEmbedder",
    "MockEmbedder",
    "MockReranker",
    "RetrievalPipeline",
    "RetrievedChunk",
    "rrf_fuse",
]
