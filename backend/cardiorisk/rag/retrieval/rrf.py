"""Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

RRF is the score-scale-free fusion algorithm we use to combine the
dense (HNSW) and sparse (BM25) ranking lists. Per ADR-016 §3, the
choice over weighted-sum / softmax fusion is driven by:

1. Dense cosine and BM25 scores live on completely different scales;
   weighted-sum fusion would need per-leg normalisation (brittle as
   the corpus changes) or a learned weight (out of Phase 3.2 budget).
2. RRF has one knob (``k``); the published default 60 is the
   industry standard and works without tuning.

Formula for document ``d`` across rankers ``R``:

.. math::
    \\mathrm{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + \\mathrm{rank}_r(d)}

where ``rank_r(d)`` is the 1-indexed position of ``d`` in ranker
``r``'s ranking. Documents missing from a ranker contribute zero
from that ranker (i.e. they are not penalised explicitly; the
missing-ranker contribution is implicitly absorbed into the lower
total).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

#: Published default `k` from Cormack et al., 2009.
DEFAULT_K: Final[int] = 60


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists by Reciprocal Rank Fusion.

    Args:
        rankings: A sequence of rankings; each ranking is an ordered
            sequence of chunk_ids from best (rank 1) to worst.
        k: RRF smoothing constant. Default 60 (the published value).

    Returns:
        List of ``(chunk_id, fused_score)`` sorted descending by
        ``fused_score``. Ties are broken alphabetically by chunk_id
        for stability across runs.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
