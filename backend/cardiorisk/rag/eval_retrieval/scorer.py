"""Hit@k / MRR scoring for the retrieval eval set.

Hit definition (mirrors ``eval/retrieval/README.md``):

- **Standard Q.** A chunk counts as a hit when:

  1. Its ``doc_id`` equals ``expected_doc_id``.
  2. Its ``[page_start, page_end]`` overlaps ``expected_page_range``.
  3. Its text contains every entry in ``expected_span_keywords``
     (case-insensitive substring match, whitespace collapsed).

- **Negative-case Q (``expected_no_hit: true``).** Inverted: a "hit"
  means **no** top-k chunk contained all keywords. ``doc_id`` and
  ``page_range`` are ignored.

Aggregation: hit@1, hit@5, MRR computed per (cell, question), then
averaged with a 2,000-resample percentile bootstrap CI over
questions (the resampling unit). Per-tag subgroup metrics are point
estimates only - per-tag n is too small (5-8 Qs) for an interpretable CI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from cardiorisk.eval.bootstrap import CI

from ..retrieval.pipeline import RetrievedChunk
from .loader import EvalQuestion

#: Default top-k considered when computing hit@k.
DEFAULT_TOP_K: Final[int] = 5
#: Bootstrap resample count (matches Phase 2.3a discipline).
DEFAULT_N_RESAMPLES: Final[int] = 2000
#: Pinned RNG seed (matches the rest of the project).
SEED: Final[int] = 20260505

_WS_RE = re.compile(r"\s+")


def _collapse(s: str) -> str:
    return _WS_RE.sub(" ", s).lower()


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _is_hit(question: EvalQuestion, chunk: RetrievedChunk) -> bool:
    """Standard-Q hit predicate (the negative-case inversion happens upstream)."""
    c = chunk.chunk
    if c.doc_id != question.expected_doc_id:
        return False
    if not _ranges_overlap(
        (c.page_start, c.page_end),
        (int(question.expected_page_range[0]), int(question.expected_page_range[1])),
    ):
        return False
    haystack = _collapse(c.text)
    return all(_collapse(kw) in haystack for kw in question.expected_span_keywords)


def _no_top_k_match(
    question: EvalQuestion,
    retrieved: list[RetrievedChunk],
    top_k: int,
) -> bool:
    """Negative-case hit predicate: no top-k chunk contains all keywords.

    For the negative case we only check keyword presence; doc_id and
    page_range are sentinel and ignored.
    """
    for hit in retrieved[:top_k]:
        haystack = _collapse(hit.chunk.text)
        if all(_collapse(kw) in haystack for kw in question.expected_span_keywords):
            return False
    return True


@dataclass(frozen=True)
class QuestionResult:
    """Per-question scoring output."""

    question_id: str
    hit_at_1: bool
    hit_at_5: bool
    rank_of_first_hit: int | None  # 1-indexed; None if not retrieved in top_k
    expected_no_hit: bool
    tags: tuple[str, ...]


def score_question(
    question: EvalQuestion,
    retrieved: list[RetrievedChunk],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> QuestionResult:
    """Score one (question, retrieved-list) pair.

    For standard Qs:

    - ``hit_at_1`` = the rank-1 chunk is a hit.
    - ``hit_at_5`` = at least one chunk in the top-``top_k`` is a hit.
    - ``rank_of_first_hit`` = 1-indexed rank of the first hit, or
      ``None`` if no hit in the top-``top_k``.

    For negative-case Qs (``expected_no_hit: true``):

    - ``hit_at_1`` = no top-1 chunk contained all keywords.
    - ``hit_at_5`` = no top-``top_k`` chunk contained all keywords.
    - ``rank_of_first_hit`` = ``1`` if ``hit_at_1`` else ``None``
      (MRR semantics: a successful negative case contributes 1.0).
    """
    if question.expected_no_hit:
        no_match_top1 = _no_top_k_match(question, retrieved, top_k=1)
        no_match_topk = _no_top_k_match(question, retrieved, top_k=top_k)
        return QuestionResult(
            question_id=question.id,
            hit_at_1=no_match_top1,
            hit_at_5=no_match_topk,
            rank_of_first_hit=1 if no_match_top1 else None,
            expected_no_hit=True,
            tags=question.tags,
        )

    rank_of_first_hit: int | None = None
    for rank, chunk in enumerate(retrieved[:top_k], start=1):
        if _is_hit(question, chunk):
            rank_of_first_hit = rank
            break
    return QuestionResult(
        question_id=question.id,
        hit_at_1=(rank_of_first_hit == 1),
        hit_at_5=(rank_of_first_hit is not None),
        rank_of_first_hit=rank_of_first_hit,
        expected_no_hit=False,
        tags=question.tags,
    )


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics for one eval cell.

    Attributes:
        n_questions: Total Qs evaluated (after fixture filtering).
        hit_at_1: Mean hit@1 across questions.
        hit_at_5: Mean hit@5 across questions.
        mrr: Mean reciprocal rank across questions.
        ci_hit_at_1: Bootstrap 95% CI for hit@1.
        ci_hit_at_5: Bootstrap 95% CI for hit@5.
        ci_mrr: Bootstrap 95% CI for MRR.
        per_tag: ``{tag: {hit_at_1, hit_at_5, mrr, n}}`` point-estimates.
    """

    n_questions: int
    hit_at_1: float
    hit_at_5: float
    mrr: float
    ci_hit_at_1: CI
    ci_hit_at_5: CI
    ci_mrr: CI
    per_tag: dict[str, dict[str, float]] = field(default_factory=dict)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> CI:
    """Percentile-method bootstrap CI on the mean of a 1-d array."""
    if values.size == 0:
        return CI(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_resamples=n_resamples,
            alpha=alpha,
        )
    rng = np.random.default_rng(seed)
    n = values.size
    point = float(values.mean())
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(values[idx].mean())
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return CI(point=point, lower=lower, upper=upper, n_resamples=n_resamples, alpha=alpha)


def aggregate_scores(
    results: Iterable[QuestionResult],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = SEED,
) -> EvalReport:
    """Aggregate :class:`QuestionResult` rows into an :class:`EvalReport`."""
    rows = list(results)
    n = len(rows)
    if n == 0:
        nan_ci = CI(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_resamples=n_resamples,
            alpha=0.05,
        )
        return EvalReport(
            n_questions=0,
            hit_at_1=float("nan"),
            hit_at_5=float("nan"),
            mrr=float("nan"),
            ci_hit_at_1=nan_ci,
            ci_hit_at_5=nan_ci,
            ci_mrr=nan_ci,
            per_tag={},
        )

    h1 = np.array([1.0 if r.hit_at_1 else 0.0 for r in rows], dtype=np.float64)
    h5 = np.array([1.0 if r.hit_at_5 else 0.0 for r in rows], dtype=np.float64)
    rr = np.array(
        [1.0 / r.rank_of_first_hit if r.rank_of_first_hit is not None else 0.0 for r in rows],
        dtype=np.float64,
    )

    ci_h1 = _bootstrap_mean_ci(h1, n_resamples=n_resamples, seed=seed)
    ci_h5 = _bootstrap_mean_ci(h5, n_resamples=n_resamples, seed=seed + 1)
    ci_mrr = _bootstrap_mean_ci(rr, n_resamples=n_resamples, seed=seed + 2)

    # Per-tag point estimates. A Q with multiple tags contributes
    # to each.
    per_tag: dict[str, dict[str, float]] = {}
    tag_to_idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        for tag in r.tags:
            tag_to_idx.setdefault(tag, []).append(i)
    for tag, idxs in tag_to_idx.items():
        idx_arr = np.array(idxs, dtype=np.int64)
        per_tag[tag] = {
            "hit_at_1": float(h1[idx_arr].mean()),
            "hit_at_5": float(h5[idx_arr].mean()),
            "mrr": float(rr[idx_arr].mean()),
            "n": float(len(idx_arr)),
        }

    return EvalReport(
        n_questions=n,
        hit_at_1=float(h1.mean()),
        hit_at_5=float(h5.mean()),
        mrr=float(rr.mean()),
        ci_hit_at_1=ci_h1,
        ci_hit_at_5=ci_h5,
        ci_mrr=ci_mrr,
        per_tag=per_tag,
    )
