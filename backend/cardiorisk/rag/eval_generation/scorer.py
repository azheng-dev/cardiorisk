"""Citation precision / recall / hallucination / refusal metrics.

Per ``eval/generation/README.md``:

- **Citation precision** — over verified claims, fraction whose
  headline citation actually entails the claim under the verifier.
  In practice this is ~1.0 because the verifier dropped any
  un-entailed claim upstream; a sub-1.0 number means a
  multi-claim sentence slipped through and one of the bundled
  claims wasn't covered. The metric is here for parity with the
  literature and for the eval harness to detect regressions.
- **Citation recall (keyword)** — fraction of
  ``expected_keywords`` that appear in the verified answer text.
- **Hallucination rate** — for positive cases, fraction whose
  verified answer contains a substantive claim that does not match
  any expected doc_id (i.e. the verifier passed but the citation
  wasn't to the right document).
- **Refusal accuracy** — for ``should_refuse: true`` cases, fraction
  that produced a refusal.
- **Per-tag breakdown** — same metrics restricted to each tag.

Bootstrap CIs (2,000 resamples) are reported on the four headline
aggregates; per-tag estimates are point estimates only (n is too
small for an interpretable CI).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from cardiorisk.eval.bootstrap import CI

from ..generation.generator import GeneratedAnswer
from .loader import EvalCase

#: Bootstrap resample count (matches Phase 3.2 discipline).
DEFAULT_N_RESAMPLES: Final[int] = 2000
#: Pinned RNG seed (matches the rest of the project).
SEED: Final[int] = 20260505

_WS_RE = re.compile(r"\s+")


def _collapse(s: str) -> str:
    return _WS_RE.sub(" ", s).lower()


@dataclass(frozen=True)
class CaseResult:
    """Per-case scoring output."""

    case_id: str
    tags: tuple[str, ...]
    is_refusal_case: bool
    refused: bool
    keyword_recall: float  # in [0, 1]
    citation_precision: float  # in [0, 1]; NaN-encoded as 1.0 when no claims to score
    hallucination: bool  # positive cases only; False for refusal cases
    n_verified_claims: int
    n_suppressed_claims: int


def score_case(case: EvalCase, answer: GeneratedAnswer) -> CaseResult:
    """Score one (case, generated answer) pair."""
    verified_text_lower = _collapse(answer.verified_text)

    # Keyword recall — only meaningful for positive cases. Refusal
    # cases get 1.0 if the system refused, 0.0 otherwise (the eval
    # treats "correctly refused" as equivalent to "perfectly recalled
    # the answer" for the headline aggregate).
    if case.should_refuse:
        keyword_recall = 1.0 if answer.is_refusal else 0.0
    elif not case.expected_keywords:
        keyword_recall = 1.0
    else:
        hits = sum(1 for kw in case.expected_keywords if _collapse(kw) in verified_text_lower)
        keyword_recall = hits / len(case.expected_keywords)

    # Citation precision — fraction of verified claims whose headline
    # citation hit the entailment threshold. The verifier already
    # filtered, so this is ~1.0 by construction; left here so a
    # future relaxed-verifier mode (e.g. log-only) still surfaces.
    if answer.verified_claims:
        citation_precision = float(
            sum(1 for c in answer.verified_claims if c.headline_score > 0)
            / len(answer.verified_claims)
        )
    else:
        citation_precision = 1.0  # no claims to score; not a precision miss

    # Hallucination flag — for positive cases only. True iff the
    # verified answer contains at least one verified claim whose
    # headline citation is to a document NOT in expected_doc_ids.
    hallucination = False
    if not case.should_refuse and case.expected_doc_ids:
        expected = set(case.expected_doc_ids)
        retrieved_doc_by_chunk = {r.chunk.chunk_id: r.chunk.doc_id for r in answer.retrieved}
        for claim in answer.verified_claims:
            doc = retrieved_doc_by_chunk.get(claim.headline_chunk_id)
            if doc is not None and doc not in expected:
                hallucination = True
                break

    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        is_refusal_case=case.should_refuse,
        refused=answer.is_refusal,
        keyword_recall=keyword_recall,
        citation_precision=citation_precision,
        hallucination=hallucination,
        n_verified_claims=len(answer.verified_claims),
        n_suppressed_claims=len(answer.suppressed_claims),
    )


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics for one eval cell."""

    n_cases: int
    n_positive: int
    n_refusal: int
    citation_precision: float
    keyword_recall: float
    hallucination_rate: float
    refusal_accuracy: float
    ci_keyword_recall: CI
    ci_hallucination_rate: CI
    ci_refusal_accuracy: CI
    per_tag: dict[str, dict[str, float]] = field(default_factory=dict)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> CI:
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
    results: Iterable[CaseResult],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = SEED,
) -> EvalReport:
    """Aggregate :class:`CaseResult` rows into an :class:`EvalReport`."""
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
            n_cases=0,
            n_positive=0,
            n_refusal=0,
            citation_precision=float("nan"),
            keyword_recall=float("nan"),
            hallucination_rate=float("nan"),
            refusal_accuracy=float("nan"),
            ci_keyword_recall=nan_ci,
            ci_hallucination_rate=nan_ci,
            ci_refusal_accuracy=nan_ci,
        )

    refusal_rows = [r for r in rows if r.is_refusal_case]
    positive_rows = [r for r in rows if not r.is_refusal_case]

    keyword_recall = np.array([r.keyword_recall for r in rows], dtype=np.float64)
    citation_precision_vals = np.array([r.citation_precision for r in rows], dtype=np.float64)
    hallucination = np.array(
        [1.0 if r.hallucination else 0.0 for r in positive_rows], dtype=np.float64
    )
    refusal_correct = np.array([1.0 if r.refused else 0.0 for r in refusal_rows], dtype=np.float64)

    ci_recall = _bootstrap_mean_ci(keyword_recall, n_resamples=n_resamples, seed=seed)
    ci_halluc = _bootstrap_mean_ci(hallucination, n_resamples=n_resamples, seed=seed + 1)
    ci_refuse = _bootstrap_mean_ci(refusal_correct, n_resamples=n_resamples, seed=seed + 2)

    per_tag: dict[str, dict[str, float]] = {}
    tag_to_idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        for tag in r.tags:
            tag_to_idx.setdefault(tag, []).append(i)
    halluc_idx_in_rows: dict[str, int] = {}
    halluc_pos = 0
    for r in rows:
        if not r.is_refusal_case:
            halluc_idx_in_rows[r.case_id] = halluc_pos
            halluc_pos += 1
    for tag, idxs in tag_to_idx.items():
        idx_arr = np.array(idxs, dtype=np.int64)
        tag_rows = [rows[i] for i in idxs]
        tag_halluc_idx = [
            halluc_idx_in_rows[r.case_id]
            for r in tag_rows
            if not r.is_refusal_case and r.case_id in halluc_idx_in_rows
        ]
        tag_refuse = [1.0 if r.refused else 0.0 for r in tag_rows if r.is_refusal_case]
        per_tag[tag] = {
            "n": float(len(idx_arr)),
            "keyword_recall": float(keyword_recall[idx_arr].mean()),
            "citation_precision": float(citation_precision_vals[idx_arr].mean()),
            "hallucination_rate": (
                float(hallucination[np.array(tag_halluc_idx, dtype=np.int64)].mean())
                if tag_halluc_idx
                else float("nan")
            ),
            "refusal_accuracy": float(np.mean(tag_refuse)) if tag_refuse else float("nan"),
        }

    return EvalReport(
        n_cases=n,
        n_positive=len(positive_rows),
        n_refusal=len(refusal_rows),
        citation_precision=float(citation_precision_vals.mean()),
        keyword_recall=float(keyword_recall.mean()),
        hallucination_rate=float(hallucination.mean()) if hallucination.size else float("nan"),
        refusal_accuracy=(float(refusal_correct.mean()) if refusal_correct.size else float("nan")),
        ci_keyword_recall=ci_recall,
        ci_hallucination_rate=ci_halluc,
        ci_refusal_accuracy=ci_refuse,
        per_tag=per_tag,
    )
