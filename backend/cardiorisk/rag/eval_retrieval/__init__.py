"""Phase 3.2 retrieval-eval layer (binding decision: ADR-016).

Loads the 50-Q hand-curated eval set from
``eval/retrieval/questions.jsonl``, scores
:class:`cardiorisk.rag.retrieval.RetrievalPipeline` outputs against
each question, and orchestrates the full ``{chunker x rerank} = 6``
eval matrix.

Module map:

- :mod:`.loader` — load + JSON-Schema-validate the eval set; filter
  by ``--use-fixture`` (which skips ``requires_full_corpus: true``
  rows).
- :mod:`.scorer` — :func:`score_question` (hit / hit-rank / per-Q
  metrics), :func:`aggregate_scores` (hit@1 / hit@5 / MRR + bootstrap
  CIs + per-tag breakdown). Mirrors the binary-classification
  :mod:`cardiorisk.eval.bootstrap` discipline (percentile method,
  default 2,000 resamples, pinned seed).
- :mod:`.orchestrator` — runs the full eval matrix and writes
  ``reports/v1/retrieval/{per_cell,aggregate}.json`` + figures.
- :mod:`.figures` — bar charts + heatmap renderers (matplotlib).
"""

from .loader import EvalQuestion, load_questions
from .scorer import (
    EvalReport,
    QuestionResult,
    aggregate_scores,
    score_question,
)

__all__ = [
    "EvalQuestion",
    "EvalReport",
    "QuestionResult",
    "aggregate_scores",
    "load_questions",
    "score_question",
]
