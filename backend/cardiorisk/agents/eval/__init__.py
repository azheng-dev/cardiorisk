"""Phase 4 agent eval harness.

Modules:

- :mod:`cardiorisk.agents.eval.loader` — load + JSON-Schema-validate
  the synthetic case set in ``eval/agents/cases.jsonl``.
- :mod:`cardiorisk.agents.eval.scorer` — per-case + aggregate scoring
  primitives (per-stage pass/fail + per-band confusion matrix +
  wall-clock breakdowns).
- :mod:`cardiorisk.agents.eval.figures` — matplotlib renderers for the
  three Phase-4 dashboard figures.
- :mod:`cardiorisk.agents.eval.orchestrator` — end-to-end driver
  ``run_eval(...)`` that builds a graph + runs every case under an
  auto-approve harness + writes the report JSONs and figures.

The module sits inside :mod:`cardiorisk.agents` (not the top-level
``cardiorisk.eval``) because the eval is *about* the graph and
imports the graph helpers directly. Keeping it here also makes the
public ``cardiorisk.eval`` package (Phase 2.3a) the model-eval
namespace; the agent eval is a separate concern.
"""

from .loader import AgentEvalCase, load_cases
from .orchestrator import EvalConfig, run_eval
from .scorer import (
    AggregateReport,
    CaseReport,
    StageReport,
    aggregate_reports,
    score_case,
)

__all__ = [
    "AgentEvalCase",
    "AggregateReport",
    "CaseReport",
    "EvalConfig",
    "StageReport",
    "aggregate_reports",
    "load_cases",
    "run_eval",
    "score_case",
]
