"""End-to-end Phase-4 / Phase-6 agent eval driver.

Runs the 4-agent graph against every case in the eval set under an
auto-approve harness (every HITL gate gets an Approve decision so the
graph runs to completion without a human). Writes:

- ``reports/v1/agents/per_case.json`` — list of per-case dicts
  (one :class:`CaseReport` + Phase-6 judge score per case).
- ``reports/v1/agents/aggregate.json`` — :class:`AggregateReport` +
  the ``EvalConfig`` block + a small environment block (LLM /
  NLI / retrieval used) + Phase-6 ``judge_aggregate`` + ``usage``
  block (token counts + USD cost) + optional ``regression`` block
  produced by :func:`check_regression`.
- ``reports/v1/figures/agents/{per_stage_pass_rate,risk_band_confusion,
  per_tag_pass_rate}.png`` — the dashboard figures.

The orchestrator is ``--smoke``-aware: in smoke mode we run a small
subset of cases against a fixture-only retrieval pipeline + Mock LLM
+ Mock NLI + Mock judge. The full mode runs the entire 100-case set
against whatever pipeline the caller passes in (Phase 6 adds the
Gemini live LLM + Gemini judge cells).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from cardiorisk.agents.graph import (
    build_graph,
    latest_interrupt,
    state_from_dict,
)
from cardiorisk.agents.state import (
    AgentState,
    ApproveDecision,
)
from cardiorisk.data.paths import (
    REPORTS_V1_AGENTS,
    REPORTS_V1_AGENTS_FIGURES,
)
from cardiorisk.rag.generation.generator import CitationGenerator
from cardiorisk.rag.generation.llm import UsageTotals

from .figures import render_all
from .judge import BaseJudge, JudgeAggregate, JudgeScore, MockJudge
from .loader import AgentEvalCase, load_cases
from .scorer import (
    CaseReport,
    aggregate_reports,
    report_to_dict,
    score_case,
)


@dataclass(frozen=True)
class EvalConfig:
    """Knob block recorded into the aggregate.json."""

    n_cases: int
    risk_model_name: str
    risk_held_out_source: str
    llm_client_name: str
    nli_verifier_name: str
    embedder_name: str
    reranker_name: str
    is_smoke: bool
    cases_path: str
    extras: dict[str, Any] = field(default_factory=dict)


def _run_case(
    *,
    case: AgentEvalCase,
    generator: CitationGenerator,
    risk_model_name: str,
    risk_held_out_source: str,
) -> AgentState:
    """Run one case through a fresh graph + auto-approve every gate."""
    graph = build_graph(
        generator=generator,
        risk_model_name=risk_model_name,
        risk_held_out_source=risk_held_out_source,
    )
    config = cast(RunnableConfig, {"configurable": {"thread_id": case.id}})
    init = AgentState(case_id=case.id, patient=case.patient).model_dump()
    graph.invoke(cast(Any, init), config=config)

    approve = ApproveDecision().model_dump()
    # Up to 4 gates: triage, risk, guideline, letter.
    for _ in range(4):
        snap = graph.get_state(config)
        if latest_interrupt(snap) is None:
            break
        graph.invoke(Command(resume=approve), config=config)

    snap = graph.get_state(config)
    return state_from_dict(snap.values)


def _serialise_aggregate(
    *,
    aggregate: dict[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "aggregate": aggregate,
    }


def run_eval(
    *,
    generator: CitationGenerator,
    cases: list[AgentEvalCase] | None = None,
    risk_model_name: str = "tabicl",
    risk_held_out_source: str = "Cleveland",
    llm_client_name: str = "MockLLMClient",
    nli_verifier_name: str = "MockNLIVerifier",
    embedder_name: str = "n/a",
    reranker_name: str = "off",
    is_smoke: bool = False,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    cases_path_override: Path | None = None,
    judge: BaseJudge | None = None,
    regression_baseline_path: Path | None = None,
    regression_tolerance_pp: float = 2.0,
) -> dict[str, Any]:
    """Run the full eval and persist per-case + aggregate reports + figures.

    Returns the summary dict that was written to
    ``aggregate.json`` so the CLI can ``print(json.dumps(...))`` it
    in CI for quick triage.

    Phase 6 additions:

    - ``judge`` — optional :class:`BaseJudge`. If supplied, every case
      is scored end-to-end (graph + judge) and the per-case report
      embeds the :class:`JudgeScore`. Defaults to a :class:`MockJudge`
      so the CI run always has a judge-scoring cell to regress
      against.
    - ``regression_baseline_path`` — optional path to a previous
      ``aggregate.json`` (or a stripped-down ``baseline_mock.json``).
      When set, the new aggregate is diffed against the baseline by
      :func:`check_regression`; if any tracked metric drops by more
      than ``regression_tolerance_pp`` percentage points, the
      summary's ``regression.failed`` flag is True and
      :func:`run_eval` returns normally (the CLI surfaces the flag as
      a non-zero exit code, not the orchestrator).
    """
    cases = cases if cases is not None else load_cases(cases_path_override)
    out_dir = output_dir or REPORTS_V1_AGENTS
    fig_dir = figures_dir or REPORTS_V1_AGENTS_FIGURES
    if is_smoke:
        out_dir = out_dir / "smoke"
        fig_dir = fig_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if judge is None:
        judge = MockJudge()

    per_case: list[CaseReport] = []
    judge_scores: list[JudgeScore] = []
    terminal_drafts: list[str] = []
    t0 = time.perf_counter()
    for case in cases:
        terminal_state = _run_case(
            case=case,
            generator=generator,
            risk_model_name=risk_model_name,
            risk_held_out_source=risk_held_out_source,
        )
        per_case.append(score_case(case, terminal_state))
        draft = terminal_state.letter.draft if terminal_state.letter else ""
        terminal_drafts.append(draft)
        judge_scores.append(
            judge.score(
                case_id=case.id,
                letter_draft=draft,
                expected_recommendation_family=case.expected_recommendation_family,
                tag=case.tag,
            )
        )
    elapsed_s = time.perf_counter() - t0

    aggregate = aggregate_reports(per_case)
    judge_aggregate = JudgeAggregate.from_scores(
        judge.name, judge_scores, [r.tag for r in per_case]
    )

    config = EvalConfig(
        n_cases=len(cases),
        risk_model_name=risk_model_name,
        risk_held_out_source=risk_held_out_source,
        llm_client_name=llm_client_name,
        nli_verifier_name=nli_verifier_name,
        embedder_name=embedder_name,
        reranker_name=reranker_name,
        is_smoke=is_smoke,
        cases_path=str(cases_path_override or "eval/agents/cases.jsonl"),
        extras={
            "wall_clock_s": round(elapsed_s, 2),
            "judge_name": judge.name,
        },
    )

    per_case_json = []
    for cr, js in zip(per_case, judge_scores, strict=True):
        d = report_to_dict(cr)
        d["judge"] = asdict(js)
        per_case_json.append(d)

    aggregate_dict = asdict(aggregate)
    usage_block = {
        "generator_llm": _usage_snapshot(generator.llm_usage),
        "judge_llm": judge.usage.snapshot(),
    }
    judge_block = asdict(judge_aggregate)
    summary: dict[str, Any] = {
        "config": asdict(config),
        "aggregate": aggregate_dict,
        "judge_aggregate": judge_block,
        "usage": usage_block,
    }

    if regression_baseline_path is not None:
        summary["regression"] = check_regression(
            current_summary=summary,
            baseline_path=regression_baseline_path,
            tolerance_pp=regression_tolerance_pp,
        )

    (out_dir / "per_case.json").write_text(
        json.dumps(per_case_json, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    render_all(aggregate, fig_dir)

    return summary


def _usage_snapshot(usage: UsageTotals | None) -> dict[str, float | int]:
    if usage is None:
        return UsageTotals().snapshot()
    return usage.snapshot()


# ---------------------------------------------------------------------------
# Phase 6.5 — regression gate.
# ---------------------------------------------------------------------------
#: Metrics the regression gate watches. Each value is a (jsonpath, label)
#: pair. ``check_regression`` reads the metric from both the current
#: summary and the baseline, computes the delta, and flags a fail if
#: the metric drops by more than the tolerance (in percentage points).
REGRESSION_METRICS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("aggregate", "triage_pass_rate"), "triage_pass_rate"),
    (("aggregate", "risk_band_match_rate"), "risk_band_match_rate"),
    (("aggregate", "guideline_pass_rate"), "guideline_pass_rate"),
    (("aggregate", "letter_pass_rate"), "letter_pass_rate"),
    (("aggregate", "full_pipeline_pass_rate"), "full_pipeline_pass_rate"),
    (("aggregate", "recommendation_correctness_rate"), "recommendation_correctness_rate"),
    (("aggregate", "mean_citation_precision"), "mean_citation_precision"),
    (("aggregate", "mean_citation_recall"), "mean_citation_recall"),
    (("judge_aggregate", "pass_rate"), "judge_pass_rate"),
)

#: Metrics that should NOT increase by more than the tolerance (the
#: lower-is-better axis). Mirrors :data:`REGRESSION_METRICS` but
#: inverted at the comparison step.
REGRESSION_METRICS_LOWER_IS_BETTER: tuple[tuple[tuple[str, ...], str], ...] = (
    (("aggregate", "mean_hallucination_rate"), "mean_hallucination_rate"),
)


def _read_path(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def check_regression(
    *,
    current_summary: dict[str, Any],
    baseline_path: Path,
    tolerance_pp: float = 2.0,
) -> dict[str, Any]:
    """Diff the current summary against a previously-saved baseline and
    return a dict the CLI can fail on.

    Output shape::

        {
            "baseline_path": "...",
            "tolerance_pp": 2.0,
            "failed": bool,
            "deltas": {
                "<metric>": {
                    "current": float,
                    "baseline": float,
                    "delta_pp": float,
                    "fail": bool,
                    "direction": "higher_is_better" | "lower_is_better"
                },
                ...
            }
        }

    Missing baseline metrics are recorded with ``fail=False`` — the
    gate only fails on metrics that exist in both files and that drift
    by more than the tolerance in the wrong direction. The first time
    the gate runs against an updated metric set, those metrics show
    up under ``deltas`` with ``baseline=None``.
    """
    if not baseline_path.exists():
        raise FileNotFoundError(f"regression baseline {baseline_path} not found")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    tol = abs(tolerance_pp) / 100.0
    deltas: dict[str, dict[str, Any]] = {}
    failed = False

    for path, label in REGRESSION_METRICS:
        cur = _read_path(current_summary, path)
        base = _read_path(baseline, path)
        if cur is None or base is None:
            deltas[label] = {
                "current": cur,
                "baseline": base,
                "delta_pp": None,
                "fail": False,
                "direction": "higher_is_better",
            }
            continue
        delta = cur - base
        is_fail = delta < -tol
        deltas[label] = {
            "current": round(cur, 6),
            "baseline": round(base, 6),
            "delta_pp": round(delta * 100.0, 4),
            "fail": is_fail,
            "direction": "higher_is_better",
        }
        if is_fail:
            failed = True

    for path, label in REGRESSION_METRICS_LOWER_IS_BETTER:
        cur = _read_path(current_summary, path)
        base = _read_path(baseline, path)
        if cur is None or base is None:
            deltas[label] = {
                "current": cur,
                "baseline": base,
                "delta_pp": None,
                "fail": False,
                "direction": "lower_is_better",
            }
            continue
        delta = cur - base
        is_fail = delta > tol
        deltas[label] = {
            "current": round(cur, 6),
            "baseline": round(base, 6),
            "delta_pp": round(delta * 100.0, 4),
            "fail": is_fail,
            "direction": "lower_is_better",
        }
        if is_fail:
            failed = True

    return {
        "baseline_path": str(baseline_path),
        "tolerance_pp": tolerance_pp,
        "failed": failed,
        "deltas": deltas,
    }


__all__ = [
    "REGRESSION_METRICS",
    "REGRESSION_METRICS_LOWER_IS_BETTER",
    "EvalConfig",
    "check_regression",
    "run_eval",
]
