"""End-to-end Phase-4 agent eval driver.

Runs the 4-agent graph against every case in the eval set under an
auto-approve harness (every HITL gate gets an Approve decision so the
graph runs to completion without a human). Writes:

- ``reports/v1/agents/per_case.json`` — list of :class:`CaseReport`.
- ``reports/v1/agents/aggregate.json`` — :class:`AggregateReport` +
  the ``EvalConfig`` block + a small environment block (LLM /
  NLI / retrieval used).
- ``reports/v1/figures/agents/{per_stage_pass_rate,risk_band_confusion,
  per_tag_pass_rate}.png`` — the dashboard figures.

The orchestrator is `--smoke`-aware: in smoke mode we run the first 3
cases against a fixture-only retrieval pipeline + Mock LLM + Mock
NLI. The full mode runs all 30 cases against whatever pipeline the
caller passes in. Phase 6 will revisit with real-LLM evals.
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

from .figures import render_all
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
) -> dict[str, Any]:
    """Run the full eval and persist per-case + aggregate reports + figures.

    Returns the summary dict that was written to
    ``aggregate.json`` so the CLI can ``print(json.dumps(...))`` it
    in CI for quick triage.
    """
    cases = cases if cases is not None else load_cases(cases_path_override)
    out_dir = output_dir or REPORTS_V1_AGENTS
    fig_dir = figures_dir or REPORTS_V1_AGENTS_FIGURES
    if is_smoke:
        out_dir = out_dir / "smoke"
        fig_dir = fig_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_case: list[CaseReport] = []
    t0 = time.perf_counter()
    for case in cases:
        terminal_state = _run_case(
            case=case,
            generator=generator,
            risk_model_name=risk_model_name,
            risk_held_out_source=risk_held_out_source,
        )
        per_case.append(score_case(case, terminal_state))
    elapsed_s = time.perf_counter() - t0

    aggregate = aggregate_reports(per_case)

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
        extras={"wall_clock_s": round(elapsed_s, 2)},
    )

    per_case_json = [report_to_dict(r) for r in per_case]
    aggregate_dict = asdict(aggregate)
    summary = _serialise_aggregate(aggregate=aggregate_dict, config=config)

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


__all__ = ["EvalConfig", "run_eval"]
