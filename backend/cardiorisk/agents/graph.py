"""LangGraph wiring for the Phase 4 4-agent flow.

Pattern: each *agent* is a node that runs the agent's pure-Python logic
and returns a partial state update. Each agent is followed by a
*review* node that calls :func:`langgraph.types.interrupt` with the
agent's artefact + a small ``payload`` describing the gate. The graph
is compiled with :class:`InMemorySaver` so the parent process can
retrieve the paused state, render it for a human, and resume the
graph with a structured :class:`~cardiorisk.agents.state.Decision`.

Graph topology::

    START
      → triage
      → triage_review        (interrupt; resume with Decision)
      → risk
      → risk_review          (interrupt; resume with Decision)
      → guideline
      → guideline_review     (interrupt; resume with Decision)
      → letter
      → letter_review        (interrupt; resume with Decision)
      → END

A reject decision at any review gate sets ``terminated=True`` and the
conditional edge routes straight to END (the graph stops).

The state schema is :class:`~cardiorisk.agents.state.AgentState`
(Pydantic). LangGraph's default reducer is "replace"; we hand-roll
the tuple appends for ``decisions`` + ``audit`` inside each node so
the LangGraph framework only sees field-level updates.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from cardiorisk.rag.generation.generator import CitationGenerator

from .guideline import run_guideline
from .letter import run_letter
from .retries import TransientAgentError
from .risk import DEFAULT_HELD_OUT_SOURCE, DEFAULT_MODEL, run_risk
from .state import (
    AgentStage,
    AgentState,
    AuditEntry,
    Decision,
    DecisionStatus,
    EditDecision,
    GuidelineResult,
    LetterResult,
    RejectDecision,
    TriageResult,
    append_audit,
    append_decision,
)
from .triage import run_triage


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _audit(
    state: AgentState,
    *,
    stage: AgentStage,
    started_at: datetime,
    error: str | None = None,
    retry_count: int = 0,
) -> tuple[AuditEntry, ...]:
    completed_at = _now_utc()
    entry = AuditEntry(
        stage=stage,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=(completed_at - started_at).total_seconds() * 1000.0,
        error=error,
        retry_count=retry_count,
    )
    return append_audit(state, entry)


# ----------------------------------------------------------------- agent nodes
def _make_triage_node() -> Callable[[AgentState], dict[str, Any]]:
    def triage_node(state: AgentState) -> dict[str, Any]:
        started = _now_utc()
        try:
            result = run_triage(state.patient)
        except ValidationError as exc:
            return {
                "terminated": True,
                "termination_reason": f"triage validation failed: {exc}",
                "current_stage": AgentStage.triage,
                "audit": _audit(state, stage=AgentStage.triage, started_at=started, error=str(exc)),
            }
        return {
            "triage": result,
            "current_stage": AgentStage.triage,
            "audit": _audit(state, stage=AgentStage.triage, started_at=started),
        }

    return triage_node


def _make_risk_node(
    *,
    model_name: str,
    held_out_source: str,
) -> Callable[[AgentState], dict[str, Any]]:
    def risk_node(state: AgentState) -> dict[str, Any]:
        if state.triage is None:
            return {
                "terminated": True,
                "termination_reason": "risk_node: triage result missing",
            }
        started = _now_utc()
        result = run_risk(
            state.triage.normalised_patient,
            model_name=model_name,
            held_out_source=held_out_source,
        )
        return {
            "risk": result,
            "current_stage": AgentStage.risk,
            "audit": _audit(state, stage=AgentStage.risk, started_at=started),
        }

    return risk_node


def _make_guideline_node(*, generator: CitationGenerator) -> Callable[[AgentState], dict[str, Any]]:
    def guideline_node(state: AgentState) -> dict[str, Any]:
        if state.triage is None or state.risk is None:
            return {
                "terminated": True,
                "termination_reason": "guideline_node: prior stage missing",
            }
        started = _now_utc()
        try:
            result, attempts = run_guideline(
                patient=state.triage.normalised_patient,
                risk=state.risk,
                generator=generator,
            )
        except TransientAgentError as exc:
            return {
                "terminated": True,
                "termination_reason": f"guideline transient failure: {exc}",
                "current_stage": AgentStage.guideline,
                "audit": _audit(
                    state,
                    stage=AgentStage.guideline,
                    started_at=started,
                    error=str(exc),
                ),
            }
        return {
            "guideline": result,
            "current_stage": AgentStage.guideline,
            "audit": _audit(
                state,
                stage=AgentStage.guideline,
                started_at=started,
                retry_count=attempts,
            ),
        }

    return guideline_node


def _make_letter_node() -> Callable[[AgentState], dict[str, Any]]:
    def letter_node(state: AgentState) -> dict[str, Any]:
        if state.triage is None or state.risk is None or state.guideline is None:
            return {
                "terminated": True,
                "termination_reason": "letter_node: prior stage missing",
            }
        started = _now_utc()
        result = run_letter(
            patient=state.triage.normalised_patient,
            risk=state.risk,
            guideline=state.guideline,
        )
        return {
            "letter": result,
            "current_stage": AgentStage.letter,
            "audit": _audit(state, stage=AgentStage.letter, started_at=started),
        }

    return letter_node


# ----------------------------------------------------------------- review nodes
def _make_review_node(stage: AgentStage) -> Callable[[AgentState], dict[str, Any]]:
    """Build a HITL review node for ``stage``.

    The node calls :func:`interrupt` with a payload describing the
    artefact for the clinician. On resume, the value passed via
    ``Command(resume=...)`` MUST be a dict that pydantic can coerce
    into a :class:`~cardiorisk.agents.state.Decision`. The node
    appends the resulting :class:`AgentDecisionRecord` to the audit
    log; if the decision is ``edit`` for an artefact that supports
    edits, the patched artefact replaces the agent's original.
    """

    def review_node(state: AgentState) -> dict[str, Any]:
        artefact_payload: Any
        if stage is AgentStage.triage:
            artefact_payload = state.triage.model_dump() if state.triage else None
        elif stage is AgentStage.risk:
            artefact_payload = state.risk.model_dump() if state.risk else None
        elif stage is AgentStage.guideline:
            artefact_payload = state.guideline.model_dump() if state.guideline else None
        else:  # AgentStage.letter
            artefact_payload = state.letter.model_dump() if state.letter else None

        raw_decision = interrupt(
            {
                "stage": stage.value,
                "artefact": artefact_payload,
            }
        )

        # Coerce the resume payload into a Decision (pydantic union).
        decision = _coerce_decision(raw_decision)
        update: dict[str, Any] = {
            "decisions": append_decision(state, stage=stage, decision=decision),
        }
        if decision.status is DecisionStatus.reject:
            update["terminated"] = True
            update["termination_reason"] = f"rejected at {stage.value}: {decision.reason}"
        elif decision.status is DecisionStatus.edit:
            patched = _apply_edits(state=state, stage=stage, edits=decision.edits)
            if patched is not None:
                update[stage.value] = patched
        return update

    return review_node


def _coerce_decision(raw: Any) -> Decision:
    """Coerce a resume payload into a Decision pydantic union member.

    Resumes from the FastAPI surface arrive as plain dicts; we let
    pydantic do the discriminator dispatch so the node body stays
    declarative. The Decision union is discriminated on ``status``.
    """
    if isinstance(raw, dict):
        status = raw.get("status")
        if status == DecisionStatus.approve.value:
            from .state import ApproveDecision

            return ApproveDecision(**raw)
        if status == DecisionStatus.edit.value:
            return EditDecision(**raw)
        if status == DecisionStatus.reject.value:
            return RejectDecision(**raw)
    raise ValueError(f"unrecognised resume payload: {raw!r}")


def _apply_edits(*, state: AgentState, stage: AgentStage, edits: dict[str, Any]) -> Any | None:
    """Apply clinician edits to an artefact.

    Phase 4 supports edits for triage, guideline, and letter; risk
    edits are blocked (the calibrated probability and attributions
    are model-derived; a clinician edit would silently destroy the
    model's contract). A risk edit decision is recorded but
    ignored; the existing artefact stays in state.
    """
    if stage is AgentStage.triage and state.triage is not None:
        return TriageResult(**{**state.triage.model_dump(), **edits})
    if stage is AgentStage.guideline and state.guideline is not None:
        merged = {**state.guideline.model_dump(), **edits}
        return GuidelineResult(**merged)
    if stage is AgentStage.letter and state.letter is not None:
        return LetterResult(**{**state.letter.model_dump(), **edits})
    if stage is AgentStage.risk and state.risk is not None:
        # Risk artefact is model-derived; edits are intentionally
        # ignored to keep calibrated probabilities honest. The
        # decision is still recorded in the audit log.
        return None
    return None


# ----------------------------------------------------------------- routing
def _route_after_review(state: AgentState) -> str:
    return END if state.terminated else "_continue"


# ----------------------------------------------------------------- compile
def build_graph(
    *,
    generator: CitationGenerator,
    risk_model_name: str = DEFAULT_MODEL,
    risk_held_out_source: str = DEFAULT_HELD_OUT_SOURCE,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build + compile the 4-agent graph."""
    g = StateGraph(AgentState)

    # LangGraph's `add_node` is generic over `NodeInputT` extending a
    # BaseModel-like protocol; mypy can't narrow the callable to that
    # generic from a `Callable[[AgentState], dict[str, Any]]` alias, so
    # we suppress the overload mismatch at the call sites.
    g.add_node("triage", _make_triage_node())  # type: ignore[call-overload]
    g.add_node("triage_review", _make_review_node(AgentStage.triage))  # type: ignore[call-overload]
    g.add_node(  # type: ignore[call-overload]
        "risk",
        _make_risk_node(model_name=risk_model_name, held_out_source=risk_held_out_source),
    )
    g.add_node("risk_review", _make_review_node(AgentStage.risk))  # type: ignore[call-overload]
    g.add_node("guideline", _make_guideline_node(generator=generator))  # type: ignore[call-overload]
    g.add_node("guideline_review", _make_review_node(AgentStage.guideline))  # type: ignore[call-overload]
    g.add_node("letter", _make_letter_node())  # type: ignore[call-overload]
    g.add_node("letter_review", _make_review_node(AgentStage.letter))  # type: ignore[call-overload]

    g.add_edge(START, "triage")
    g.add_edge("triage", "triage_review")
    g.add_conditional_edges(
        "triage_review",
        _route_after_review,
        {"_continue": "risk", END: END},
    )
    g.add_edge("risk", "risk_review")
    g.add_conditional_edges(
        "risk_review",
        _route_after_review,
        {"_continue": "guideline", END: END},
    )
    g.add_edge("guideline", "guideline_review")
    g.add_conditional_edges(
        "guideline_review",
        _route_after_review,
        {"_continue": "letter", END: END},
    )
    g.add_edge("letter", "letter_review")
    g.add_conditional_edges(
        "letter_review",
        _route_after_review,
        {"_continue": END, END: END},
    )

    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ----------------------------------------------------------------- helpers
def state_from_dict(payload: dict[str, Any]) -> AgentState:
    """Reconstruct an :class:`AgentState` from the dict the graph emits.

    LangGraph returns a dict at every checkpoint; the FastAPI surface
    needs the typed object. This helper drops the framework's
    ``__interrupt__`` and ``__interrupt_id__`` keys so pydantic
    validation accepts the payload.
    """
    cleaned = {k: v for k, v in payload.items() if not k.startswith("__")}
    return AgentState.model_validate(cleaned)


def latest_interrupt(snapshot: Any) -> dict[str, Any] | None:
    """Return the active interrupt payload for a graph snapshot, if any."""
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", ()) or ()
        for it in interrupts:
            return cast(dict[str, Any], getattr(it, "value", {}))
    return None


def stage_payload(stage: AgentStage, state: AgentState) -> Any:
    """Return the artefact payload for a given stage."""
    if stage is AgentStage.triage:
        return state.triage
    if stage is AgentStage.risk:
        return state.risk
    if stage is AgentStage.guideline:
        return state.guideline
    if stage is AgentStage.letter:
        return state.letter
    raise AssertionError(f"unknown stage: {stage!r}")  # pragma: no cover


# Convenience wall-clock helper exported so the eval can wrap a full
# graph run in the same timer the audit log uses; tests verify the
# audit and the wall-clock agree to a few ms.
def perf_counter_ms() -> float:
    return time.perf_counter() * 1000.0


__all__ = [
    "build_graph",
    "latest_interrupt",
    "perf_counter_ms",
    "stage_payload",
    "state_from_dict",
]
