"""Pydantic request/response schemas for the FastAPI surface.

Phase 7 reshapes the public response to be **frontend-compatible**:
the same JSON schema the Next.js zod parser accepts (see
``frontend/src/lib/agents/schema.ts``). Concretely:

- ``status``: ``"awaiting_decision" | "complete" | "rejected"`` —
  derived from terminated + next_interrupt.
- ``next_stage``: ``AgentStage | None`` — the stage currently
  awaiting a HITL decision (or ``None`` when terminated/complete).
- ``decisions``: flattened to ``{stage, status, note?, timestamp}``
  rows so the audit screen renders without an extra join.
- ``trace_id``: per-case Langfuse trace id (or mock sentinel) so the
  UI can deep-link to the trace view.

The "internal" fields (``terminated``, ``termination_reason``,
``next_interrupt``) are still emitted for debugging — the FE zod
schema strips unknown keys, so they're harmless on the wire.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cardiorisk.agents.state import (
    STAGE_ORDER,
    AgentDecisionRecord,
    AgentStage,
    AgentState,
    AuditEntry,
    DecisionStatus,
    GuidelineResult,
    LetterResult,
    PatientInput,
    RiskResult,
    TriageResult,
)


class CaseCreate(BaseModel):
    """Request body for ``POST /v1/agents/cases``.

    Phase 7: ``case_id`` is now optional. When absent, the server
    mints a ``c{8-hex}`` id so the frontend never has to manage case
    ids client-side. Existing callers that supply a case_id keep
    working unchanged (the explicit id wins).
    """

    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    patient: PatientInput


class InterruptPayload(BaseModel):
    """Description of the artefact awaiting clinician review."""

    stage: AgentStage
    artefact: dict[str, Any] | None = None


class FlatDecisionRecord(BaseModel):
    """Frontend-shape HITL decision row.

    Mirrors ``frontend/src/lib/agents/schema.ts ::decisionRecordSchema``
    1:1 so a regression in either fails the same way. ``status``
    here is the past-tense form (``approved | edited | rejected``);
    the underlying :class:`~cardiorisk.agents.state.Decision`'s
    ``status`` uses the verb form (``approve | edit | reject``).
    """

    model_config = ConfigDict(frozen=True)

    stage: AgentStage
    status: Literal["approved", "edited", "rejected", "pending"]
    note: str | None = None
    timestamp: datetime

    @classmethod
    def from_record(cls, record: AgentDecisionRecord) -> FlatDecisionRecord:
        verb_to_past: dict[str, Literal["approved", "edited", "rejected", "pending"]] = {
            DecisionStatus.approve.value: "approved",
            DecisionStatus.edit.value: "edited",
            DecisionStatus.reject.value: "rejected",
        }
        d = record.decision
        # Reject decisions carry the rejection text on ``reason``;
        # surface it under ``note`` so the UI shows one field.
        note: str | None
        if d.status.value == DecisionStatus.reject.value:
            reason_attr = getattr(d, "reason", None)
            note = reason_attr or d.note
        else:
            note = d.note
        return cls(
            stage=record.stage,
            status=verb_to_past[d.status.value],
            note=note,
            timestamp=d.timestamp,
        )


class CaseStateResponse(BaseModel):
    """Wire shape for ``GET /v1/agents/cases/{case_id}``.

    Composed of two layers:

    - **FE-compatible fields** (``status`` / ``next_stage`` /
      ``decisions`` / ``trace_id``) consumed by the Phase 5.3 UI.
    - **Internal fields** (``terminated`` / ``termination_reason`` /
      ``next_interrupt``) kept on the wire for debugging; the FE
      strips them via zod's default "unknown-key drop" behaviour.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    status: Literal["awaiting_decision", "complete", "rejected"]
    next_stage: AgentStage | None = None
    trace_id: str | None = None

    patient: PatientInput
    triage: TriageResult | None = None
    risk: RiskResult | None = None
    guideline: GuidelineResult | None = None
    letter: LetterResult | None = None

    decisions: list[FlatDecisionRecord] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)

    # ---------------- internal-but-exposed (debug + introspection) -----
    terminated: bool = False
    termination_reason: str | None = None
    next_interrupt: InterruptPayload | None = None

    @classmethod
    def from_state(
        cls, state: AgentState, *, next_interrupt: InterruptPayload | None
    ) -> CaseStateResponse:
        status: Literal["awaiting_decision", "complete", "rejected"]
        if state.terminated:
            status = "rejected"
        elif next_interrupt is None:
            status = "complete"
        else:
            status = "awaiting_decision"
        return cls(
            case_id=state.case_id,
            status=status,
            next_stage=next_interrupt.stage if next_interrupt is not None else None,
            trace_id=state.trace_id,
            patient=state.patient,
            triage=state.triage,
            risk=state.risk,
            guideline=state.guideline,
            letter=state.letter,
            decisions=[FlatDecisionRecord.from_record(d) for d in state.decisions],
            audit=list(state.audit),
            terminated=state.terminated,
            termination_reason=state.termination_reason,
            next_interrupt=next_interrupt,
        )


class DecideRequest(BaseModel):
    """Request body for ``POST /v1/agents/cases/{case_id}/decide``.

    Phase 7 swaps the nested ``{decision: {...}}`` shape for the
    FE-compatible flat shape: ``{stage, status, note?}``. ``stage``
    is informational (the server already knows which stage it's
    paused at via the interrupt). ``status`` is the past-tense form
    (``approved | edited | rejected``); we translate to the verb
    form on the wire to the :class:`~cardiorisk.agents.state.Decision`
    union.
    """

    stage: AgentStage
    status: Literal["approved", "edited", "rejected"]
    note: str | None = Field(default=None, max_length=2000)


class DecideResponse(CaseStateResponse):
    """Response for ``POST /v1/agents/cases/{case_id}/decide``."""


__all__ = [
    "STAGE_ORDER",
    "CaseCreate",
    "CaseStateResponse",
    "DecideRequest",
    "DecideResponse",
    "FlatDecisionRecord",
    "InterruptPayload",
]
