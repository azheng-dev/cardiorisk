"""Pydantic request/response schemas for the FastAPI surface."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cardiorisk.agents.state import (
    AgentDecisionRecord,
    AgentStage,
    AgentState,
    AuditEntry,
    Decision,
    GuidelineResult,
    LetterResult,
    PatientInput,
    RiskResult,
    TriageResult,
)


class CaseCreate(BaseModel):
    """Request body for ``POST /v1/cases``."""

    case_id: str = Field(min_length=1, max_length=128)
    patient: PatientInput


class InterruptPayload(BaseModel):
    """Description of the artefact awaiting clinician review.

    ``stage`` is the review-gate name; ``artefact`` is the dict
    representation of the corresponding :class:`TriageResult` /
    :class:`RiskResult` / :class:`GuidelineResult` /
    :class:`LetterResult`.
    """

    stage: AgentStage
    artefact: dict[str, Any] | None = None


class CaseStateResponse(BaseModel):
    """``GET /v1/cases/{case_id}`` and decision-response payload.

    Mirrors :class:`AgentState` but flattens ``decisions`` and
    ``audit`` to lists for ergonomic JSON.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    patient: PatientInput
    triage: TriageResult | None = None
    risk: RiskResult | None = None
    guideline: GuidelineResult | None = None
    letter: LetterResult | None = None
    decisions: list[AgentDecisionRecord] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)
    terminated: bool = False
    termination_reason: str | None = None
    next_interrupt: InterruptPayload | None = None

    @classmethod
    def from_state(
        cls, state: AgentState, *, next_interrupt: InterruptPayload | None
    ) -> CaseStateResponse:
        return cls(
            case_id=state.case_id,
            patient=state.patient,
            triage=state.triage,
            risk=state.risk,
            guideline=state.guideline,
            letter=state.letter,
            decisions=list(state.decisions),
            audit=list(state.audit),
            terminated=state.terminated,
            termination_reason=state.termination_reason,
            next_interrupt=next_interrupt,
        )


class DecideRequest(BaseModel):
    """Request body for ``POST /v1/cases/{case_id}/decide``.

    The body is a :class:`Decision` (the discriminated union from
    :mod:`cardiorisk.agents.state`); the discriminator is ``status``
    (``"approve" | "edit" | "reject"``). FastAPI validates the body
    before it ever hits the graph — invalid decisions return 422
    rather than corrupting the graph state.
    """

    decision: Annotated[Decision, Field(discriminator="status")]


class DecideResponse(CaseStateResponse):
    """Response for ``POST /v1/cases/{case_id}/decide``.

    Identical shape to :class:`CaseStateResponse`. Modeled separately
    so OpenAPI docs differentiate the two endpoints clearly.
    """


__all__ = [
    "CaseCreate",
    "CaseStateResponse",
    "DecideRequest",
    "DecideResponse",
    "InterruptPayload",
]
