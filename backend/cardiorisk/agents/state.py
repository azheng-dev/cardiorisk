"""Pydantic state schema for the Phase 4 agent graph.

Every node in :mod:`cardiorisk.agents.graph` reads the current
``AgentState`` and returns a partial dict of fields to merge in. The
state mutation pattern matches LangGraph's reducer model — fields are
either replaced (default) or appended to a tuple (for the audit log).

Public surface:

- :class:`AgentStage` — enum naming the four agent stages.
- :class:`Decision` — discriminated union of approve / edit / reject
  decisions a clinician can submit at a HITL gate. Every decision
  carries an ``actor`` and ``timestamp`` so the audit log is honest.
- :class:`AgentDecisionRecord` — one row in the audit log, pairs the
  agent's artefact with the clinician's decision.
- :class:`PatientInput` — minimal HFP-schema-aligned patient payload.
- :class:`TriageResult`, :class:`RiskResult`, :class:`GuidelineResult`,
  :class:`LetterResult` — per-stage artefact records.
- :class:`AgentState` — the LangGraph state object.

The schema deliberately mirrors the surfaces a Phase-5 UI will render:
``state.guideline.answer.verified_claims`` is the answer body,
``state.guideline.answer.suppressed_claims`` is the suppression audit,
``state.decisions`` is the HITL audit trail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cardiorisk.rag.generation.generator import GeneratedAnswer

#: Order matters — the graph walks stages in this order.
STAGE_ORDER: tuple[str, ...] = ("triage", "risk", "guideline", "letter")


class AgentStage(StrEnum):
    """Names of the four agent stages, in the order the graph executes."""

    triage = "triage"
    risk = "risk"
    guideline = "guideline"
    letter = "letter"


class DecisionStatus(StrEnum):
    """Outcome of a HITL gate."""

    approve = "approve"
    edit = "edit"
    reject = "reject"


class _DecisionBase(BaseModel):
    """Common fields for every HITL decision.

    ``actor`` is a free-form clinician identifier (no auth wired in
    Phase 4; Phase 8 swaps in the Supabase user id). ``timestamp``
    defaults to ``datetime.now(UTC)`` and is recorded server-side
    rather than client-side so audit times can't be backdated by the
    UI.
    """

    model_config = ConfigDict(frozen=True)

    actor: str = Field(default="anonymous", min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str | None = Field(default=None, max_length=2000)


class ApproveDecision(_DecisionBase):
    """Clinician accepts the agent's artefact as-is and unblocks the graph."""

    status: Literal[DecisionStatus.approve] = DecisionStatus.approve


class EditDecision(_DecisionBase):
    """Clinician supplies an edited version of the agent's artefact.

    The edit is stored verbatim in ``edits`` (free-form dict) and the
    graph then uses the edited artefact instead of the agent's
    original output. The edit MUST preserve the artefact's outer
    schema; partial edits are unsupported in Phase 4 (Phase 5.3 UI
    will enforce field-level edits).
    """

    status: Literal[DecisionStatus.edit] = DecisionStatus.edit
    edits: dict[str, Any] = Field(default_factory=dict)


class RejectDecision(_DecisionBase):
    """Clinician rejects the agent's artefact and ends the case.

    Rejection is terminal — Phase 4 does not support "send back to
    agent for retry". Phase 6 will revisit if the eval finds the
    rejection-without-recourse rate too high.
    """

    status: Literal[DecisionStatus.reject] = DecisionStatus.reject
    reason: str = Field(min_length=1, max_length=2000)


Decision = Annotated[
    ApproveDecision | EditDecision | RejectDecision,
    Field(discriminator="status"),
]


class AgentDecisionRecord(BaseModel):
    """One row in the immutable audit log."""

    model_config = ConfigDict(frozen=True)

    stage: AgentStage
    decision: Decision


# ---------------------------------------------------------------- patient input
class PatientInput(BaseModel):
    """Synthetic patient payload aligned with the HFP feature schema.

    These are the columns the v1 model expects; field names match the
    ``cardiorisk.data.combine`` schema. ``Sex`` is one of {"M", "F"};
    ``ChestPainType`` is one of {"TA", "ATA", "NAP", "ASY"};
    ``RestingECG`` is one of {"Normal", "ST", "LVH"}; ``ExerciseAngina``
    is one of {"Y", "N"}; ``ST_Slope`` is one of {"Up", "Flat", "Down"}.

    The fields are constrained to the legal ranges the EDA notebook
    documented so a malformed UI submission fails at the schema
    boundary rather than mid-pipeline.
    """

    model_config = ConfigDict(frozen=True)

    Age: int = Field(ge=18, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int = Field(ge=60, le=260)
    Cholesterol: int = Field(ge=0, le=800)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=50, le=240)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-3.0, le=8.0)
    ST_Slope: Literal["Up", "Flat", "Down"]


# ---------------------------------------------------------------- agent results
class TriageResult(BaseModel):
    """Output of the triage agent."""

    model_config = ConfigDict(frozen=True)

    normalised_patient: PatientInput
    sanity_flags: tuple[str, ...] = ()
    summary: str


class RiskAttribution(BaseModel):
    """One feature → attribution magnitude row."""

    model_config = ConfigDict(frozen=True)

    feature: str
    contribution: float


class RiskResult(BaseModel):
    """Output of the risk agent."""

    model_config = ConfigDict(frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "intermediate", "high"]
    threshold_high: float = 0.10
    threshold_low: float = 0.05
    model_name: str
    model_artefact_present: bool
    top_attributions: tuple[RiskAttribution, ...] = ()
    summary: str


class GuidelineResult(BaseModel):
    """Output of the guideline agent.

    ``answer`` is the structured :class:`GeneratedAnswer` from the
    Phase 3.3 ``CitationGenerator``. We carry the whole object
    (verified + suppressed claims + retrieved chunks) so Phase 5.3's
    UI can render the suppression-audit panel without a second
    backend call.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    question: str
    answer: GeneratedAnswer
    summary: str


class LetterResult(BaseModel):
    """Output of the letter agent.

    ``citations`` are chunk_ids from the guideline answer that were
    re-cited in the letter body. ``draft`` is the bracketed-citation
    referral letter; ``redacted_claims`` records any claims the
    letter agent wrote then dropped because the verifier rejected
    them.
    """

    model_config = ConfigDict(frozen=True)

    draft: str
    citations: tuple[str, ...] = ()
    redacted_claims: tuple[str, ...] = ()
    summary: str


# ---------------------------------------------------------------- audit log
class AuditEntry(BaseModel):
    """One entry in the agent-stage audit log."""

    model_config = ConfigDict(frozen=True)

    stage: AgentStage
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    error: str | None = None
    retry_count: int = 0


# ---------------------------------------------------------------- root state
class AgentState(BaseModel):
    """LangGraph state object carried through the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    patient: PatientInput
    triage: TriageResult | None = None
    risk: RiskResult | None = None
    guideline: GuidelineResult | None = None
    letter: LetterResult | None = None
    decisions: tuple[AgentDecisionRecord, ...] = ()
    audit: tuple[AuditEntry, ...] = ()
    terminated: bool = False
    termination_reason: str | None = None
    current_stage: AgentStage | None = None


def append_decision(
    state: AgentState, *, stage: AgentStage, decision: Decision
) -> tuple[AgentDecisionRecord, ...]:
    """Return a new ``decisions`` tuple with ``decision`` appended.

    This is a helper not a reducer — the LangGraph node returns the
    tuple and lets the framework merge it; we keep the merge logic
    here so individual nodes don't reach into the state shape.
    """
    record = AgentDecisionRecord(stage=stage, decision=decision)
    return (*state.decisions, record)


def append_audit(state: AgentState, entry: AuditEntry) -> tuple[AuditEntry, ...]:
    """Return a new ``audit`` tuple with ``entry`` appended."""
    return (*state.audit, entry)


__all__ = [
    "STAGE_ORDER",
    "AgentDecisionRecord",
    "AgentStage",
    "AgentState",
    "ApproveDecision",
    "AuditEntry",
    "Decision",
    "DecisionStatus",
    "EditDecision",
    "GuidelineResult",
    "LetterResult",
    "PatientInput",
    "RejectDecision",
    "RiskAttribution",
    "RiskResult",
    "TriageResult",
    "append_audit",
    "append_decision",
]
