"""Tests for the Phase 4 agent state schema.

Covers the Pydantic models, the discriminated ``Decision`` union, and
the small ``append_decision`` / ``append_audit`` helpers. The schema
is the contract every agent + the FastAPI surface speak; if any of
these tests fail, the whole graph is wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cardiorisk.agents.state import (
    AgentDecisionRecord,
    AgentStage,
    AgentState,
    ApproveDecision,
    AuditEntry,
    DecisionStatus,
    EditDecision,
    PatientInput,
    RejectDecision,
    RiskAttribution,
    RiskResult,
    TriageResult,
    append_audit,
    append_decision,
)


def _patient() -> PatientInput:
    return PatientInput(
        Age=58,
        Sex="M",
        ChestPainType="ATA",
        RestingBP=140,
        Cholesterol=240,
        FastingBS=0,
        RestingECG="Normal",
        MaxHR=150,
        ExerciseAngina="N",
        Oldpeak=1.2,
        ST_Slope="Up",
    )


# ----------------------------------------------------------------- PatientInput
class TestPatientInput:
    def test_minimal_valid_patient_constructs(self) -> None:
        p = _patient()
        assert p.Age == 58
        assert p.Sex == "M"

    def test_invalid_sex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientInput(
                Age=58,
                Sex="X",  # type: ignore[arg-type]
                ChestPainType="ATA",
                RestingBP=140,
                Cholesterol=240,
                FastingBS=0,
                RestingECG="Normal",
                MaxHR=150,
                ExerciseAngina="N",
                Oldpeak=1.2,
                ST_Slope="Up",
            )

    def test_invalid_chest_pain_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientInput(
                Age=58,
                Sex="M",
                ChestPainType="XX",  # type: ignore[arg-type]
                RestingBP=140,
                Cholesterol=240,
                FastingBS=0,
                RestingECG="Normal",
                MaxHR=150,
                ExerciseAngina="N",
                Oldpeak=1.2,
                ST_Slope="Up",
            )

    def test_age_lower_bound_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PatientInput(
                Age=0,
                Sex="M",
                ChestPainType="ATA",
                RestingBP=140,
                Cholesterol=240,
                FastingBS=0,
                RestingECG="Normal",
                MaxHR=150,
                ExerciseAngina="N",
                Oldpeak=1.2,
                ST_Slope="Up",
            )


# ----------------------------------------------------------------- Decision union
class TestDecisionUnion:
    def test_approve_decision_carries_status(self) -> None:
        d = ApproveDecision()
        assert d.status is DecisionStatus.approve

    def test_edit_decision_requires_edits_dict(self) -> None:
        d = EditDecision(edits={"summary": "new"})
        assert d.status is DecisionStatus.edit
        assert d.edits == {"summary": "new"}

    def test_reject_decision_requires_reason(self) -> None:
        d = RejectDecision(reason="patient context insufficient")
        assert d.status is DecisionStatus.reject
        assert "patient context" in d.reason


# ----------------------------------------------------------------- helpers
class TestAppendHelpers:
    def test_append_decision_returns_new_tuple(self) -> None:
        state = AgentState(case_id="c1", patient=_patient())
        d = ApproveDecision()
        new_tuple = append_decision(state, stage=AgentStage.triage, decision=d)
        assert len(new_tuple) == 1
        record = new_tuple[0]
        assert isinstance(record, AgentDecisionRecord)
        assert record.stage is AgentStage.triage
        assert record.decision.status is DecisionStatus.approve

    def test_append_decision_preserves_immutability(self) -> None:
        """The original ``state.decisions`` tuple must be unchanged after append."""
        state = AgentState(case_id="c1", patient=_patient())
        original_id = id(state.decisions)
        append_decision(state, stage=AgentStage.triage, decision=ApproveDecision())
        assert id(state.decisions) == original_id
        assert state.decisions == ()

    def test_append_audit_extends_audit_log(self) -> None:
        state = AgentState(case_id="c1", patient=_patient())
        now = datetime.now(UTC)
        entry = AuditEntry(
            stage=AgentStage.triage,
            started_at=now,
            completed_at=now + timedelta(milliseconds=12),
            duration_ms=12.0,
            error=None,
            retry_count=0,
        )
        new_tuple = append_audit(state, entry)
        assert len(new_tuple) == 1
        assert new_tuple[0].stage is AgentStage.triage


# ----------------------------------------------------------------- AgentState
class TestAgentState:
    def test_initial_state_is_empty(self) -> None:
        state = AgentState(case_id="c1", patient=_patient())
        assert state.triage is None
        assert state.risk is None
        assert state.guideline is None
        assert state.letter is None
        assert state.terminated is False
        assert state.termination_reason is None
        assert state.decisions == ()
        assert state.audit == ()

    def test_state_round_trips_through_dict(self) -> None:
        triage = TriageResult(
            normalised_patient=_patient(),
            sanity_flags=("flag_one",),
            summary="ok",
        )
        risk = RiskResult(
            probability=0.42,
            risk_band="intermediate",
            threshold_high=0.10,
            threshold_low=0.05,
            model_name="tabicl",
            model_artefact_present=True,
            top_attributions=(RiskAttribution(feature="Age", contribution=0.12),),
            summary="risk 42%",
        )
        state = AgentState(case_id="c1", patient=_patient(), triage=triage, risk=risk)
        payload = state.model_dump()
        # `model_dump()` (Python mode) keeps tuples as tuples.
        assert tuple(payload["triage"]["sanity_flags"]) == ("flag_one",)
        assert payload["risk"]["risk_band"] == "intermediate"
        # Round-trip: model_dump output must validate back
        restored = AgentState.model_validate(payload)
        assert restored.triage is not None
        assert restored.triage.summary == "ok"
        assert restored.risk is not None
        assert restored.risk.probability == pytest.approx(0.42)
