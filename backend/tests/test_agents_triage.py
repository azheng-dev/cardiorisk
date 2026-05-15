"""Tests for the rule-based triage agent."""

from __future__ import annotations

import pytest

from cardiorisk.agents.state import PatientInput
from cardiorisk.agents.triage import run_triage


def _patient(**overrides: object) -> PatientInput:
    base = {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 240,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Up",
    }
    base.update(overrides)
    return PatientInput(**base)  # type: ignore[arg-type]


class TestRunTriage:
    def test_clean_patient_has_no_flags(self) -> None:
        result = run_triage(_patient())
        assert result.sanity_flags == ()
        assert result.normalised_patient == _patient()
        assert "58" in result.summary

    def test_cholesterol_zero_flagged_as_missing_sentinel(self) -> None:
        result = run_triage(_patient(Cholesterol=0))
        assert "cholesterol_missing_sentinel" in result.sanity_flags

    def test_negative_oldpeak_flagged(self) -> None:
        result = run_triage(_patient(Oldpeak=-0.4))
        assert "oldpeak_negative" in result.sanity_flags

    def test_extreme_resting_bp_flagged(self) -> None:
        for bp in (85, 200):
            result = run_triage(_patient(RestingBP=bp))
            assert "resting_bp_extreme" in result.sanity_flags

    def test_age_outside_training_range_flagged(self) -> None:
        for age in (25, 80):
            result = run_triage(_patient(Age=age))
            assert "age_outside_training_range" in result.sanity_flags

    def test_summary_is_a_one_line_string(self) -> None:
        result = run_triage(_patient())
        assert "\n" not in result.summary
        assert isinstance(result.summary, str)
        assert len(result.summary) > 10

    def test_normalised_patient_equals_input_when_no_normalisation_needed(self) -> None:
        p = _patient()
        result = run_triage(p)
        assert result.normalised_patient == p

    def test_multiple_flags_compose(self) -> None:
        p = _patient(Cholesterol=0, Oldpeak=-0.5)
        result = run_triage(p)
        assert "cholesterol_missing_sentinel" in result.sanity_flags
        assert "oldpeak_negative" in result.sanity_flags

    def test_triage_is_deterministic(self) -> None:
        p = _patient(Cholesterol=0)
        a = run_triage(p)
        b = run_triage(p)
        assert a.sanity_flags == b.sanity_flags
        assert a.summary == b.summary

    def test_triage_rejects_invalid_patient(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PatientInput(
                Age=10,
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
