"""Tests for the risk agent (mock fallback + artefact loader)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from cardiorisk.agents.risk import (
    MockRiskClassifier,
    _band,
    _patient_to_frame,
    run_risk,
)
from cardiorisk.agents.state import PatientInput
from cardiorisk.data.preprocess import MISSINGNESS_INDICATOR_COLUMNS


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


class TestBand:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.0, "low"),
            (0.04, "low"),
            (0.05, "low"),
            (0.06, "intermediate"),
            (0.099, "intermediate"),
            (0.10, "high"),
            (0.5, "high"),
        ],
    )
    def test_band_partitions(self, p: float, expected: str) -> None:
        assert _band(p, low=0.05, high=0.10) == expected


class TestPatientToFrame:
    def test_adds_missingness_indicators(self) -> None:
        df = _patient_to_frame(_patient())
        for col in MISSINGNESS_INDICATOR_COLUMNS:
            assert f"{col}_was_missing" in df.columns

    def test_cholesterol_zero_becomes_nan(self) -> None:
        df = _patient_to_frame(_patient(Cholesterol=0))
        assert df["Cholesterol"].isna().iloc[0]


class TestMockRiskClassifier:
    def test_predict_proba_in_range(self) -> None:
        m = MockRiskClassifier()
        for p in [
            _patient(),
            _patient(Age=78, ExerciseAngina="Y", ST_Slope="Down"),
            _patient(Age=30, Sex="F", ExerciseAngina="N", ST_Slope="Up"),
        ]:
            prob = m.predict_proba(p)
            assert 0.0 <= prob <= 1.0

    def test_high_risk_archetype_higher_than_low(self) -> None:
        m = MockRiskClassifier()
        low = m.predict_proba(_patient(Age=35, Sex="F", ExerciseAngina="N", ST_Slope="Up"))
        high = m.predict_proba(
            _patient(Age=78, Sex="M", ChestPainType="ASY", ExerciseAngina="Y", ST_Slope="Down")
        )
        assert high > low

    def test_feature_attributions_sorted_by_abs_contribution(self) -> None:
        m = MockRiskClassifier()
        attribs = m.feature_attributions(_patient())
        assert len(attribs) >= 1
        for a, b in itertools.pairwise(attribs):
            assert abs(a.contribution) >= abs(b.contribution)

    def test_feature_attributions_truncated_to_six_via_run_risk(self, tmp_path: Path) -> None:
        empty_models = tmp_path / "models"
        empty_models.mkdir()
        result = run_risk(_patient(), models_dir=empty_models)
        assert len(result.top_attributions) <= 6


class TestRunRisk:
    def test_falls_back_to_mock_when_no_artefact(self, tmp_path: Path) -> None:
        empty_models = tmp_path / "models"
        empty_models.mkdir()
        result = run_risk(_patient(), models_dir=empty_models)
        assert result.model_artefact_present is False
        assert result.model_name == "mock-risk-v1"
        assert result.risk_band in ("low", "intermediate", "high")
        assert 0.0 <= result.probability <= 1.0
        assert "%" in result.summary

    def test_returns_risk_band_for_high_archetype(self, tmp_path: Path) -> None:
        empty_models = tmp_path / "models"
        empty_models.mkdir()
        result = run_risk(
            _patient(
                Age=78,
                Sex="M",
                ChestPainType="ASY",
                ExerciseAngina="Y",
                ST_Slope="Down",
                Oldpeak=3.0,
                MaxHR=100,
            ),
            models_dir=empty_models,
        )
        assert result.risk_band in ("intermediate", "high")

    def test_run_risk_is_deterministic(self, tmp_path: Path) -> None:
        empty_models = tmp_path / "models"
        empty_models.mkdir()
        a = run_risk(_patient(), models_dir=empty_models)
        b = run_risk(_patient(), models_dir=empty_models)
        assert a.probability == b.probability
        assert a.risk_band == b.risk_band

    def test_summary_mentions_band_and_model(self, tmp_path: Path) -> None:
        empty_models = tmp_path / "models"
        empty_models.mkdir()
        result = run_risk(_patient(), models_dir=empty_models)
        assert result.risk_band in result.summary
        assert "mock-risk-v1" in result.summary
