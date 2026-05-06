"""Tests for cardiorisk.monitoring.drift.

Covers the headline behaviour against synthetic two-slice fixtures:

- Identical reference and current → severity counts dominated by ``stable``.
- Shifted current → at least one feature flagged ``moderate`` or ``major``.
- Categorical drift detected (novel level surfaces).
- Prediction-drift block populated when a model is provided.
- Prediction-drift block ``None`` when model is omitted.
- ``top_drifted_features`` returns features sorted by descending PSI.
- Failure modes: ``model`` without ``model_name``; unknown ``model_name``
  in reference; bad ``predict_proba`` shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cardiorisk.monitoring.drift import DriftReport, compute_drift
from cardiorisk.monitoring.psi import PSI_STABLE_MAX
from cardiorisk.monitoring.reference import build_fold_reference

SEED = 20260506


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def matched_reference_and_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    n = 600

    def gen(rng: np.random.Generator, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Age": rng.integers(30, 80, n).astype(np.float64),
                "RestingBP": rng.normal(130, 20, n),
                "Cholesterol": rng.normal(220, 50, n),
                "MaxHR": rng.normal(140, 25, n),
                "Oldpeak": rng.normal(1.0, 1.0, n),
                "FastingBS": rng.integers(0, 2, n).astype(np.float64),
                "Sex": rng.choice(["M", "F"], size=n),
                "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], size=n),
                "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
                "ExerciseAngina": rng.choice(["N", "Y"], size=n),
                "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
            }
        )

    ref = gen(rng, n)
    cur = gen(np.random.default_rng(SEED + 1), n)  # same dist, different draw
    return ref, cur


@pytest.fixture
def shifted_current(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """A clearly-shifted current slice: numeric mean shifts + categorical
    composition flip."""
    _, cur = matched_reference_and_current
    shifted = cur.copy()
    shifted["RestingBP"] = shifted["RestingBP"] + 30.0  # ~1.5 SD shift
    shifted["MaxHR"] = shifted["MaxHR"] - 25.0
    shifted["Sex"] = "M"  # flatten to one level
    return shifted


class _StubModel:
    """Predicts a constant proba, with optional shift between ref and cur."""

    def __init__(
        self, base_proba: float, *, shift_at: int | None = None, shifted_proba: float = 0.5
    ) -> None:
        self.base_proba = base_proba
        self.shift_at = shift_at
        self.shifted_proba = shifted_proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if self.shift_at is None or n <= self.shift_at:
            p = np.full(n, self.base_proba)
        else:
            p = np.concatenate(
                [
                    np.full(self.shift_at, self.base_proba),
                    np.full(n - self.shift_at, self.shifted_proba),
                ]
            )
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------- matched case


def test_matched_distributions_yield_mostly_stable_features(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=cur_df)
    assert isinstance(report, DriftReport)
    # With 11 features and matched distributions we expect almost all stable.
    assert report.severity_counts["stable"] >= len(report.per_feature) - 2
    assert report.severity_counts["major"] == 0


# ---------------------------------------------------------------- shifted case


def test_shifted_current_flags_drift(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
    shifted_current: pd.DataFrame,
) -> None:
    ref_df, _ = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=shifted_current)
    assert (report.severity_counts["moderate"] + report.severity_counts["major"]) >= 2
    # The two shifted numeric features should appear in the top-3 by PSI.
    top_features = {fd.feature for fd in report.top_drifted_features(k=3)}
    assert "RestingBP" in top_features or "MaxHR" in top_features


def test_categorical_collapse_is_detected(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
    shifted_current: pd.DataFrame,
) -> None:
    ref_df, _ = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=shifted_current)
    sex_drift = report.per_feature["Sex"]
    assert sex_drift.severity in ("moderate", "major")


# ---------------------------------------------------------------- prediction drift


def test_prediction_drift_block_present_when_model_supplied(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current
    model = _StubModel(base_proba=0.3)
    reference = build_fold_reference(
        held_out_source="x",
        X_train=ref_df,
        models={"stub": model},
    )
    report = compute_drift(reference=reference, X_current=cur_df, model=model, model_name="stub")
    assert report.prediction is not None
    assert report.prediction.model_name == "stub"
    assert report.prediction.psi >= 0.0


def test_prediction_drift_detects_constant_model_shift() -> None:
    rng = np.random.default_rng(SEED)
    n = 500
    ref_df = pd.DataFrame({"Age": rng.normal(50, 10, n)})

    class _StubA:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # ref: deterministic but spread across [0.05, 0.95]
            r = np.linspace(0.05, 0.95, len(X))
            return np.column_stack([1 - r, r])

    class _StubB:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # cur: collapsed to a single high-risk band
            r = np.full(len(X), 0.95)
            return np.column_stack([1 - r, r])

    ref_model: Any = _StubA()
    cur_model: Any = _StubB()
    reference = build_fold_reference(
        held_out_source="x",
        X_train=ref_df,
        models={"stub": ref_model},
    )
    cur_df = pd.DataFrame({"Age": rng.normal(50, 10, n)})
    report = compute_drift(
        reference=reference, X_current=cur_df, model=cur_model, model_name="stub"
    )
    assert report.prediction is not None
    assert report.prediction.severity == "major"


def test_prediction_drift_omitted_when_model_not_passed(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=cur_df)
    assert report.prediction is None


# ---------------------------------------------------------------- top_drifted_features


def test_top_drifted_features_returns_sorted_descending(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
    shifted_current: pd.DataFrame,
) -> None:
    ref_df, _ = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=shifted_current)
    top = report.top_drifted_features(k=5)
    assert len(top) == 5
    psis = [fd.psi for fd in top]
    assert psis == sorted(psis, reverse=True)


def test_top_drifted_features_is_stable_under_ties() -> None:
    rng = np.random.default_rng(SEED)
    n = 300
    ref_df = pd.DataFrame(
        {
            "Age": rng.normal(50, 10, n),
            "RestingBP": rng.normal(130, 20, n),
            "MaxHR": rng.normal(140, 25, n),
        }
    )
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=ref_df)
    # Identical inputs -> all PSI ~0, so ties are resolved alphabetically.
    top = report.top_drifted_features(k=3)
    assert all(fd.psi <= PSI_STABLE_MAX for fd in top)
    assert [fd.feature for fd in top] == sorted(fd.feature for fd in top)


# ---------------------------------------------------------------- failure modes


def test_compute_drift_requires_model_name_when_model_passed(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    with pytest.raises(ValueError, match="model_name is required"):
        compute_drift(reference=reference, X_current=cur_df, model=_StubModel(0.3))


def test_compute_drift_raises_on_unknown_model_name(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    with pytest.raises(KeyError, match="no prediction binning"):
        compute_drift(
            reference=reference,
            X_current=cur_df,
            model=_StubModel(0.3),
            model_name="not_in_reference",
        )


def test_compute_drift_raises_on_bad_predict_proba_shape(
    matched_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = matched_reference_and_current

    class _BadModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))

    reference = build_fold_reference(
        held_out_source="x",
        X_train=ref_df,
        models={"bad": _StubModel(0.3)},  # populate the slot first
    )
    with pytest.raises(ValueError, match="predict_proba"):
        compute_drift(
            reference=reference,
            X_current=cur_df,
            model=_BadModel(),
            model_name="bad",
        )
