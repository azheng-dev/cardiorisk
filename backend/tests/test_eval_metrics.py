"""Tests for cardiorisk.eval.metrics.

Mostly closed-form checks against analytically known answers (perfect
predictor, random predictor, base-rate predictor) so the tests double
as documentation of what each metric should be.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.eval.metrics import (
    CalibrationFit,
    HeadlineMetrics,
    auprc,
    auroc,
    brier,
    calibration_slope_intercept,
    headline_metrics,
    sensitivity_at_specificity,
)


def _balanced_labels(n: int = 100) -> np.ndarray:
    return np.array([0] * (n // 2) + [1] * (n // 2))


# ---------------------------------------------------------------- AUROC


def test_auroc_perfect_predictor_is_one() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    assert auroc(y, p) == pytest.approx(1.0)


def test_auroc_constant_prediction_is_one_half() -> None:
    y = _balanced_labels()
    p = np.full_like(y, 0.5, dtype=float)
    assert auroc(y, p) == pytest.approx(0.5)


def test_auroc_inverted_predictor_is_zero() -> None:
    y = _balanced_labels()
    p = (1 - y).astype(float) * 0.9 + 0.05
    assert auroc(y, p) == pytest.approx(0.0)


def test_auroc_returns_nan_when_only_one_class_present() -> None:
    y = np.zeros(20, dtype=int)
    p = np.linspace(0, 1, 20)
    assert np.isnan(auroc(y, p))


# ---------------------------------------------------------------- AUPRC


def test_auprc_perfect_predictor_is_one() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    assert auprc(y, p) == pytest.approx(1.0)


def test_auprc_returns_nan_when_only_one_class_present() -> None:
    y = np.ones(20, dtype=int)
    p = np.linspace(0, 1, 20)
    assert np.isnan(auprc(y, p))


# ---------------------------------------------------------------- Brier


def test_brier_perfect_predictor_is_zero() -> None:
    y = _balanced_labels()
    p = y.astype(float)
    assert brier(y, p) == pytest.approx(0.0)


def test_brier_constant_half_is_one_quarter() -> None:
    """A model that predicts 0.5 for every row scores Brier = 0.25
    regardless of base rate or label distribution."""
    y = _balanced_labels()
    p = np.full_like(y, 0.5, dtype=float)
    assert brier(y, p) == pytest.approx(0.25)


def test_brier_constant_prediction_at_base_rate_equals_variance() -> None:
    """A model predicting the base rate scores Brier = p*(1-p) where p
    is the base rate. Strictly proper scoring rule check."""
    p_base = 0.30
    y = np.array([1] * int(p_base * 100) + [0] * (100 - int(p_base * 100)))
    p = np.full_like(y, p_base, dtype=float)
    assert brier(y, p) == pytest.approx(p_base * (1 - p_base))


# ---------------------------------------------------------------- calibration


def test_calibration_perfect_logit_yields_slope_one_intercept_zero() -> None:
    """If predicted probabilities perfectly match outcome rates within
    each logit bin, calibration slope is 1 and intercept is 0."""
    rng = np.random.default_rng(0)
    n = 5000
    logit = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    fit = calibration_slope_intercept(y, p)
    assert fit.slope == pytest.approx(1.0, abs=0.1)
    assert fit.intercept == pytest.approx(0.0, abs=0.1)


def test_calibration_returns_nan_when_only_one_class_present() -> None:
    y = np.zeros(50, dtype=int)
    p = np.linspace(0.1, 0.9, 50)
    fit = calibration_slope_intercept(y, p)
    assert isinstance(fit, CalibrationFit)
    assert np.isnan(fit.slope)
    assert np.isnan(fit.intercept)


# ---------------------------------------------------------------- sens@spec


def test_sensitivity_at_full_specificity_for_perfect_predictor_is_one() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    assert sensitivity_at_specificity(y, p, 0.85) == pytest.approx(1.0)
    assert sensitivity_at_specificity(y, p, 0.99) == pytest.approx(1.0)


def test_sensitivity_at_specificity_constant_predictor_can_only_meet_low_specs() -> None:
    """A constant 0.5 predictor classifies everyone the same; specificity
    is either 0 (all positive) or 1 (all negative). At target_spec=0.85
    only the all-negative cut meets it, with sensitivity 0."""
    y = _balanced_labels()
    p = np.full_like(y, 0.5, dtype=float)
    assert sensitivity_at_specificity(y, p, 0.85) == pytest.approx(0.0)


def test_sensitivity_at_specificity_rejects_invalid_target() -> None:
    y = _balanced_labels()
    p = np.linspace(0, 1, len(y))
    with pytest.raises(ValueError, match="must be in"):
        sensitivity_at_specificity(y, p, 0.0)
    with pytest.raises(ValueError, match="must be in"):
        sensitivity_at_specificity(y, p, 1.0)


# ---------------------------------------------------------------- headline_metrics


def test_headline_metrics_returns_all_six_fields() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    hm = headline_metrics(y, p)
    assert isinstance(hm, HeadlineMetrics)
    assert set(hm.as_dict().keys()) == {
        "auroc",
        "auprc",
        "brier",
        "calibration_slope",
        "calibration_intercept",
        "sensitivity_at_85_spec",
        "sensitivity_at_90_spec",
    }


def test_headline_metrics_perfect_predictor_dominates_all_metrics() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    hm = headline_metrics(y, p)
    assert hm.auroc == pytest.approx(1.0)
    assert hm.auprc == pytest.approx(1.0)
    assert hm.sensitivity_at_85_spec == pytest.approx(1.0)
    assert hm.sensitivity_at_90_spec == pytest.approx(1.0)


# ---------------------------------------------------------------- input validation


def test_validation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        auroc(np.array([0, 1]), np.array([0.5]))


def test_validation_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        auroc(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_validation_rejects_probabilities_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        auroc(np.array([0, 1]), np.array([0.5, 1.5]))


def test_validation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        auroc(np.array([]), np.array([]))
