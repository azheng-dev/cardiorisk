"""Tests for cardiorisk.eval.dca.

DCA has clean closed-form properties we can check exactly:

- Treat-none net benefit is identically zero.
- Treat-all net benefit at threshold p_t is
  ``prevalence - (1 - prevalence) * (p_t / (1 - p_t))``.
- A perfect predictor with thresholding matches treat-all when
  threshold <= prevalence (catches everyone correctly) but with no
  false positives, so its NB equals prevalence at all thresholds.
- For a binary classifier prediction, NB reduces to the published
  ``TP/N - FP/N * (p_t / (1 - p_t))`` form.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.eval.dca import (
    AUSCVDRISK_THRESHOLDS,
    DCACurve,
    decision_curve,
    net_benefit,
    net_benefit_treat_all,
)


def _balanced_labels(n: int = 100) -> np.ndarray:
    return np.array([0] * (n // 2) + [1] * (n // 2))


# ---------------------------------------------------------------- net_benefit


def test_net_benefit_perfect_predictor_at_low_threshold_equals_prevalence() -> None:
    """A perfect predictor (1 for positives, 0 for negatives) at any
    threshold p_t in (0, 1) catches all positives and no negatives, so
    NB = prevalence - 0 = prevalence."""
    y = _balanced_labels()
    p = y.astype(float)
    assert net_benefit(y, p, 0.05) == pytest.approx(0.5)
    assert net_benefit(y, p, 0.50) == pytest.approx(0.5)
    assert net_benefit(y, p, 0.95) == pytest.approx(0.5)


def test_net_benefit_matches_published_formula() -> None:
    """NB(p_t) = TP/N - (FP/N) * (p_t / (1-p_t)). Spot-check against
    a hand-computable example: 4 positives + 4 negatives, predictions
    [0.9, 0.8, 0.4, 0.3, 0.7, 0.6, 0.2, 0.1] for labels [1,1,1,1,0,0,0,0].
    At threshold 0.5: predicted positive = first two pos + first two neg.
    TP = 2, FP = 2, N = 8.
    NB = 2/8 - (2/8) * (0.5/0.5) = 0.25 - 0.25 = 0.0."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.6, 0.2, 0.1])
    assert net_benefit(y, p, 0.5) == pytest.approx(0.0)


def test_net_benefit_threshold_must_be_in_open_unit_interval() -> None:
    y = _balanced_labels()
    p = y.astype(float)
    with pytest.raises(ValueError, match="threshold must be in"):
        net_benefit(y, p, 0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        net_benefit(y, p, 1.0)


# ---------------------------------------------------------------- treat-all


def test_treat_all_at_threshold_equals_published_formula() -> None:
    """treat_all NB = prevalence - (1 - prevalence) * (p_t / (1 - p_t))."""
    prev = 0.30
    pt = 0.10
    expected = prev - (1 - prev) * (pt / (1 - pt))
    assert net_benefit_treat_all(prev, pt) == pytest.approx(expected)


def test_treat_all_at_threshold_equal_to_prevalence_is_zero() -> None:
    """When p_t = prevalence, treat-all and treat-none give equal NB
    (both zero in the standard derivation)."""
    prev = 0.20
    assert net_benefit_treat_all(prev, prev) == pytest.approx(0.0)


# ---------------------------------------------------------------- decision_curve


def test_decision_curve_returns_three_aligned_arrays() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    curve = decision_curve(y, p)
    assert isinstance(curve, DCACurve)
    n = len(curve.thresholds)
    assert curve.net_benefit_model.shape == (n,)
    assert curve.net_benefit_treat_all.shape == (n,)
    assert curve.net_benefit_treat_none.shape == (n,)


def test_decision_curve_treat_none_is_identically_zero() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.9 + 0.05
    curve = decision_curve(y, p)
    assert np.all(curve.net_benefit_treat_none == 0.0)


def test_decision_curve_perfect_model_dominates_treat_all_above_prevalence() -> None:
    """A perfect predictor's NB is constant = prevalence at every
    threshold, while treat-all's NB drops below prevalence as p_t
    increases past the prevalence point. So perfect-model is_useful_at
    is True everywhere except possibly at p_t == prevalence."""
    y = _balanced_labels(n=200)
    p = y.astype(float)
    curve = decision_curve(y, p)
    assert curve.is_useful_at(0.10)
    assert curve.is_useful_at(0.30)
    assert curve.is_useful_at(0.70)


def test_decision_curve_at_returns_all_three_policies() -> None:
    y = _balanced_labels()
    p = y.astype(float)
    curve = decision_curve(y, p)
    nb = curve.at(0.05)
    assert set(nb.keys()) == {"model", "treat_all", "treat_none"}
    assert nb["treat_none"] == 0.0


def test_decision_curve_uses_default_thresholds_if_unspecified() -> None:
    y = _balanced_labels()
    p = y.astype(float) * 0.5 + 0.25
    curve = decision_curve(y, p)
    # Default sweep is 1%-99% step 1%, so 99 points.
    assert len(curve.thresholds) == 99
    # AusCVDRisk thresholds are inside the default sweep.
    for t in AUSCVDRISK_THRESHOLDS:
        assert (np.abs(curve.thresholds - t) < 0.005).any()


def test_decision_curve_rejects_thresholds_at_boundary() -> None:
    y = _balanced_labels()
    p = y.astype(float)
    with pytest.raises(ValueError, match="thresholds must be in"):
        decision_curve(y, p, thresholds=np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="thresholds must be in"):
        decision_curve(y, p, thresholds=np.array([0.5, 1.0]))


# ---------------------------------------------------------------- input validation


def test_dca_validation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        net_benefit(np.array([0, 1]), np.array([0.5]), 0.1)


def test_dca_validation_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        net_benefit(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]), 0.1)


def test_treat_all_rejects_invalid_prevalence() -> None:
    with pytest.raises(ValueError, match="prevalence"):
        net_benefit_treat_all(-0.1, 0.5)
    with pytest.raises(ValueError, match="prevalence"):
        net_benefit_treat_all(1.1, 0.5)
