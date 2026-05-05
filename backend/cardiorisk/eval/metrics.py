"""Discrimination + calibration scalar metrics for binary risk models.

All functions take ``y_true`` (0/1 labels) and ``y_proba`` (predicted
positive-class probability) as 1-D numpy arrays / pandas Series and
return scalars or small dataclasses. Per :doc:`../../../docs/research/04-revised-design.md`
§5.1 the headline metric set is:

- AUROC (discrimination)
- AUPRC (discrimination at operating point under class imbalance)
- Brier score (proper scoring rule combining discrimination + calibration)
- Calibration slope + intercept (numeric calibration)
- Sensitivity at 85% specificity (clinical operating point)
- Sensitivity at 90% specificity (stricter operating point)

The :func:`headline_metrics` convenience function returns all six in a
:class:`HeadlineMetrics` dataclass, used by the per-fold reporter and
by the cross-model results writeup in 2.3b.

Calibration semantics:

A *perfectly calibrated* model has calibration slope = 1.0 and
calibration intercept = 0.0 in the logistic-regression-on-logits sense
([Steyerberg et al. 2010](https://academic.oup.com/eurheartj/article/35/29/1925/2293256)).
A slope < 1 means the model's probabilities are too extreme (over-
confident); a slope > 1 means too conservative. Intercept ≠ 0 indicates
systematic over- or under-prediction of the base rate.

Sensitivity-at-specificity uses the *largest* sensitivity achievable
while specificity ≥ target — clinically conservative; ties go in the
patient's favour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

#: Default operating-point specificities reported in the headline table.
DEFAULT_SPECIFICITY_TARGETS: Final[tuple[float, ...]] = (0.85, 0.90)

#: Probability clip used before logit() to avoid -inf / +inf at p=0 or p=1.
#: Matches sklearn's brier_score_loss internal handling.
_PROBA_CLIP: Final[float] = 1e-15

#: ``C`` value used to make the calibration logistic regression effectively
#: unregularised (see :func:`calibration_slope_intercept`).
_CALIB_C: Final[float] = 1e10


def _validate_inputs(
    y_true: npt.ArrayLike, y_proba: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D float64 arrays, validate shape and class set."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_proba shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("y_true is empty; need at least one row to compute metrics")
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0 and 1; got values {unique.tolist()}")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("y_proba must be in [0, 1]")
    return y.astype(np.int64, copy=False), p


def auroc(y_true: npt.ArrayLike, y_proba: npt.ArrayLike) -> float:
    """Area under the ROC curve. NaN if only one class present in ``y_true``."""
    y, p = _validate_inputs(y_true, y_proba)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc(y_true: npt.ArrayLike, y_proba: npt.ArrayLike) -> float:
    """Area under the precision-recall curve (average precision)."""
    y, p = _validate_inputs(y_true, y_proba)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y_true: npt.ArrayLike, y_proba: npt.ArrayLike) -> float:
    """Brier score = mean squared error between probability and outcome.

    Lower is better. A model that always predicts the base rate has Brier
    = base_rate * (1 - base_rate); a model that always predicts 0.5 has
    Brier = 0.25. Strictly proper scoring rule; jointly penalises poor
    discrimination *and* poor calibration.
    """
    y, p = _validate_inputs(y_true, y_proba)
    return float(brier_score_loss(y, p))


@dataclass(frozen=True)
class CalibrationFit:
    """Slope + intercept from a logistic regression of y on logit(p)."""

    slope: float
    intercept: float


def calibration_slope_intercept(y_true: npt.ArrayLike, y_proba: npt.ArrayLike) -> CalibrationFit:
    """Fit logistic regression of ``y`` on ``logit(p)``; return slope + intercept.

    Perfectly calibrated: slope=1, intercept=0. Slope <1 = overconfident
    probabilities (too extreme); slope >1 = underconfident; |intercept|>0
    = systematic miscalibration of the base rate.

    Returns ``CalibrationFit(nan, nan)`` if only one class is present
    (the regression is undefined).
    """
    y, p = _validate_inputs(y_true, y_proba)
    if np.unique(y).size < 2:
        return CalibrationFit(slope=float("nan"), intercept=float("nan"))
    p_clipped = np.clip(p, _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    # Effectively no regularisation: at C=1e10 the L2 contribution is 1e-10,
    # well below the L-BFGS convergence tolerance, so the fit recovers the
    # MLE. sklearn 1.8 deprecated `penalty=None` and routes `C=np.inf` to
    # the same deprecation path; a finite-but-huge C avoids the warning
    # while preserving the unregularised semantics we want for calibration.
    lr = LogisticRegression(C=_CALIB_C, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p, y)
    return CalibrationFit(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def sensitivity_at_specificity(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    target_specificity: float,
) -> float:
    """Largest sensitivity achievable while specificity >= target.

    Walks the ROC curve, finds all thresholds whose specificity meets the
    target, returns the maximum sensitivity among them. NaN if only one
    class is present.

    Conservative tie-breaking: when multiple thresholds tie on
    specificity, the most-sensitive is chosen (favouring detection over
    rule-out).
    """
    if not 0.0 < target_specificity < 1.0:
        raise ValueError(f"target_specificity must be in (0, 1); got {target_specificity}")
    y, p = _validate_inputs(y_true, y_proba)
    if np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    spec = 1.0 - fpr
    eligible = spec >= target_specificity
    if not eligible.any():
        return 0.0
    return float(tpr[eligible].max())


@dataclass(frozen=True)
class HeadlineMetrics:
    """All six per-fold headline metrics from `04-revised-design.md` §5.1."""

    auroc: float
    auprc: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    sensitivity_at_85_spec: float
    sensitivity_at_90_spec: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "brier": self.brier,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
            "sensitivity_at_85_spec": self.sensitivity_at_85_spec,
            "sensitivity_at_90_spec": self.sensitivity_at_90_spec,
        }


def headline_metrics(y_true: npt.ArrayLike, y_proba: npt.ArrayLike) -> HeadlineMetrics:
    """Compute all six headline metrics in one pass.

    Used by the Phase-2.3b training driver to produce one
    :class:`HeadlineMetrics` per LODO fold per model.
    """
    cal = calibration_slope_intercept(y_true, y_proba)
    return HeadlineMetrics(
        auroc=auroc(y_true, y_proba),
        auprc=auprc(y_true, y_proba),
        brier=brier(y_true, y_proba),
        calibration_slope=cal.slope,
        calibration_intercept=cal.intercept,
        sensitivity_at_85_spec=sensitivity_at_specificity(y_true, y_proba, 0.85),
        sensitivity_at_90_spec=sensitivity_at_specificity(y_true, y_proba, 0.90),
    )
