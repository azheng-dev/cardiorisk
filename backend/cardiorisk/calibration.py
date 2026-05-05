"""Post-hoc probability calibration on the within-fold calibration slice.

Per :doc:`../../docs/adr/006-risk-model-architecture.md` and
:doc:`../../docs/research/04-revised-design.md` §3.5, every LODO fold's
training rows are split 80/10/10 into train / val / **calibration**.
The calibration slice is held out specifically to fit a *post-hoc*
calibrator on top of an already-fitted base model, decoupling the
calibration data from both the training and evaluation paths.

Two methods supported (see ADR-006 + the user's Phase-2.3 decision):

- ``"isotonic"`` for XGBoost — non-parametric monotonic regression;
  more flexible, needs more calibration data.
- ``"sigmoid"`` (Platt) for L1 LR — parametric one-parameter logistic
  fit; more stable on small calibration slices like ours (~60 rows
  per fold).

TabPFN is calibrated by construction and does not pass through this
module.

Implementation: thin wrapper around sklearn's
:class:`~sklearn.calibration.CalibratedClassifierCV` with
:class:`~sklearn.frozen.FrozenEstimator` so the underlying model is
*not* refit during calibration. (sklearn deprecated ``cv='prefit'`` in
1.6 in favour of ``FrozenEstimator``; we follow the modern API.)
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

CalibrationMethod = Literal["isotonic", "sigmoid"]

#: Default per-model calibration choices from the Phase-2.3 plan.
DEFAULT_METHOD_FOR_MODEL: Final[dict[str, CalibrationMethod]] = {
    "xgboost": "isotonic",
    "lr": "sigmoid",
    # tabpfn deliberately absent: native calibration, no wrapper needed.
}


def calibrate(
    fitted_estimator: ClassifierMixin,
    X_calib: npt.ArrayLike,
    y_calib: npt.ArrayLike,
    *,
    method: CalibrationMethod = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap ``fitted_estimator`` in a post-hoc calibrator fit on (X_calib, y_calib).

    The base estimator is **not** refit (we wrap it in
    :class:`~sklearn.frozen.FrozenEstimator`). Only the post-hoc
    calibrator (isotonic regression or Platt scaling) sees the
    calibration slice.

    Parameters
    ----------
    fitted_estimator : sklearn-compatible classifier
        Must already be fitted on the training slice. Must implement
        ``predict_proba``.
    X_calib, y_calib : array-like
        The 10% calibration slice from
        :func:`cardiorisk.features.cv.within_fold_split`.
    method : {'isotonic', 'sigmoid'}
        Calibration method. ``'isotonic'`` for XGBoost, ``'sigmoid'``
        (Platt) for L1 LR per the Phase-2.3 design.

    Returns
    -------
    CalibratedClassifierCV
        A new estimator whose ``predict_proba`` returns calibrated
        probabilities. The original ``fitted_estimator`` is preserved
        unchanged inside ``FrozenEstimator``.
    """
    if method not in ("isotonic", "sigmoid"):
        raise ValueError(f"method must be 'isotonic' or 'sigmoid'; got {method!r}")
    if not hasattr(fitted_estimator, "predict_proba"):
        raise TypeError(
            f"fitted_estimator must implement predict_proba; got {type(fitted_estimator).__name__}"
        )

    y = np.asarray(y_calib).ravel()
    if np.unique(y).size < 2:
        raise ValueError(
            "calibration slice must contain both classes; got "
            f"{np.unique(y).tolist()}. Increase the calibration slice size or "
            "stratify the upstream within-fold split."
        )

    frozen = FrozenEstimator(fitted_estimator)
    calibrated = CalibratedClassifierCV(estimator=frozen, method=method)
    calibrated.fit(X_calib, y)
    return calibrated


def calibrate_for_model(
    fitted_estimator: ClassifierMixin,
    X_calib: npt.ArrayLike,
    y_calib: npt.ArrayLike,
    *,
    model_name: str,
) -> ClassifierMixin | CalibratedClassifierCV:
    """Apply the model's design-doc-mandated calibration method.

    Falls back to returning the unwrapped estimator if the model is
    listed as natively calibrated (currently just TabPFN).
    """
    if model_name not in DEFAULT_METHOD_FOR_MODEL:
        # TabPFN, or any future model declared natively-calibrated.
        return fitted_estimator
    method = DEFAULT_METHOD_FOR_MODEL[model_name]
    return calibrate(fitted_estimator, X_calib, y_calib, method=method)


__all__ = [
    "DEFAULT_METHOD_FOR_MODEL",
    "BaseEstimator",
    "CalibrationMethod",
    "calibrate",
    "calibrate_for_model",
]
