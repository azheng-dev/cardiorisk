"""Tests for cardiorisk.calibration.

Covers:

- Wrapper produces calibrated probabilities (predict_proba shape, range).
- Both isotonic and sigmoid methods fit successfully.
- The base estimator is preserved (FrozenEstimator does not refit it).
- Calibration improves Brier score on a known-miscalibrated input.
- calibrate_for_model dispatches to the right method per model_name.
- TabPFN passes through unwrapped (declared natively-calibrated).
- Failure modes: bad method, missing predict_proba, single-class calib.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk.calibration import (
    DEFAULT_METHOD_FOR_MODEL,
    calibrate,
    calibrate_for_model,
)
from cardiorisk.eval.metrics import brier


@pytest.fixture
def fitted_lr_and_calib_data() -> tuple[
    LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Fit a base LR on a train slice; provide a separate calibration
    slice; provide a held-out test slice for downstream Brier checks."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(0, 0.5, n) > 0).astype(int)

    X_train, X_calib, X_test = X[:400], X[400:500], X[500:]
    y_train, y_calib, y_test = y[:400], y[400:500], y[500:]

    base = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return base, X_calib, y_calib, X_test, y_test


# ---------------------------------------------------------------- calibrate basic


def test_calibrate_isotonic_returns_calibrated_classifier(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    cal = calibrate(base, X_calib, y_calib, method="isotonic")
    proba = cal.predict_proba(X_calib)
    assert proba.shape == (X_calib.shape[0], 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibrate_sigmoid_returns_calibrated_classifier(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    cal = calibrate(base, X_calib, y_calib, method="sigmoid")
    proba = cal.predict_proba(X_calib)
    assert proba.shape == (X_calib.shape[0], 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_calibrate_does_not_refit_base_estimator(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    """FrozenEstimator semantics: the base coefficients must be unchanged
    after calibration."""
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    coef_before = base.coef_.copy()
    intercept_before = base.intercept_.copy()
    _ = calibrate(base, X_calib, y_calib, method="sigmoid")
    assert np.allclose(base.coef_, coef_before)
    assert np.allclose(base.intercept_, intercept_before)


def test_calibrate_improves_brier_on_miscalibrated_predictor(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    """Train an overconfident base LR by exaggerating its raw scores,
    then calibrate. Calibrated Brier should beat uncalibrated Brier on
    the test slice.

    Subclass LogisticRegression so the result still implements all the
    sklearn-estimator surface that CalibratedClassifierCV inspects.
    """
    _base, X_calib, y_calib, X_test, y_test = fitted_lr_and_calib_data

    class OverconfidentLR(LogisticRegression):  # type: ignore[misc]
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = super().predict_proba(X)
            # Exponentiate to extremes (bad calibration; same ordering).
            num = p**3
            denom = num + (1 - p) ** 3
            return np.asarray(np.clip(num / denom, 1e-6, 1 - 1e-6))

    overconf = OverconfidentLR(max_iter=1000).fit(np.asarray(X_calib), np.asarray(y_calib))
    # Re-fit on a separate train slice would be cleaner; for the test
    # the calibration fit on its own slice is enough to demonstrate the
    # property because CalibratedClassifierCV freezes the base.
    p_uncal = overconf.predict_proba(X_test)[:, 1]

    cal = calibrate(overconf, X_calib, y_calib, method="sigmoid")
    p_cal = cal.predict_proba(X_test)[:, 1]

    assert brier(y_test, p_cal) < brier(y_test, p_uncal)


# ---------------------------------------------------------------- calibrate_for_model


def test_calibrate_for_model_xgboost_uses_isotonic(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    """Per the Phase-2.3 plan, xgboost gets isotonic. Verify the dispatcher
    routes correctly even though we use an LR as the stand-in classifier."""
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    assert DEFAULT_METHOD_FOR_MODEL["xgboost"] == "isotonic"
    cal = calibrate_for_model(base, X_calib, y_calib, model_name="xgboost")
    # Returned thing must be a calibrated classifier, not the bare base.
    assert hasattr(cal, "calibrated_classifiers_")


def test_calibrate_for_model_lr_uses_sigmoid(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    assert DEFAULT_METHOD_FOR_MODEL["lr"] == "sigmoid"
    cal = calibrate_for_model(base, X_calib, y_calib, model_name="lr")
    assert hasattr(cal, "calibrated_classifiers_")


def test_calibrate_for_model_tabpfn_passes_through_unwrapped(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    """TabPFN is declared natively-calibrated; calibrate_for_model
    returns the original estimator unchanged."""
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    out = calibrate_for_model(base, X_calib, y_calib, model_name="tabpfn")
    assert out is base


# ---------------------------------------------------------------- failure modes


def test_calibrate_rejects_unknown_method(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    base, X_calib, y_calib, _, _ = fitted_lr_and_calib_data
    with pytest.raises(ValueError, match="method"):
        calibrate(base, X_calib, y_calib, method="oxidative")  # type: ignore[arg-type]


def test_calibrate_rejects_estimator_without_predict_proba(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    _, X_calib, y_calib, _, _ = fitted_lr_and_calib_data

    class NoProba:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="predict_proba"):
        calibrate(NoProba(), X_calib, y_calib)


def test_calibrate_rejects_single_class_calibration_slice(
    fitted_lr_and_calib_data: tuple[
        LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    base, X_calib, _, _, _ = fitted_lr_and_calib_data
    y_one_class = np.zeros(X_calib.shape[0], dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        calibrate(base, X_calib, y_one_class)
