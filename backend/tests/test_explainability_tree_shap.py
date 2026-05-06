"""Tests for :mod:`cardiorisk.explainability.tree_shap`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from cardiorisk.calibration import calibrate
from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    clean_for_modelling,
)
from cardiorisk.data.synthetic import generate_fixture
from cardiorisk.explainability.encoder import INDICATOR_COLUMNS
from cardiorisk.explainability.tree_shap import (
    TreeSHAPResult,
    explain_xgboost_with_tree_shap,
)
from cardiorisk.models.xgboost_model import XGBoostModel, build_xgboost


@pytest.fixture(scope="module")
def fitted_xgb_and_test() -> tuple[XGBoostModel, pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows = generate_fixture(n=120, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])
    model = build_xgboost(n_trials=1).fit(X.iloc[:80], y[:80])
    return model, X.iloc[:80], X.iloc[80:90], y[:80]


def test_treeshap_smoke_returns_result(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    assert isinstance(out, TreeSHAPResult)


def test_treeshap_shape(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    n_post = len(model.pipeline_.named_steps["preprocess"].get_feature_names_out())
    assert out.shap_values_post.shape == (len(X_test), n_post)
    assert out.shap_values_raw.shape[0] == len(X_test)


def test_treeshap_raw_features_cover_expected(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    expected = (
        set(NUMERIC_COLUMNS)
        | set(BINARY_NUMERIC_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | set(INDICATOR_COLUMNS)
    )
    assert set(out.raw_feature_names) == expected


def test_treeshap_raw_equals_summed_post(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    np.testing.assert_allclose(
        out.shap_values_raw.sum(axis=1),
        out.shap_values_post.sum(axis=1),
        atol=1e-10,
    )


def test_treeshap_logit_reconstruction(fitted_xgb_and_test):
    """expected_value + sum(SHAP) ~ booster's raw log-odds output."""
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)

    # Get the raw log-odds the booster produces (uncalibrated).
    pre = model.pipeline_.named_steps["preprocess"]
    clf = model.pipeline_.named_steps["clf"]
    X_post = np.asarray(pre.transform(X_test), dtype=np.float64)
    raw_margin = np.asarray(clf.predict(X_post, output_margin=True), dtype=np.float64)
    reconstructed = out.expected_value + out.shap_values_post.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-4)


def test_treeshap_mean_abs_dict(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    means = out.mean_abs_per_raw_feature
    assert set(means.keys()) == set(out.raw_feature_names)
    assert all(v >= 0 for v in means.values())


def test_treeshap_unwraps_calibrated_classifier(fitted_xgb_and_test):
    """The calibrated artefact (CalibratedClassifierCV) must work too."""
    model, X_train, X_test, y_train = fitted_xgb_and_test
    cal = calibrate(model, X_train, y_train, method="isotonic")
    assert isinstance(cal, CalibratedClassifierCV)
    out_cal = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=cal, X_test=X_test)
    out_bare = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=model, X_test=X_test)
    # Same underlying booster -> identical SHAP values.
    np.testing.assert_array_equal(out_cal.shap_values_post, out_bare.shap_values_post)


def test_treeshap_unwraps_frozen_estimator(fitted_xgb_and_test):
    model, _, X_test, _ = fitted_xgb_and_test
    frozen = FrozenEstimator(model)
    out = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=frozen, X_test=X_test)
    assert out.shap_values_raw.shape[0] == len(X_test)


def test_treeshap_rejects_unfit_xgb():
    with pytest.raises(RuntimeError, match="must be fit"):
        explain_xgboost_with_tree_shap(
            calibrated_or_bare_xgb=XGBoostModel(n_trials=1),
            X_test=pd.DataFrame(),
        )


def test_treeshap_rejects_wrong_type():
    with pytest.raises(TypeError, match="XGBoostModel"):
        explain_xgboost_with_tree_shap(
            calibrated_or_bare_xgb="not a model",
            X_test=pd.DataFrame(),
        )
