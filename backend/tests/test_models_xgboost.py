"""Smoke tests for the XGBoost + Optuna wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from cardiorisk.models.base import MODEL_NAMES, ModelWrapper
from cardiorisk.models.xgboost_model import XGBoostModel, build_xgboost


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    from cardiorisk.data.preprocess import clean_for_modelling
    from cardiorisk.data.synthetic import generate_fixture

    rows = generate_fixture(n=100, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])
    return X, y


def test_build_xgboost_returns_wrapper():
    m = build_xgboost(n_trials=1)
    assert isinstance(m, XGBoostModel)
    assert m.model_name == "xgboost"
    assert m.model_name in MODEL_NAMES


def test_xgboost_is_sklearn_classifier():
    assert is_classifier(XGBoostModel(n_trials=1))


def test_xgboost_satisfies_modelwrapper_protocol():
    assert isinstance(XGBoostModel(n_trials=1), ModelWrapper)


def test_fit_and_predict_proba(small_dataset):
    X, y = small_dataset
    m = build_xgboost(n_trials=1).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_returns_class_labels(small_dataset):
    X, y = small_dataset
    m = build_xgboost(n_trials=1).fit(X, y)
    pred = m.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})


def test_best_score_is_a_finite_auroc(small_dataset):
    X, y = small_dataset
    m = build_xgboost(n_trials=1).fit(X, y)
    assert np.isfinite(m.best_score_)
    assert 0.0 <= m.best_score_ <= 1.0


def test_best_params_contains_expected_keys(small_dataset):
    X, y = small_dataset
    m = build_xgboost(n_trials=1).fit(X, y)
    expected_keys = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
    }
    assert expected_keys.issubset(set(m.best_params_.keys()))


def test_predict_proba_before_fit_raises():
    X = pd.DataFrame({"Age": [50.0]})
    with pytest.raises(RuntimeError, match="must be fit"):
        XGBoostModel(n_trials=1).predict_proba(X)


def test_determinism_under_seed(small_dataset):
    """Same Optuna seed + same data -> same best params -> same predictions.

    Optuna's TPE sampler is seeded, the inner CV is seeded, and XGBoost is
    seeded; all three of those plus the same data should be enough for
    bit-exact reproducibility of the best-params choice.
    """
    X, y = small_dataset
    m1 = build_xgboost(n_trials=2).fit(X, y)
    m2 = build_xgboost(n_trials=2).fit(X, y)
    assert m1.best_params_ == m2.best_params_
    np.testing.assert_array_equal(m1.predict_proba(X), m2.predict_proba(X))
