"""Smoke tests for the TabICL TFM wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from cardiorisk.models.base import MODEL_NAMES, ModelWrapper
from cardiorisk.models.tabicl import TabICLModel, build_tabicl


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    from cardiorisk.data.preprocess import clean_for_modelling
    from cardiorisk.data.synthetic import generate_fixture

    rows = generate_fixture(n=80, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])
    return X, y


def test_build_tabicl_returns_wrapper():
    m = build_tabicl()
    assert isinstance(m, TabICLModel)
    assert m.model_name == "tabicl"
    assert m.model_name in MODEL_NAMES


def test_tabicl_is_sklearn_classifier():
    assert is_classifier(TabICLModel())


def test_tabicl_satisfies_modelwrapper_protocol():
    assert isinstance(TabICLModel(), ModelWrapper)


def test_fit_and_predict_proba(small_dataset):
    X, y = small_dataset
    m = build_tabicl().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_returns_class_labels(small_dataset):
    X, y = small_dataset
    m = build_tabicl().fit(X, y)
    pred = m.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})


def test_predict_proba_before_fit_raises():
    X = pd.DataFrame({"Age": [50.0]})
    with pytest.raises(RuntimeError, match="must be fit"):
        TabICLModel().predict_proba(X)


def test_nan_passes_through(small_dataset):
    """The whole reason the TFM wrapper uses make_tabpfn_pipeline is that
    NaN should pass through to the model unmodified. Verify by injecting
    NaN into the test slice and confirming we still get valid probas
    (no crashes, no NaN output).
    """
    X, y = small_dataset
    m = build_tabicl().fit(X, y)
    X_with_nan = X.copy()
    X_with_nan.iloc[0, X_with_nan.columns.get_loc("Cholesterol")] = np.nan
    proba = m.predict_proba(X_with_nan)
    assert proba.shape == (len(y), 2)
    assert np.isfinite(proba).all()


def test_determinism_under_seed(small_dataset):
    """TabICL is deterministic given (model_path, random_state, device, input).

    With seed pinned + CPU device + same data, two fit/predict sequences
    should produce identical probabilities.
    """
    X, y = small_dataset
    p1 = build_tabicl().fit(X, y).predict_proba(X)
    p2 = build_tabicl().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)
