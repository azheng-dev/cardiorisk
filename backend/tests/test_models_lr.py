"""Smoke tests for the L1 LR + RCS wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from cardiorisk.models.base import MODEL_NAMES, ModelWrapper
from cardiorisk.models.lr import DEFAULT_C_GRID, LRModel, build_lr


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """A 100-row synthetic dataset matching the cleaned HFP schema."""
    from cardiorisk.data.preprocess import clean_for_modelling
    from cardiorisk.data.synthetic import generate_fixture

    rows = generate_fixture(n=100, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])
    return X, y


def test_build_lr_returns_lrmodel():
    m = build_lr()
    assert isinstance(m, LRModel)
    assert m.model_name == "lr"
    assert m.model_name in MODEL_NAMES


def test_lrmodel_is_sklearn_classifier():
    """Required for CalibratedClassifierCV to wrap us via FrozenEstimator."""
    assert is_classifier(LRModel())


def test_lrmodel_satisfies_modelwrapper_protocol():
    m = LRModel()
    assert isinstance(m, ModelWrapper)


def test_fit_and_predict_proba(small_dataset):
    X, y = small_dataset
    m = build_lr().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_returns_class_labels(small_dataset):
    X, y = small_dataset
    m = build_lr().fit(X, y)
    pred = m.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})
    assert pred.shape == (len(y),)


def test_predict_proba_before_fit_raises():
    X = pd.DataFrame({"Age": [50.0]})
    with pytest.raises(RuntimeError, match="must be fit"):
        LRModel().predict_proba(X)


def test_grid_search_picks_a_c_in_grid(small_dataset):
    X, y = small_dataset
    m = build_lr().fit(X, y)
    assert m.best_c_ in DEFAULT_C_GRID


def test_determinism_under_seed(small_dataset):
    """Two fits with the same seed produce identical predictions."""
    X, y = small_dataset
    p1 = build_lr().fit(X, y).predict_proba(X)
    p2 = build_lr().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_classes_attribute_after_fit(small_dataset):
    X, y = small_dataset
    m = build_lr().fit(X, y)
    np.testing.assert_array_equal(m.classes_, np.array([0, 1]))
