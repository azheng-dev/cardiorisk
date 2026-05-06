"""Smoke tests for the Phase-2.4 Honours-Ensemble PyTorch wrapper.

Same shape as ``test_models_lr.py`` / ``test_models_xgboost.py`` /
``test_models_tabicl.py`` so the per-wrapper test contract reads
identically across all four v1 models.

Determinism note: the four PyTorch sub-networks (DNN + CNN + LSTM +
BiLSTM) are reproducible to ~1e-6 across CPU runs at the same seed
(see :mod:`cardiorisk.models.ensemble` docstring). We do *not* enable
``torch.use_deterministic_algorithms(True)`` because it disables CPU
kernels we depend on; ``np.testing.assert_allclose`` with
``atol=1e-5`` is the right tolerance for this comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from cardiorisk.models.base import MODEL_NAMES, ModelWrapper
from cardiorisk.models.ensemble import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS,
    SMOKE_N_EPOCHS,
    EnsembleModel,
    build_ensemble,
)


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


def test_build_ensemble_returns_ensemble_model() -> None:
    m = build_ensemble()
    assert isinstance(m, EnsembleModel)
    assert m.model_name == "ensemble"
    assert m.model_name in MODEL_NAMES


def test_default_constants_match_honours_notebook() -> None:
    """Honours notebook (Data_Pre-processing.ipynb cell 55) hyperparameters."""
    assert DEFAULT_N_EPOCHS == 100
    assert DEFAULT_BATCH_SIZE == 32
    assert DEFAULT_LR == 1e-3
    assert DEFAULT_DROPOUT == 0.2


def test_smoke_epochs_is_one() -> None:
    """Smoke mode flows gradients through every layer without converging."""
    assert SMOKE_N_EPOCHS == 1


def test_ensemble_is_sklearn_classifier() -> None:
    """Required for CalibratedClassifierCV to wrap us via FrozenEstimator."""
    assert is_classifier(EnsembleModel())


def test_ensemble_satisfies_modelwrapper_protocol() -> None:
    m = EnsembleModel()
    assert isinstance(m, ModelWrapper)


def test_fit_and_predict_proba(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_class_labels(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)
    pred = m.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})
    assert pred.shape == (len(y),)


def test_predict_proba_before_fit_raises() -> None:
    X = pd.DataFrame({"Age": [50.0]})
    with pytest.raises(RuntimeError, match="must be fit"):
        EnsembleModel().predict_proba(X)


def test_predict_before_fit_raises() -> None:
    X = pd.DataFrame({"Age": [50.0]})
    with pytest.raises(RuntimeError, match="must be fit"):
        EnsembleModel().predict(X)


def test_classes_attribute_after_fit(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)
    np.testing.assert_array_equal(m.classes_, np.array([0, 1]))


def test_n_features_in_after_fit(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    """The WOA preprocessing pipeline expands categoricals + indicators."""
    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)
    assert m.n_features_in_ > 0
    # WOA pipeline = OHE(categoricals) + scaled(continuous + binary) + indicator
    # passthrough; on the synthetic schema this expands to ~25 features.
    assert 10 < m.n_features_in_ < 50


def test_four_submodels_after_fit(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Mean-averaged ensemble must hold exactly four trained sub-networks."""
    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)
    assert len(m.submodels_) == 4


def test_predict_proba_is_mean_of_four_submodel_outputs(
    small_dataset: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Audit: the wrapper is a *mean* (not a learned meta-model) of the four sigmoids.

    Honest reproduction check — the Honours notebook explicitly does
    ``np.mean(stacked_predictions, axis=-1)`` after stacking the four
    sub-network outputs (see ``Data_Pre-processing.ipynb`` cell 55).
    If a future refactor accidentally introduces a learned meta-model,
    this test fails loudly.
    """
    import torch

    X, y = small_dataset
    m = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y)

    X_dense_unknown = m.preprocessor_.transform(X)
    X_dense = np.asarray(X_dense_unknown, dtype=np.float64)
    x_dnn, x_cnn, x_lstm, _ = m._to_tensors(X_dense, None)
    dnn, cnn, lstm, bilstm = m.submodels_
    with torch.no_grad():
        manual = (
            torch.sigmoid(dnn(x_dnn)).numpy().ravel()
            + torch.sigmoid(cnn(x_cnn)).numpy().ravel()
            + torch.sigmoid(lstm(x_lstm)).numpy().ravel()
            + torch.sigmoid(bilstm(x_lstm)).numpy().ravel()
        ) / 4.0
    via_wrapper = m.predict_proba(X)[:, 1]
    np.testing.assert_allclose(via_wrapper, manual, atol=1e-6)


def test_determinism_under_seed(small_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Two fits with the same seed produce identical predictions to ~1e-5."""
    X, y = small_dataset
    p1 = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y).predict_proba(X)
    p2 = build_ensemble(n_epochs=SMOKE_N_EPOCHS).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
