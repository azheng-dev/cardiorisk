"""Smoke tests for :mod:`cardiorisk.explainability.kernel_shap`.

KernelSHAP is slow even at smoke scale. We pin the smallest realistic
budget (``SMOKE_BACKGROUND_K`` = 5, ``SMOKE_NSAMPLES`` = 16) and
explain only a handful of test rows so the suite stays under ~15s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.data.preprocess import clean_for_modelling
from cardiorisk.data.synthetic import generate_fixture
from cardiorisk.explainability.encoder import fit_encoder
from cardiorisk.explainability.kernel_shap import (
    SMOKE_BACKGROUND_K,
    SMOKE_NSAMPLES,
    KernelSHAPResult,
    explain_with_kernel_shap,
)
from cardiorisk.models.lr import build_lr


@pytest.fixture(scope="module")
def fitted_lr_and_data():
    """Fit an LR on 80 synthetic rows and return (model, X_train, X_test)."""
    rows = generate_fixture(n=120, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])

    train = X.iloc[:80]
    test = X.iloc[80:90]
    model = build_lr().fit(train, y[:80])
    return model, train, test


def test_kernel_shap_smoke_returns_result(fitted_lr_and_data):
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)

    result = explain_with_kernel_shap(
        predict_proba=model.predict_proba,
        encoded_space=enc,
        X_train=X_train,
        X_test=X_test,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
    )
    assert isinstance(result, KernelSHAPResult)


def test_kernel_shap_shape_matches_encoded_space(fitted_lr_and_data):
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)

    result = explain_with_kernel_shap(
        predict_proba=model.predict_proba,
        encoded_space=enc,
        X_train=X_train,
        X_test=X_test,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
    )
    assert result.shap_values_encoded.shape == (len(X_test), enc.n_columns)
    assert result.shap_values_raw.shape == (len(X_test), enc.n_groups)


def test_kernel_shap_aggregation_is_consistent(fitted_lr_and_data):
    """Per-raw SHAP must equal the encoder's aggregation of per-encoded SHAP."""
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)

    result = explain_with_kernel_shap(
        predict_proba=model.predict_proba,
        encoded_space=enc,
        X_train=X_train,
        X_test=X_test,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
    )
    expected_raw = enc.aggregate_shap(result.shap_values_encoded)
    np.testing.assert_allclose(result.shap_values_raw, expected_raw, atol=1e-12)


def test_kernel_shap_mean_abs_dict(fitted_lr_and_data):
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)

    result = explain_with_kernel_shap(
        predict_proba=model.predict_proba,
        encoded_space=enc,
        X_train=X_train,
        X_test=X_test,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
    )
    means = result.mean_abs_per_raw_feature
    assert set(means.keys()) == set(enc.raw_feature_names)
    assert all(v >= 0 for v in means.values())


def test_kernel_shap_rejects_zero_background(fitted_lr_and_data):
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)
    with pytest.raises(ValueError, match="background_k"):
        explain_with_kernel_shap(
            predict_proba=model.predict_proba,
            encoded_space=enc,
            X_train=X_train,
            X_test=X_test,
            background_k=0,
            nsamples=SMOKE_NSAMPLES,
        )


def test_kernel_shap_rejects_zero_nsamples(fitted_lr_and_data):
    model, X_train, X_test = fitted_lr_and_data
    enc = fit_encoder(X_train)
    with pytest.raises(ValueError, match="nsamples"):
        explain_with_kernel_shap(
            predict_proba=model.predict_proba,
            encoded_space=enc,
            X_train=X_train,
            X_test=X_test,
            background_k=SMOKE_BACKGROUND_K,
            nsamples=0,
        )


def test_kernel_shap_rejects_empty_test(fitted_lr_and_data):
    model, X_train, _ = fitted_lr_and_data
    enc = fit_encoder(X_train)
    with pytest.raises(ValueError, match="at least one row"):
        explain_with_kernel_shap(
            predict_proba=model.predict_proba,
            encoded_space=enc,
            X_train=X_train,
            X_test=X_train.iloc[0:0],
            background_k=SMOKE_BACKGROUND_K,
            nsamples=SMOKE_NSAMPLES,
        )
