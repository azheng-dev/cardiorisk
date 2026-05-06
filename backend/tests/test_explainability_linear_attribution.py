"""Tests for :mod:`cardiorisk.explainability.linear_attribution`.

Two important properties are checked:

1. **Sum-back consistency.** Per-raw-feature SHAP equals the sum of
   the per-spline-basis SHAP values for that raw feature -- the
   property ADR-013 §"LR + RCS attribution detail" relies on for the
   cross-model comparison row.
2. **Reconstruction.** ``intercept + sum(shap_per_basis[i]) ==
   logit(predict_proba(x_i))`` for every test row, modulo float.
   This is the *exact-by-construction* property of additive SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    clean_for_modelling,
)
from cardiorisk.data.synthetic import generate_fixture
from cardiorisk.explainability.encoder import INDICATOR_COLUMNS
from cardiorisk.explainability.linear_attribution import (
    LinearAttributionResult,
    attribute_lr,
)
from cardiorisk.models.lr import LRModel, build_lr


@pytest.fixture(scope="module")
def fitted_lr_and_test() -> tuple[LRModel, pd.DataFrame]:
    rows = generate_fixture(n=120, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    df = clean_for_modelling(df)
    y = df["HeartDisease"].to_numpy()
    X = df.drop(columns=["HeartDisease", "source"])
    model = build_lr().fit(X.iloc[:80], y[:80])
    return model, X.iloc[80:90]


def test_attribute_lr_returns_result(fitted_lr_and_test):
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    assert isinstance(out, LinearAttributionResult)


def test_per_basis_shape(fitted_lr_and_test):
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    n_post = len(model.pipeline_.named_steps["preprocess"].get_feature_names_out())
    assert out.shap_per_basis.shape == (len(X_test), n_post)


def test_per_raw_groups_cover_expected_features(fitted_lr_and_test):
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    expected = (
        set(NUMERIC_COLUMNS)
        | set(BINARY_NUMERIC_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | set(INDICATOR_COLUMNS)
    )
    assert set(out.raw_feature_names) == expected


def test_per_raw_equals_summed_per_basis(fitted_lr_and_test):
    """The headline sum-back property: per-raw == sum of per-basis."""
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)

    # Row sums must agree to float precision -- both views are linear
    # combinations of the same quantities, just regrouped.
    np.testing.assert_allclose(
        out.shap_per_raw.sum(axis=1),
        out.shap_per_basis.sum(axis=1),
        atol=1e-10,
    )


def test_logit_reconstruction(fitted_lr_and_test):
    """Intercept + sum(per-basis SHAP) == logit(predict_proba) per row."""
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    proba = model.predict_proba(X_test)[:, 1]
    proba = np.clip(proba, 1e-9, 1 - 1e-9)
    logit = np.log(proba / (1 - proba))
    reconstructed = out.intercept + out.shap_per_basis.sum(axis=1)
    np.testing.assert_allclose(reconstructed, logit, atol=1e-8)


def test_mean_abs_per_raw_dict(fitted_lr_and_test):
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    means = out.mean_abs_per_raw_feature
    assert set(means.keys()) == set(out.raw_feature_names)
    assert all(v >= 0 for v in means.values())


def test_attribute_lr_accepts_pipeline_directly(fitted_lr_and_test):
    """Convenience: passing the raw fitted Pipeline works too."""
    model, X_test = fitted_lr_and_test
    out_from_pipeline = attribute_lr(lr_model=model.pipeline_, X_test=X_test)
    out_from_wrapper = attribute_lr(lr_model=model, X_test=X_test)
    np.testing.assert_array_equal(out_from_pipeline.shap_per_raw, out_from_wrapper.shap_per_raw)


def test_attribute_lr_rejects_unfit_lrmodel():
    with pytest.raises(RuntimeError, match="must be fit"):
        attribute_lr(lr_model=LRModel(), X_test=pd.DataFrame())


def test_attribute_lr_rejects_wrong_type():
    with pytest.raises(TypeError, match="LRModel"):
        attribute_lr(lr_model="not a model", X_test=pd.DataFrame())


def test_per_basis_includes_rcs_columns(fitted_lr_and_test):
    """RCS expansion adds spline-basis columns for the continuous features.

    The LR ColumnTransformer's nested Pipeline strips the raw column name
    metadata when SimpleImputer returns a numpy array, so the RCS columns
    appear under sklearn's positional fallback names ``x0..x4`` rather
    than ``Age_rcs1``. Either pattern counts as a valid spline basis.
    """
    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    assert any("_rcs" in name for name in out.basis_feature_names)


def test_rcs_column_attribution_sums_into_age(fitted_lr_and_test):
    """All Age-derived basis columns must sum into the per-raw 'Age' column.

    The LR pipeline produces ``x0`` + ``x0_rcs1`` + ``x0_rcs2`` for ``Age``
    (positional fallback because SimpleImputer drops the column name).
    The sum-back property still holds.
    """
    from cardiorisk.data.preprocess import NUMERIC_COLUMNS

    model, X_test = fitted_lr_and_test
    out = attribute_lr(lr_model=model, X_test=X_test)
    age_idx = NUMERIC_COLUMNS.index("Age")
    age_basis_prefix = f"x{age_idx}"
    age_basis_indices = [
        i
        for i, n in enumerate(out.basis_feature_names)
        if n == "Age"
        or n.startswith("Age_rcs")
        or n == age_basis_prefix
        or n.startswith(f"{age_basis_prefix}_rcs")
    ]
    assert len(age_basis_indices) >= 2  # linear term + at least one RCS
    age_raw_idx = out.raw_feature_names.index("Age")
    np.testing.assert_allclose(
        out.shap_per_basis[:, age_basis_indices].sum(axis=1),
        out.shap_per_raw[:, age_raw_idx],
        atol=1e-12,
    )
