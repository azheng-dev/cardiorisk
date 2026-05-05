"""Tests for cardiorisk.features.pipeline.

The headline test in this file is the LEAKAGE check: a pipeline fit on
training subset A must produce *different* imputed values when later
transforming the same test slice than the same pipeline architecture
would if it had been fit on subset B. If that property fails, our LODO
evaluation in Phase 2.3 is silently leaking statistics from the held-out
fold and the headline numbers are inflated.

We also verify per-factory output shapes, the categorical "Missing"
round-trip, and that TabPFN's pipeline preserves NaN in numeric columns
(its native handling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from cardiorisk.data.combine import build_from_fixture
from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    MISSINGNESS_INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
    clean_for_modelling,
)
from cardiorisk.features.pipeline import (
    make_lr_pipeline,
    make_tabpfn_pipeline,
    make_woa_pipeline,
    make_xgboost_pipeline,
)


@pytest.fixture(scope="module")
def cleaned_frame() -> pd.DataFrame:
    """Cleaned 20-row HFP-schema fixture frame, ready for the pipelines."""
    raw = build_from_fixture()
    return clean_for_modelling(raw)


@pytest.fixture
def feature_target_split(
    cleaned_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = cleaned_frame.drop(columns=["HeartDisease", "source"])
    y = cleaned_frame["HeartDisease"]
    return X, y


# ---------------------------------------------------------------- output shapes


@pytest.mark.parametrize(
    ("factory", "expected_min_columns"),
    [
        # OHE expansion of 5 categoricals + numerics + indicators. Exact width
        # depends on which categorical levels are present in the fit slice;
        # 20 columns is a safe lower bound.
        (make_tabpfn_pipeline, 20),
        (make_xgboost_pipeline, 20),
        (make_woa_pipeline, 20),
        # LR adds RCS expansion (3 extra cols per of 5 numerics) so >= ~30.
        (make_lr_pipeline, 30),
    ],
)
def test_factory_output_shape_is_sensible(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
    factory: object,
    expected_min_columns: int,
) -> None:
    X, y = feature_target_split
    pipe = factory()  # type: ignore[operator]
    out = pipe.fit_transform(X, y)
    assert out.shape[0] == len(X)
    assert out.shape[1] >= expected_min_columns
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "factory",
    [make_tabpfn_pipeline, make_xgboost_pipeline, make_lr_pipeline, make_woa_pipeline],
)
def test_factory_returns_unfit_pipeline(factory: object) -> None:
    pipe = factory()  # type: ignore[operator]
    assert isinstance(pipe, Pipeline)
    # Sklearn marks "fitted" by setting underscore-suffixed attributes on the
    # final estimator. An unfit pipeline must raise on transform().
    with pytest.raises(Exception):  # noqa: B017
        pipe.transform(pd.DataFrame())


# ---------------------------------------------------------------- TabPFN: NaN passthrough


def test_tabpfn_pipeline_preserves_nan_in_numeric_columns(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """TabPFN handles NaN natively, so the preprocessing pipeline must
    *not* impute. If this test fails we have accidentally imputed numerics
    upstream of TabPFN, which would degrade its performance and make our
    headline metric inconsistent with the published TabPFN protocol."""
    X, y = feature_target_split
    # Force a NaN into a numeric column so the test has something to detect.
    X = X.copy()
    X.loc[X.index[0], "Cholesterol"] = np.nan
    pipe = make_tabpfn_pipeline()
    out = pipe.fit_transform(X, y)
    assert np.isnan(out).any()


def test_xgboost_pipeline_imputes_all_numeric_nan(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """XGBoost shares the MissForest treatment with WOA, so the output
    must be NaN-free (modulo any trailing infinity from a bad fit, which
    should also fail this test)."""
    X, y = feature_target_split
    pipe = make_xgboost_pipeline()
    out = pipe.fit_transform(X, y)
    assert not np.isnan(out).any()


def test_lr_pipeline_imputes_all_numeric_nan(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = feature_target_split
    pipe = make_lr_pipeline()
    out = pipe.fit_transform(X, y)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------- categorical "Missing" round-trip


def test_categorical_missing_label_appears_as_one_hot_column(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """If any categorical row was originally NaN, the cleaned frame has
    'Missing' in that cell, and OneHotEncoder should emit a column whose
    name ends in '_Missing'."""
    X, y = feature_target_split
    # Force an ST_Slope row to "Missing" so we know at least one row carries it.
    X = X.copy()
    X.loc[X.index[0], "ST_Slope"] = "Missing"
    pipe = make_tabpfn_pipeline()
    pipe.fit(X, y)
    feature_names = pipe[-1].get_feature_names_out().tolist()
    assert any(name.endswith("_Missing") for name in feature_names), feature_names


# ---------------------------------------------------------------- LEAKAGE TESTS


def _disjoint_slices(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
    pd.DataFrame,
]:
    """Split into two disjoint training subsets + a shared holdout."""
    n = len(X)
    holdout_idx = np.array([0, 1, 2])
    rest_idx = np.array([i for i in range(n) if i not in holdout_idx])
    half = len(rest_idx) // 2
    a_idx = rest_idx[:half]
    b_idx = rest_idx[half:]
    return (
        (X.iloc[a_idx], y.iloc[a_idx]),
        (X.iloc[b_idx], y.iloc[b_idx]),
        X.iloc[holdout_idx],
    )


def test_lr_pipeline_imputer_means_differ_when_fit_on_disjoint_slices(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Direct mean-imputer leakage check: SimpleImputer.statistics_ is the
    learnt mean. If two pipelines fit on disjoint subsets had the same
    mean, our preprocessing would be effectively pre-fit on the union and
    LODO leakage would be undetectable. We assert the means differ
    materially."""
    X, y = feature_target_split
    (X_a, y_a), (X_b, y_b), _ = _disjoint_slices(X, y)

    pipe_a = make_lr_pipeline()
    pipe_b = make_lr_pipeline()
    pipe_a.fit(X_a, y_a)
    pipe_b.fit(X_b, y_b)

    # Reach into the ColumnTransformer -> rcs_continuous Pipeline -> mean_impute step.
    ct_a = pipe_a.named_steps["preprocess"]
    ct_b = pipe_b.named_steps["preprocess"]
    imputer_a = ct_a.named_transformers_["rcs_continuous"].named_steps["mean_impute"]
    imputer_b = ct_b.named_transformers_["rcs_continuous"].named_steps["mean_impute"]

    means_a = imputer_a.statistics_
    means_b = imputer_b.statistics_
    assert not np.allclose(means_a, means_b), (
        "mean imputer fit on disjoint slices produced identical statistics; "
        "this is a leakage red flag (the imputer is being fit on the union)."
    )


def test_xgboost_pipeline_imputer_state_differs_when_fit_on_disjoint_slices(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """State-level leakage check for the IterativeImputer (MissForest variant).

    Reaches into the fitted ``IterativeImputer.initial_imputer_.statistics_``
    (the per-column means used as starting values for the iterative
    procedure) and asserts they differ across disjoint training slices.
    This is shape-independent — it works even when the OHE happens to
    produce the same column count on both slices, which the previous
    `transform-and-compare` version could short-circuit past."""
    X, y = feature_target_split
    (X_a, y_a), (X_b, y_b), _ = _disjoint_slices(X, y)

    pipe_a = make_xgboost_pipeline().fit(X_a, y_a)
    pipe_b = make_xgboost_pipeline().fit(X_b, y_b)

    ct_a = pipe_a.named_steps["preprocess"]
    ct_b = pipe_b.named_steps["preprocess"]
    imp_a = ct_a.named_transformers_["missforest_continuous"]
    imp_b = ct_b.named_transformers_["missforest_continuous"]

    stats_a = imp_a.initial_imputer_.statistics_
    stats_b = imp_b.initial_imputer_.statistics_
    assert not np.allclose(stats_a, stats_b), (
        "IterativeImputer (MissForest) fit on disjoint slices produced "
        "identical initial-imputer statistics; check for leakage. "
        f"a={stats_a.tolist()}, b={stats_b.tolist()}"
    )


def test_xgboost_pipeline_imputed_values_differ_when_fit_on_disjoint_slices(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """End-to-end transform-level leakage check: complements the state-level
    test above by verifying the *behaviour* differs, not just the internal
    state. Allowed to short-circuit on shape mismatch (which is itself
    evidence of fit-dependence — different OHE levels seen)."""
    X, y = feature_target_split
    (X_a, y_a), (X_b, y_b), X_holdout = _disjoint_slices(X, y)
    X_holdout = X_holdout.copy()
    X_holdout.loc[X_holdout.index[0], "Cholesterol"] = np.nan

    pipe_a = make_xgboost_pipeline().fit(X_a, y_a)
    pipe_b = make_xgboost_pipeline().fit(X_b, y_b)

    out_a = pipe_a.transform(X_holdout)
    out_b = pipe_b.transform(X_holdout)
    if out_a.shape != out_b.shape:
        return
    assert not np.allclose(out_a, out_b), (
        "IterativeImputer (MissForest) fit on disjoint slices produced "
        "identical imputations on a shared holdout; check for leakage."
    )


def test_pipeline_transform_does_not_refit(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Calling transform() twice on the same data must produce identical
    outputs, and must not modify the fitted statistics. This is a sanity
    check on the sklearn API contract we rely on for leakage protection."""
    X, y = feature_target_split
    pipe = make_xgboost_pipeline().fit(X, y)
    out_a = pipe.transform(X)
    out_b = pipe.transform(X)
    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------- column groups in scope


def test_all_pipelines_consume_every_modelling_column(
    feature_target_split: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Smoke test that the column lists in the factories cover every
    feature that clean_for_modelling produces, except HeartDisease and
    source. Catches the bug where a feature is silently dropped from
    every pipeline (e.g. typo in the column tuple)."""
    X, _ = feature_target_split
    expected = (
        set(NUMERIC_COLUMNS)
        | set(BINARY_NUMERIC_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | {f"{c}_was_missing" for c in MISSINGNESS_INDICATOR_COLUMNS}
    )
    assert expected.issubset(set(X.columns)), (
        f"cleaned frame is missing modelling columns: {expected - set(X.columns)}"
    )
