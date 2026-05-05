"""Tests for cardiorisk.data.preprocess.

Cleaning steps are pure functions on rows, so the tests focus on:

- Behavioural correctness of each step (chol-zero, indicators, categorical NaN).
- Idempotency (re-running is a no-op or has the same effect).
- Schema invariants the downstream pipelines depend on.
- Numeric dtype coercion (Int64 -> float64) so sklearn ColumnTransformer is
  happy under passthrough.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.data.combine import HFP_COLUMNS
from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    MISSING_CATEGORY_LABEL,
    MISSINGNESS_INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
    add_missingness_indicators,
    clean_cholesterol_zero_to_nan,
    clean_for_modelling,
    coerce_numeric_to_float64,
    replace_categorical_missing,
)


def _hfp_row(**overrides: object) -> dict[str, object]:
    """Build a single HFP-shaped row with sensible defaults; override fields by name."""
    base: dict[str, object] = {
        "Age": 50,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 120.0,
        "Cholesterol": 200.0,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150.0,
        "ExerciseAngina": "N",
        "Oldpeak": 1.0,
        "ST_Slope": "Up",
        "HeartDisease": 0,
    }
    base.update(overrides)
    return base


def _hfp_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build an HFP-schema frame with the canonical Int64 + float dtypes."""
    df = pd.DataFrame(rows)
    df["Age"] = df["Age"].astype("Int64")
    df["FastingBS"] = df["FastingBS"].astype("Int64")
    df["HeartDisease"] = df["HeartDisease"].astype("Int64")
    return df


# ---------------------------------------------------------------- chol cleaning


def test_clean_cholesterol_zero_to_nan_replaces_only_zeros() -> None:
    df = _hfp_frame(
        [
            _hfp_row(Cholesterol=0.0),
            _hfp_row(Cholesterol=200.0),
            _hfp_row(Cholesterol=np.nan),
        ]
    )
    out = clean_cholesterol_zero_to_nan(df)
    assert pd.isna(out.loc[0, "Cholesterol"])
    assert out.loc[1, "Cholesterol"] == 200.0  # type: ignore[unreachable]
    assert pd.isna(out.loc[2, "Cholesterol"])


def test_clean_cholesterol_zero_does_not_mutate_input() -> None:
    df = _hfp_frame([_hfp_row(Cholesterol=0.0)])
    _ = clean_cholesterol_zero_to_nan(df)
    assert df.loc[0, "Cholesterol"] == 0.0


def test_clean_cholesterol_zero_idempotent() -> None:
    df = _hfp_frame([_hfp_row(Cholesterol=0.0), _hfp_row(Cholesterol=200.0)])
    once = clean_cholesterol_zero_to_nan(df)
    twice = clean_cholesterol_zero_to_nan(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_cholesterol_zero_raises_on_missing_column() -> None:
    df = pd.DataFrame({"NotCholesterol": [1, 2]})
    with pytest.raises(KeyError, match="Cholesterol"):
        clean_cholesterol_zero_to_nan(df)


# ---------------------------------------------------------------- indicators


def test_add_missingness_indicators_adds_one_column_per_input() -> None:
    df = _hfp_frame(
        [
            _hfp_row(RestingBP=np.nan),
            _hfp_row(RestingBP=120.0),
        ]
    )
    out = add_missingness_indicators(df)
    for col in MISSINGNESS_INDICATOR_COLUMNS:
        assert f"{col}_was_missing" in out.columns


def test_add_missingness_indicators_values_are_zero_one() -> None:
    df = _hfp_frame(
        [
            _hfp_row(RestingBP=np.nan, MaxHR=140.0),
            _hfp_row(RestingBP=120.0, MaxHR=np.nan),
        ]
    )
    out = add_missingness_indicators(df)
    assert list(out["RestingBP_was_missing"]) == [1, 0]
    assert list(out["MaxHR_was_missing"]) == [0, 1]
    assert out["RestingBP_was_missing"].dtype == np.int8


def test_add_missingness_indicators_idempotent() -> None:
    df = _hfp_frame([_hfp_row(RestingBP=np.nan), _hfp_row(RestingBP=120.0)])
    once = add_missingness_indicators(df)
    twice = add_missingness_indicators(once)
    pd.testing.assert_frame_equal(once, twice)


def test_add_missingness_indicators_preserves_existing_indicator() -> None:
    """Re-running after categorical NaN -> 'Missing' must NOT flip 1 to 0.

    This is a regression test for a real bug: the original implementation
    overwrote the indicator unconditionally, so the second call to
    `clean_for_modelling` would see ST_Slope == 'Missing' (no longer NaN)
    and silently zero out the indicator that the first call had set.
    """
    df = _hfp_frame([_hfp_row(ST_Slope=np.nan)])
    after_first_indicator = add_missingness_indicators(df)
    after_categorical = replace_categorical_missing(after_first_indicator)
    final = add_missingness_indicators(after_categorical)
    assert final.loc[0, "ST_Slope_was_missing"] == 1


# ---------------------------------------------------------------- categorical NaN


def test_replace_categorical_missing_uses_explicit_label() -> None:
    df = _hfp_frame(
        [
            _hfp_row(ST_Slope=np.nan),
            _hfp_row(ST_Slope="Up"),
        ]
    )
    out = replace_categorical_missing(df)
    assert out.loc[0, "ST_Slope"] == MISSING_CATEGORY_LABEL
    assert out.loc[1, "ST_Slope"] == "Up"


def test_replace_categorical_missing_handles_all_categorical_columns() -> None:
    df = _hfp_frame([_hfp_row(**dict.fromkeys(CATEGORICAL_COLUMNS, np.nan))])
    out = replace_categorical_missing(df)
    for col in CATEGORICAL_COLUMNS:
        assert out.loc[0, col] == MISSING_CATEGORY_LABEL


def test_replace_categorical_missing_idempotent() -> None:
    df = _hfp_frame([_hfp_row(ST_Slope=np.nan)])
    once = replace_categorical_missing(df)
    twice = replace_categorical_missing(once)
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------------------- numeric coercion


def test_coerce_numeric_to_float64_converts_int64_columns() -> None:
    df = _hfp_frame([_hfp_row(Age=50, FastingBS=1)])
    assert df["Age"].dtype == "Int64"
    assert df["FastingBS"].dtype == "Int64"
    out = coerce_numeric_to_float64(df)
    for col in (*NUMERIC_COLUMNS, *BINARY_NUMERIC_COLUMNS):
        assert out[col].dtype == np.float64


def test_coerce_numeric_to_float64_preserves_target() -> None:
    df = _hfp_frame([_hfp_row(HeartDisease=1)])
    out = coerce_numeric_to_float64(df)
    assert out["HeartDisease"].dtype == "Int64"


def test_coerce_numeric_to_float64_preserves_nan_in_floats() -> None:
    df = _hfp_frame([_hfp_row(RestingBP=np.nan, Age=pd.NA, FastingBS=pd.NA)])
    out = coerce_numeric_to_float64(df)
    assert pd.isna(out.loc[0, "RestingBP"])
    assert pd.isna(out.loc[0, "Age"])  # type: ignore[unreachable]
    assert pd.isna(out.loc[0, "FastingBS"])


# ---------------------------------------------------------------- clean_for_modelling


def test_clean_for_modelling_adds_indicator_columns() -> None:
    df = _hfp_frame([_hfp_row()])
    out = clean_for_modelling(df)
    expected_extra = {f"{c}_was_missing" for c in MISSINGNESS_INDICATOR_COLUMNS}
    assert expected_extra.issubset(out.columns)


def test_clean_for_modelling_chains_chol_cleaning_with_indicators() -> None:
    df = _hfp_frame([_hfp_row(Cholesterol=0.0)])
    out = clean_for_modelling(df)
    assert pd.isna(out.loc[0, "Cholesterol"])


def test_clean_for_modelling_replaces_categorical_nan() -> None:
    df = _hfp_frame([_hfp_row(ST_Slope=np.nan)])
    out = clean_for_modelling(df)
    assert out.loc[0, "ST_Slope"] == MISSING_CATEGORY_LABEL


def test_clean_for_modelling_coerces_numeric_columns_to_float64() -> None:
    df = _hfp_frame([_hfp_row()])
    out = clean_for_modelling(df)
    for col in (*NUMERIC_COLUMNS, *BINARY_NUMERIC_COLUMNS):
        assert out[col].dtype == np.float64


def test_clean_for_modelling_idempotent() -> None:
    df = _hfp_frame(
        [
            _hfp_row(Cholesterol=0.0, ST_Slope=np.nan, RestingBP=np.nan),
            _hfp_row(Cholesterol=200.0, ST_Slope="Up", RestingBP=120.0),
        ]
    )
    once = clean_for_modelling(df)
    twice = clean_for_modelling(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_for_modelling_does_not_mutate_input() -> None:
    df = _hfp_frame([_hfp_row(Cholesterol=0.0, ST_Slope=np.nan)])
    df_before = df.copy()
    _ = clean_for_modelling(df)
    pd.testing.assert_frame_equal(df, df_before)


def test_clean_for_modelling_preserves_source_column_when_present() -> None:
    df = _hfp_frame([_hfp_row()])
    df["source"] = "Cleveland"
    out = clean_for_modelling(df)
    assert "source" in out.columns
    assert out.loc[0, "source"] == "Cleveland"


def test_clean_for_modelling_raises_on_missing_hfp_column() -> None:
    df = pd.DataFrame({c: [0] for c in HFP_COLUMNS if c != "Cholesterol"})
    with pytest.raises(KeyError, match="Cholesterol"):
        clean_for_modelling(df)
