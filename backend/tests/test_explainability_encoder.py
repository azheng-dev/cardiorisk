"""Smoke tests for :mod:`cardiorisk.explainability.encoder`."""

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
from cardiorisk.explainability.encoder import (
    INDICATOR_COLUMNS,
    EncodedFeatureSpace,
    FeatureGroup,
    fit_encoder,
)


@pytest.fixture
def cleaned_frame() -> pd.DataFrame:
    rows = generate_fixture(n=80, seed=42)
    df = pd.DataFrame(rows)
    df["source"] = "test"
    return clean_for_modelling(df)


def test_fit_encoder_returns_encoded_feature_space(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    assert isinstance(enc, EncodedFeatureSpace)
    assert enc.n_columns > 0
    assert enc.n_groups == (
        len(CATEGORICAL_COLUMNS)
        + len(NUMERIC_COLUMNS)
        + len(BINARY_NUMERIC_COLUMNS)
        + len(INDICATOR_COLUMNS)
    )


def test_encoded_columns_total_matches_groups(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    total = sum(len(g.column_indices) for g in enc.feature_groups)
    assert total == enc.n_columns


def test_categorical_groups_have_multiple_columns(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    cat_groups = [g for g in enc.feature_groups if g.is_categorical]
    assert len(cat_groups) == len(CATEGORICAL_COLUMNS)
    # Sex always has at least M/F; ChestPainType has 4 levels.
    assert all(len(g.column_indices) >= 2 for g in cat_groups)


def test_numeric_and_indicator_groups_have_one_column(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    non_cat = [g for g in enc.feature_groups if not g.is_categorical]
    assert all(len(g.column_indices) == 1 for g in non_cat)


def test_encode_shape(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    out = enc.encode(cleaned_frame.head(7))
    assert out.shape == (7, enc.n_columns)
    assert out.dtype == np.float64


def test_encode_decode_roundtrip_categoricals(cleaned_frame: pd.DataFrame) -> None:
    """Encoding then decoding must recover the categorical labels exactly."""
    enc = fit_encoder(cleaned_frame)
    arr = enc.encode(cleaned_frame)
    df_back = enc.decode(arr)
    for cat in CATEGORICAL_COLUMNS:
        # Compare element-wise; index alignment doesn't matter -- both sides
        # are the row-i value of column ``cat``.
        np.testing.assert_array_equal(
            np.asarray(df_back[cat]),
            np.asarray(cleaned_frame[cat]),
        )


def test_encode_decode_roundtrip_numerics(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    arr = enc.encode(cleaned_frame)
    df_back = enc.decode(arr)
    for num in NUMERIC_COLUMNS:
        # Numerics may carry NaN (Cholesterol post-cleaning); use
        # equal_nan-aware comparison.
        np.testing.assert_array_equal(
            np.asarray(df_back[num]),
            np.asarray(cleaned_frame[num].astype(np.float64)),
        )


def test_aggregate_shap_sums_categorical_blocks(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    n_rows = 5
    shap_per_col = np.ones((n_rows, enc.n_columns), dtype=np.float64)
    agg = enc.aggregate_shap(shap_per_col)
    assert agg.shape == (n_rows, enc.n_groups)
    # Sum across each group must equal the number of OHE columns in that
    # group times 1.0 (the SHAP-per-column we set above).
    for j, g in enumerate(enc.feature_groups):
        assert np.allclose(agg[:, j], len(g.column_indices))


def test_aggregate_shap_preserves_rowwise_sum(cleaned_frame: pd.DataFrame) -> None:
    """Sum over raw features must equal sum over encoded columns row-wise."""
    rng = np.random.default_rng(0)
    enc = fit_encoder(cleaned_frame)
    shap_per_col = rng.normal(size=(10, enc.n_columns))
    agg = enc.aggregate_shap(shap_per_col)
    np.testing.assert_allclose(agg.sum(axis=1), shap_per_col.sum(axis=1), atol=1e-12)


def test_raw_feature_names_match_groups(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    assert enc.raw_feature_names == tuple(g.raw_name for g in enc.feature_groups)


def test_unfit_decode_rejects_wrong_width(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    with pytest.raises(ValueError, match="columns"):
        enc.decode(np.zeros((4, enc.n_columns - 1)))


def test_aggregate_shap_rejects_wrong_shape(cleaned_frame: pd.DataFrame) -> None:
    enc = fit_encoder(cleaned_frame)
    with pytest.raises(ValueError, match="expected shape"):
        enc.aggregate_shap(np.zeros((4, enc.n_columns - 1)))


def test_fit_encoder_rejects_missing_indicator_columns() -> None:
    """The encoder requires the cleaned schema, including the indicator suffixes."""
    rows = generate_fixture(n=20, seed=0)
    df = pd.DataFrame(rows)  # not cleaned -> no _was_missing columns
    with pytest.raises(KeyError, match="post-cleaning"):
        fit_encoder(df)


def test_fit_encoder_rejects_uncleaned_categorical_nan(cleaned_frame: pd.DataFrame) -> None:
    """Encoder requires the explicit ``Missing`` sentinel, not raw NaN."""
    df = cleaned_frame.copy()
    df.loc[0, "ChestPainType"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        fit_encoder(df)


def test_feature_group_fields() -> None:
    """``FeatureGroup`` is a frozen dataclass with the public surface we rely on."""
    g = FeatureGroup(raw_name="Age", column_indices=(0,), is_categorical=False)
    assert g.raw_name == "Age"
    assert g.column_indices == (0,)
    assert g.is_categorical is False
