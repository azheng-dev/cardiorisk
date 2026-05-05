"""Tests for cardiorisk.features.spline.

Covers knot placement (Harrell quantiles), output shape (k-1 columns per
input), determinism, NaN-handling expectations (RCS expects pre-imputed
input; we verify it raises clearly when fit on all-NaN), and the
linear-beyond-boundary property that distinguishes RCS from a generic
B-spline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.features.spline import (
    DEFAULT_N_KNOTS,
    HARRELL_QUANTILES,
    RestrictedCubicSpline,
)


def _linear_x() -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0.0, 100.0, 200)})


# ---------------------------------------------------------------- knot placement


@pytest.mark.parametrize("k", sorted(HARRELL_QUANTILES))
def test_fit_places_knots_at_harrell_quantiles(k: int) -> None:
    df = _linear_x()
    rcs = RestrictedCubicSpline(n_knots=k).fit(df)
    expected = np.quantile(df["x"].to_numpy(), HARRELL_QUANTILES[k])
    np.testing.assert_allclose(rcs.knots_[0], expected, rtol=1e-12)


def test_fit_uses_only_non_nan_rows_for_quantiles() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan]})
    rcs = RestrictedCubicSpline(n_knots=3).fit(df)
    expected = np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], HARRELL_QUANTILES[3])
    np.testing.assert_allclose(rcs.knots_[0], expected, rtol=1e-12)


def test_fit_raises_on_all_nan_feature() -> None:
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="non-NaN rows"):
        RestrictedCubicSpline(n_knots=3).fit(df)


def test_fit_rejects_unsupported_n_knots() -> None:
    df = _linear_x()
    with pytest.raises(ValueError, match="n_knots must be one of"):
        RestrictedCubicSpline(n_knots=2).fit(df)
    with pytest.raises(ValueError, match="n_knots must be one of"):
        RestrictedCubicSpline(n_knots=10).fit(df)


def test_fit_rejects_degenerate_quantiles() -> None:
    df = pd.DataFrame({"x": [1.0] * 10 + [2.0] * 2})  # almost constant
    with pytest.raises(ValueError, match="degenerate knot"):
        RestrictedCubicSpline(n_knots=4).fit(df)


# ---------------------------------------------------------------- output shape


@pytest.mark.parametrize("k", sorted(HARRELL_QUANTILES))
def test_transform_emits_k_minus_one_columns_per_input(k: int) -> None:
    df = _linear_x()
    rcs = RestrictedCubicSpline(n_knots=k).fit(df)
    out = rcs.transform(df)
    assert out.shape == (len(df), k - 1)


def test_transform_multi_feature_shape_is_sum_of_per_feature() -> None:
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 10, 100),
            "b": np.linspace(50, 200, 100),
        }
    )
    rcs = RestrictedCubicSpline(n_knots=4).fit(df)
    out = rcs.transform(df)
    assert out.shape == (100, (4 - 1) * 2)


# ---------------------------------------------------------------- behaviour


def test_first_column_is_the_linear_term() -> None:
    df = _linear_x()
    rcs = RestrictedCubicSpline(n_knots=4).fit(df)
    out = rcs.transform(df)
    np.testing.assert_allclose(out[:, 0], df["x"].to_numpy())


def test_basis_columns_are_zero_below_first_inner_knot() -> None:
    """The cubic-piece columns activate at knot t_{j-1}, so for any x < t_1
    they must all be exactly zero (linear extrapolation property)."""
    df = pd.DataFrame({"x": np.linspace(0.0, 100.0, 200)})
    rcs = RestrictedCubicSpline(n_knots=5).fit(df)
    knots = rcs.knots_[0]
    test_x = pd.DataFrame({"x": np.array([knots[0] - 1.0, knots[0] - 50.0])})
    out = rcs.transform(test_x)
    # Column 0 is the linear term (non-zero); columns 1..k-2 are cubic-pieces.
    np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-12)


def test_transform_is_deterministic_under_repeated_calls() -> None:
    df = _linear_x()
    rcs = RestrictedCubicSpline(n_knots=4).fit(df)
    out_a = rcs.transform(df)
    out_b = rcs.transform(df)
    np.testing.assert_array_equal(out_a, out_b)


def test_two_fits_on_same_data_produce_identical_knots() -> None:
    df = _linear_x()
    rcs_a = RestrictedCubicSpline(n_knots=4).fit(df)
    rcs_b = RestrictedCubicSpline(n_knots=4).fit(df)
    np.testing.assert_array_equal(rcs_a.knots_[0], rcs_b.knots_[0])


# ---------------------------------------------------------------- API contracts


def test_transform_before_fit_raises() -> None:
    df = _linear_x()
    with pytest.raises(RuntimeError, match="must be fit"):
        RestrictedCubicSpline(n_knots=DEFAULT_N_KNOTS).transform(df)


def test_transform_with_wrong_feature_count_raises() -> None:
    rcs = RestrictedCubicSpline(n_knots=4).fit(_linear_x())
    df_wrong = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="features"):
        rcs.transform(df_wrong)


def test_get_feature_names_out_emits_named_basis_columns() -> None:
    df = pd.DataFrame({"Age": np.linspace(20.0, 80.0, 100)})
    rcs = RestrictedCubicSpline(n_knots=4).fit(df)
    names = rcs.get_feature_names_out().tolist()
    assert names[0] == "Age"
    assert all(n.startswith("Age") for n in names)
    assert len(names) == 4 - 1
