"""Tests for :mod:`cardiorisk.explainability.cross_model_agreement`."""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.explainability.cross_model_agreement import (
    AgreementResult,
    aggregate_across_folds,
    compute_cross_model_agreement,
)


def test_diagonal_is_one() -> None:
    """A model is perfectly correlated with itself."""
    out = compute_cross_model_agreement(
        mean_abs_per_model={
            "tabicl": {"a": 1.0, "b": 2.0, "c": 3.0},
            "lr": {"a": 0.5, "b": 1.0, "c": 1.5},
        }
    )
    assert isinstance(out, AgreementResult)
    np.testing.assert_allclose(np.diag(out.spearman_matrix), 1.0)


def test_perfectly_concordant_is_one() -> None:
    """Two models with the same rank order get Spearman=1."""
    out = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0, "c": 3.0},
            "m2": {"a": 10.0, "b": 20.0, "c": 30.0},
        }
    )
    assert out.spearman_matrix[0, 1] == pytest.approx(1.0)
    assert out.spearman_matrix[1, 0] == pytest.approx(1.0)


def test_perfectly_discordant_is_minus_one() -> None:
    out = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0, "c": 3.0},
            "m2": {"a": 30.0, "b": 20.0, "c": 10.0},
        }
    )
    assert out.spearman_matrix[0, 1] == pytest.approx(-1.0)


def test_matrix_is_symmetric() -> None:
    out = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
            "m2": {"a": 4.0, "b": 1.0, "c": 3.0, "d": 2.0},
            "m3": {"a": 2.0, "b": 3.0, "c": 1.0, "d": 4.0},
        }
    )
    np.testing.assert_allclose(out.spearman_matrix, out.spearman_matrix.T)


def test_model_and_feature_names_preserved() -> None:
    out = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0},
            "m2": {"a": 2.0, "b": 1.0},
        }
    )
    assert out.model_names == ("m1", "m2")
    assert out.feature_names == ("a", "b")


def test_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_cross_model_agreement(mean_abs_per_model={})


def test_rejects_misaligned_features() -> None:
    with pytest.raises(ValueError, match="feature names mismatch"):
        compute_cross_model_agreement(
            mean_abs_per_model={
                "m1": {"a": 1.0, "b": 2.0},
                "m2": {"a": 1.0, "c": 2.0},
            }
        )


def test_aggregate_across_folds_means_correctly() -> None:
    base = {
        "m1": {"a": 1.0, "b": 2.0, "c": 3.0},
        "m2": {"a": 1.0, "b": 2.0, "c": 3.0},
    }
    other = {
        "m1": {"a": 1.0, "b": 2.0, "c": 3.0},
        "m2": {"a": 3.0, "b": 2.0, "c": 1.0},
    }
    fold1 = compute_cross_model_agreement(mean_abs_per_model=base)
    fold2 = compute_cross_model_agreement(mean_abs_per_model=other)
    agg = aggregate_across_folds(per_fold=[fold1, fold2])
    # Row 0 col 1: fold1 = +1, fold2 = -1, mean = 0
    assert agg.spearman_matrix[0, 1] == pytest.approx(0.0)


def test_aggregate_across_folds_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_across_folds(per_fold=[])


def test_aggregate_across_folds_rejects_axis_mismatch() -> None:
    fold1 = compute_cross_model_agreement(
        mean_abs_per_model={"m1": {"a": 1.0, "b": 2.0}, "m2": {"a": 2.0, "b": 1.0}}
    )
    fold2 = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0},
            "m3": {"a": 2.0, "b": 1.0},  # different model name
        }
    )
    with pytest.raises(ValueError, match="model_names"):
        aggregate_across_folds(per_fold=[fold1, fold2])
