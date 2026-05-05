"""Tests for cardiorisk.features.cv.

LODO-CV is the headline evaluation protocol per ADR-006. The splitter
correctness directly affects every downstream metric, so the tests here
are deliberately exhaustive: fold count, held-out source identity,
disjointness, reproducibility, and within-fold sub-split arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.features.cv import (
    SEED,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    LodoFold,
    iter_lodo_folds,
    iter_random_kfold,
    within_fold_split,
)

UCI_SOURCES = ("Cleveland", "Hungarian", "Switzerland", "LongBeachVA")


def _toy_combined(rows_per_source: int = 50, seed: int = 1) -> pd.DataFrame:
    """Build a toy LODO frame with the four UCI sources and a binary target."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for src in UCI_SOURCES:
        for _ in range(rows_per_source):
            rows.append({SOURCE_COLUMN: src, TARGET_COLUMN: int(rng.integers(0, 2))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LODO


def test_iter_lodo_folds_yields_one_fold_per_source() -> None:
    df = _toy_combined()
    folds = list(iter_lodo_folds(df))
    assert len(folds) == len(UCI_SOURCES)
    held = sorted(f.held_out_source for f in folds)
    assert held == sorted(UCI_SOURCES)


def test_iter_lodo_folds_held_out_indices_match_source_column() -> None:
    df = _toy_combined()
    for fold in iter_lodo_folds(df):
        test_sources = df.iloc[fold.test_idx][SOURCE_COLUMN].unique().tolist()
        assert test_sources == [fold.held_out_source]
        train_sources = set(df.iloc[fold.train_idx][SOURCE_COLUMN].unique())
        assert fold.held_out_source not in train_sources


def test_iter_lodo_folds_train_test_indices_are_disjoint_and_complete() -> None:
    df = _toy_combined()
    for fold in iter_lodo_folds(df):
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0
        assert len(fold.train_idx) + len(fold.test_idx) == len(df)


def test_iter_lodo_folds_iteration_order_is_deterministic() -> None:
    df = _toy_combined()
    first = [f.held_out_source for f in iter_lodo_folds(df)]
    second = [f.held_out_source for f in iter_lodo_folds(df)]
    assert first == second
    assert first == sorted(first)


def test_iter_lodo_folds_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"only_source": ["A", "B"]})
    with pytest.raises(KeyError):
        list(iter_lodo_folds(df))


def test_iter_lodo_folds_error_names_both_required_columns() -> None:
    """Error message should list both required columns so the caller knows
    what to add. Regression test for the looser per-splitter validation."""
    df = pd.DataFrame({"HeartDisease": [0, 1]})
    with pytest.raises(KeyError, match="source"):
        list(iter_lodo_folds(df))


def test_lodo_fold_validates_disjoint_in_constructor() -> None:
    with pytest.raises(ValueError, match="overlapping train/test"):
        LodoFold(
            held_out_source="X",
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([2, 3, 4]),
        )


# ---------------------------------------------------------------- within-fold


def test_within_fold_split_sums_to_train_idx_size() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    sub = within_fold_split(fold.train_idx, df[TARGET_COLUMN])
    total = len(sub.train_idx) + len(sub.val_idx) + len(sub.calib_idx)
    assert total == len(fold.train_idx)


def test_within_fold_split_partitions_are_pairwise_disjoint() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    sub = within_fold_split(fold.train_idx, df[TARGET_COLUMN])
    assert np.intersect1d(sub.train_idx, sub.val_idx).size == 0
    assert np.intersect1d(sub.train_idx, sub.calib_idx).size == 0
    assert np.intersect1d(sub.val_idx, sub.calib_idx).size == 0


def test_within_fold_split_is_subset_of_train_idx() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    sub = within_fold_split(fold.train_idx, df[TARGET_COLUMN])
    union = np.concatenate([sub.train_idx, sub.val_idx, sub.calib_idx])
    assert set(union.tolist()).issubset(set(fold.train_idx.tolist()))


def test_within_fold_split_default_proportions_are_eighty_ten_ten() -> None:
    df = _toy_combined(rows_per_source=200)  # 600 train rows -> easy to verify ratios
    fold = next(iter_lodo_folds(df))
    sub = within_fold_split(fold.train_idx, df[TARGET_COLUMN])
    n_train = len(fold.train_idx)
    assert abs(len(sub.train_idx) / n_train - 0.80) < 0.01
    assert abs(len(sub.val_idx) / n_train - 0.10) < 0.01
    assert abs(len(sub.calib_idx) / n_train - 0.10) < 0.01


def test_within_fold_split_is_deterministic_under_same_seed() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    a = within_fold_split(fold.train_idx, df[TARGET_COLUMN], seed=SEED)
    b = within_fold_split(fold.train_idx, df[TARGET_COLUMN], seed=SEED)
    np.testing.assert_array_equal(a.train_idx, b.train_idx)
    np.testing.assert_array_equal(a.val_idx, b.val_idx)
    np.testing.assert_array_equal(a.calib_idx, b.calib_idx)


def test_within_fold_split_differs_under_different_seed() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    a = within_fold_split(fold.train_idx, df[TARGET_COLUMN], seed=1)
    b = within_fold_split(fold.train_idx, df[TARGET_COLUMN], seed=2)
    # Splits should differ on at least one partition under different seeds.
    assert not np.array_equal(a.train_idx, b.train_idx)


def test_within_fold_split_rejects_invalid_fractions() -> None:
    df = _toy_combined()
    fold = next(iter_lodo_folds(df))
    with pytest.raises(ValueError, match="must be > 0"):
        within_fold_split(fold.train_idx, df[TARGET_COLUMN], val_frac=0.0)
    with pytest.raises(ValueError, match=r"must be < 1\.0"):
        within_fold_split(fold.train_idx, df[TARGET_COLUMN], val_frac=0.6, calib_frac=0.5)


# ---------------------------------------------------------------- random K-fold


def test_iter_random_kfold_yields_n_splits_folds() -> None:
    df = _toy_combined()
    folds = list(iter_random_kfold(df, n_splits=5))
    assert len(folds) == 5


def test_iter_random_kfold_partitions_test_indices_disjointly() -> None:
    df = _toy_combined()
    seen: set[int] = set()
    for _, test_idx in iter_random_kfold(df, n_splits=5):
        idx_set = set(test_idx.tolist())
        assert seen.isdisjoint(idx_set)
        seen |= idx_set
    assert seen == set(range(len(df)))


def test_iter_random_kfold_is_reproducible_under_same_seed() -> None:
    df = _toy_combined()
    a_test = [t.tolist() for _, t in iter_random_kfold(df, seed=SEED)]
    b_test = [t.tolist() for _, t in iter_random_kfold(df, seed=SEED)]
    assert a_test == b_test


def test_iter_random_kfold_does_not_require_source_column() -> None:
    """Random K-fold deliberately ignores source identity; it must not
    fail on a frame that has the target but no source. Bug fix for the
    over-validation in the original `_validate_columns` helper."""
    df = pd.DataFrame({"HeartDisease": [0, 1] * 10})
    folds = list(iter_random_kfold(df, n_splits=2))
    assert len(folds) == 2


def test_iter_random_kfold_raises_on_missing_target() -> None:
    df = pd.DataFrame({"only_source": ["A", "B"] * 5})
    with pytest.raises(KeyError, match="HeartDisease"):
        list(iter_random_kfold(df, n_splits=2))
