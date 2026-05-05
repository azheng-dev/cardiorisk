"""Cross-validation splitters for Phase 2.3 onward.

Three splitters:

- :func:`iter_lodo_folds` — Leave-One-Domain-Out CV keyed on the ``source``
  column. The headline protocol per `04-revised-design.md` §3.5: train on
  every source except one, test on the held-out source, rotate. Yields
  exactly four folds for the four UCI subsets.

- :func:`within_fold_split` — Within each LODO fold's training rows, an
  80/10/10 deterministic split into ``train`` / ``val`` / ``calibration``.
  ``val`` is for hyperparameter tuning, ``calibration`` is for post-hoc
  isotonic calibration in Phase 2.3. Stratified on the target.

- :func:`iter_random_kfold` — Stratified random K-fold on the union, used
  *only* as the "random K-fold inflates numbers vs LODO" sanity-check
  comparison in `04-revised-design.md` §3.5. Never the headline.

All three use ``SEED = 20260505`` by default — the same constant pinned in
the design doc and the synthetic-fixture generator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split

#: Pinned RNG seed for any randomised split. Same constant as
#: ``backend/scripts/generate_fixture.py`` and `04-revised-design.md` §3.5.
SEED: Final[int] = 20260505

#: Headline target column name in the HFP schema.
TARGET_COLUMN: Final[str] = "HeartDisease"

#: LODO grouping column name (added by :mod:`cardiorisk.data.combine`).
SOURCE_COLUMN: Final[str] = "source"

#: Within-fold split fractions: 80% train, 10% val, 10% calibration.
WITHIN_FOLD_VAL_FRAC: Final[float] = 0.10
WITHIN_FOLD_CALIB_FRAC: Final[float] = 0.10


@dataclass(frozen=True)
class LodoFold:
    """One Leave-One-Domain-Out fold."""

    held_out_source: str
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        # Disjoint by construction in sklearn's LeaveOneGroupOut, but assert
        # cheaply because a silent bug here corrupts every downstream metric.
        if np.intersect1d(self.train_idx, self.test_idx).size:
            raise ValueError(
                f"LODO fold for held_out_source={self.held_out_source!r} produced "
                "overlapping train/test indices"
            )


@dataclass(frozen=True)
class WithinFoldSplit:
    """80/10/10 sub-split inside one LODO fold's training rows."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    calib_idx: np.ndarray

    def __post_init__(self) -> None:
        for a_name, a, b_name, b in (
            ("train", self.train_idx, "val", self.val_idx),
            ("train", self.train_idx, "calib", self.calib_idx),
            ("val", self.val_idx, "calib", self.calib_idx),
        ):
            if np.intersect1d(a, b).size:
                raise ValueError(f"within-fold {a_name} and {b_name} indices overlap")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in (TARGET_COLUMN, SOURCE_COLUMN) if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required column(s) {missing}; expected at "
            f"least {TARGET_COLUMN!r} and {SOURCE_COLUMN!r}"
        )


def iter_lodo_folds(df: pd.DataFrame) -> Iterator[LodoFold]:
    """Yield one :class:`LodoFold` per unique value of ``df['source']``.

    Folds are emitted in alphabetical order of held-out source so the
    iteration is deterministic across reruns. Indexes are positional
    (``df.iloc``-friendly) integer arrays, not pandas labels — that lets
    callers slice the underlying ``X``/``y`` numpy arrays directly without
    paying for ``df.loc`` lookups in a hot loop.
    """
    _validate_columns(df)
    groups = df[SOURCE_COLUMN].to_numpy()
    splitter = LeaveOneGroupOut()
    folds: list[LodoFold] = []
    for train_idx, test_idx in splitter.split(X=df, y=df[TARGET_COLUMN], groups=groups):
        held = str(np.unique(groups[test_idx])[0])
        folds.append(
            LodoFold(
                held_out_source=held,
                train_idx=train_idx.astype(np.int64, copy=False),
                test_idx=test_idx.astype(np.int64, copy=False),
            )
        )
    folds.sort(key=lambda f: f.held_out_source)
    yield from folds


def within_fold_split(
    train_idx: np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    val_frac: float = WITHIN_FOLD_VAL_FRAC,
    calib_frac: float = WITHIN_FOLD_CALIB_FRAC,
    seed: int = SEED,
) -> WithinFoldSplit:
    """Split one LODO fold's training rows into 80/10/10 train/val/calib.

    Stratified on ``y`` so each split preserves class balance. ``train_idx``
    is the positional index returned by :func:`iter_lodo_folds`; ``y`` is
    the full-frame target (we index into it positionally, then re-stratify).

    Determinism: same ``seed`` + same ``train_idx`` always yields the same
    partition.
    """
    if val_frac <= 0 or calib_frac <= 0:
        raise ValueError(f"val_frac and calib_frac must be > 0; got {val_frac}, {calib_frac}")
    if val_frac + calib_frac >= 1.0:
        raise ValueError(f"val_frac + calib_frac must be < 1.0; got {val_frac + calib_frac}")

    y_arr = np.asarray(y)
    y_train_pool = y_arr[train_idx]

    # First peel off the held-out (val + calib) chunk in one stratified split,
    # then split that chunk again into val vs calib. This gives a single,
    # deterministic, leakage-free 80/10/10.
    held_out_frac = val_frac + calib_frac
    train_inner_idx, held_out_inner_idx = train_test_split(
        np.arange(len(train_idx)),
        test_size=held_out_frac,
        stratify=y_train_pool,
        random_state=seed,
    )
    y_held_out_pool = y_train_pool[held_out_inner_idx]
    val_inner_idx, calib_inner_idx = train_test_split(
        held_out_inner_idx,
        test_size=calib_frac / held_out_frac,
        stratify=y_held_out_pool,
        random_state=seed,
    )

    return WithinFoldSplit(
        train_idx=train_idx[train_inner_idx].astype(np.int64, copy=False),
        val_idx=train_idx[val_inner_idx].astype(np.int64, copy=False),
        calib_idx=train_idx[calib_inner_idx].astype(np.int64, copy=False),
    )


def iter_random_kfold(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified random K-fold CV on the *union* of all sources.

    Reported only as the sanity-check baseline in `04-revised-design.md`
    §3.5: random K-fold mixes per-source case mix between train and test,
    so its discrimination metrics are systematically optimistic compared
    with LODO-CV. Never run as the headline; included so we can publish
    the gap.

    Yields ``(train_idx, test_idx)`` tuples in fold order.
    """
    _validate_columns(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X=df, y=df[TARGET_COLUMN]):
        yield (
            train_idx.astype(np.int64, copy=False),
            test_idx.astype(np.int64, copy=False),
        )
