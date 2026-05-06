"""Per-fold reference snapshots.

A :class:`FoldReference` is the on-disk artefact that captures the
distribution one Phase-2.3b LODO fold's model was trained on, so that
later drift computations have a fixed comparand. One reference per fold
(per ADR-014) — never a single combined reference, because each LODO
model was fit on a different combined-3-source pool and therefore has a
different drift baseline.

Numeric features → quantile bin edges + reference bin counts. Default
10 equal-frequency bins (the industry PSI convention; ADR-014 surfaces
the bin-count sensitivity in the research doc).

Categorical features → per-level frequency dicts.

Prediction percentile bins → 10 equal-frequency bins computed on
``model.predict_proba(X_train)[:, 1]``. This becomes the "prediction
drift" comparand the orchestrator scores against the current slice's
predictions.

Persistence: joblib, mirroring the model-artefact storage contract of
ADR-010. Reference files live alongside the per-fold model artefacts at
``models/v1/<source>_reference.joblib`` and are gitignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

#: Default number of quantile bins for both numeric features and
#: prediction-drift binning. Industry PSI convention. Sensitivity to
#: bin count is documented in ``docs/research/11-drift-design.md``.
DEFAULT_N_BINS: Final[int] = 10


@dataclass(frozen=True)
class NumericReference:
    """Reference snapshot for one numeric feature."""

    feature: str
    edges: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    n: int  # post-NaN-drop sample size used to compute the reference
    n_missing: int


@dataclass(frozen=True)
class CategoricalReference:
    """Reference snapshot for one categorical feature."""

    feature: str
    counts: dict[str, int]
    n: int
    n_missing: int


@dataclass(frozen=True)
class PredictionReference:
    """Reference snapshot for the calibrated ``predict_proba`` of a model."""

    edges: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    n: int
    mean: float


@dataclass(frozen=True)
class FoldReference:
    """One LODO fold's full reference snapshot.

    Holds:

    - per-numeric-feature edges + counts
    - per-categorical-feature level-frequency dicts
    - per-model prediction-drift binning (``model_name -> PredictionReference``)
    - bookkeeping (held-out source, training sample size, n_bins)
    """

    held_out_source: str
    n_train: int
    n_bins: int
    numeric: dict[str, NumericReference]
    categorical: dict[str, CategoricalReference]
    prediction: dict[str, PredictionReference] = field(default_factory=dict)


def compute_quantile_edges(
    samples: npt.NDArray[np.float64],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> npt.NDArray[np.float64]:
    """Compute equal-frequency bin edges from a 1-D numeric reference sample.

    NaNs are dropped before edge computation. Duplicate edges are
    deduplicated (a low-cardinality reference — e.g. a discretised
    feature like an integer score — collapses repeated quantiles); the
    returned edge vector is monotonically strictly increasing.

    The outer edges are extended by a tiny fraction beyond the data
    range so :func:`numpy.searchsorted` does not place exactly-min /
    exactly-max values into the wrong outer bin. Same trick the standard
    PSI tutorials use.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    data = samples[~np.isnan(samples)]
    if data.size == 0:
        # Degenerate reference: single bin covering the full real line.
        return np.array([-np.inf, np.inf], dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(data, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if edges.size < 2:
        # Constant feature: collapse to a single bin spanning all of R.
        return np.array([-np.inf, np.inf], dtype=np.float64)
    return edges.astype(np.float64)


def _bin_counts(
    samples: npt.NDArray[np.float64], edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Count samples per bin, clipping out-of-range values into the outer bins."""
    n_bins = int(edges.size) - 1
    if samples.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    idx = np.clip(np.searchsorted(edges, samples, side="right") - 1, 0, n_bins - 1)
    return np.bincount(idx, minlength=n_bins).astype(np.int64)


def _build_numeric_reference(
    *,
    feature: str,
    column: pd.Series,
    n_bins: int,
) -> NumericReference:
    raw = column.to_numpy(dtype=np.float64, na_value=np.nan)
    n_missing = int(np.isnan(raw).sum())
    data = raw[~np.isnan(raw)]
    edges = compute_quantile_edges(data, n_bins=n_bins)
    counts = _bin_counts(data, edges)
    return NumericReference(
        feature=feature,
        edges=edges,
        counts=counts,
        n=int(data.size),
        n_missing=n_missing,
    )


def _build_categorical_reference(*, feature: str, column: pd.Series) -> CategoricalReference:
    n_missing = int(column.isna().sum())
    levels = column.dropna().astype(str)
    counts: dict[str, int] = {str(k): int(v) for k, v in levels.value_counts().items()}
    return CategoricalReference(
        feature=feature,
        counts=counts,
        n=int(levels.size),
        n_missing=n_missing,
    )


def _build_prediction_reference(
    *,
    proba: npt.NDArray[np.float64],
    n_bins: int,
) -> PredictionReference:
    edges = compute_quantile_edges(proba, n_bins=n_bins)
    counts = _bin_counts(proba, edges)
    mean_val = float(proba.mean()) if proba.size > 0 else float("nan")
    return PredictionReference(edges=edges, counts=counts, n=int(proba.size), mean=mean_val)


def build_fold_reference(
    *,
    held_out_source: str,
    X_train: pd.DataFrame,
    models: dict[str, Any] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS,
    categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> FoldReference:
    """Build one fold's reference snapshot from its training slice.

    Parameters
    ----------
    held_out_source : str
        The LODO source held out by this fold (i.e. the source the
        models were *not* trained on). Stored for bookkeeping.
    X_train : pd.DataFrame
        The fold's in-fold training rows. Must already be cleaned
        (``clean_for_modelling``); this matches what
        :func:`cardiorisk.training.train_v1` passes its model wrappers.
    models : optional mapping
        ``{model_name: calibrated_estimator}``. If provided, the
        per-model prediction-drift binning is populated from
        ``estimator.predict_proba(X_train)[:, 1]``.
    n_bins : int
        Number of quantile bins for both numeric features and
        prediction-drift binning. Default :data:`DEFAULT_N_BINS`.
    """
    numeric: dict[str, NumericReference] = {}
    for col in numeric_columns:
        if col not in X_train.columns:
            continue
        numeric[col] = _build_numeric_reference(feature=col, column=X_train[col], n_bins=n_bins)

    categorical: dict[str, CategoricalReference] = {}
    for col in categorical_columns:
        if col not in X_train.columns:
            continue
        categorical[col] = _build_categorical_reference(feature=col, column=X_train[col])

    prediction: dict[str, PredictionReference] = {}
    if models:
        for name, est in models.items():
            proba = np.asarray(est.predict_proba(X_train), dtype=np.float64)
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError(
                    f"model {name!r} did not return a 2-D (n, 2) predict_proba; got shape {proba.shape}"
                )
            prediction[name] = _build_prediction_reference(proba=proba[:, 1], n_bins=n_bins)

    return FoldReference(
        held_out_source=held_out_source,
        n_train=len(X_train),
        n_bins=n_bins,
        numeric=numeric,
        categorical=categorical,
        prediction=prediction,
    )


def save_reference(reference: FoldReference, path: Path) -> None:
    """Persist a :class:`FoldReference` to disk via joblib (ADR-010 contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reference, path)


def load_reference(path: Path) -> FoldReference:
    """Load a :class:`FoldReference` previously written by :func:`save_reference`."""
    if not path.exists():
        raise FileNotFoundError(f"reference artefact missing: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, FoldReference):
        raise TypeError(f"expected FoldReference at {path}, got {type(obj).__name__}")
    return obj
