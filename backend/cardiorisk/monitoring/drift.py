"""End-to-end drift computation against a fold's reference snapshot.

:func:`compute_drift` is the only public surface here. It takes:

- a :class:`~cardiorisk.monitoring.reference.FoldReference` built at
  training time (or by the one-shot ``backend/scripts/build_reference.py``
  driver),
- a "current" data slice (a :class:`pandas.DataFrame` with the same HFP
  schema), and
- a calibrated model artefact registered under a model-name key in the
  reference's ``prediction`` dict (optional — if the model isn't in the
  reference, prediction-drift is skipped).

It returns a :class:`DriftReport` that holds:

- per-feature input drift (PSI for every numeric and categorical
  feature in the reference; KS statistic + p-value for numerics),
- prediction drift (PSI on ``predict_proba(X_current)[:, 1]`` against
  the same percentile-binning that was applied at reference time),
- aggregate severity counts (n features in each band).

Report-only: ADR-014 explicitly defers any "auto-block-deployment"
behaviour. The orchestrator just dumps the report to JSON + a single
dashboard PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cardiorisk.monitoring.ks import KSResult, ks_two_sample
from cardiorisk.monitoring.psi import (
    SeverityBand,
    psi_categorical,
    psi_from_counts,
    severity_band,
)
from cardiorisk.monitoring.reference import FoldReference, bin_counts


@dataclass(frozen=True)
class FeatureDrift:
    """Per-feature drift block. ``ks_*`` are populated for numerics only."""

    feature: str
    kind: str  # "numeric" | "categorical"
    psi: float
    severity: SeverityBand
    n_ref: int
    n_cur: int
    n_missing_cur: int
    ks_statistic: float | None
    ks_p_value: float | None


@dataclass(frozen=True)
class PredictionDrift:
    """Prediction-drift block (calibrated predict_proba, PSI on quantile bins)."""

    model_name: str
    psi: float
    severity: SeverityBand
    n_ref: int
    n_cur: int
    mean_ref: float
    mean_cur: float


@dataclass(frozen=True)
class DriftReport:
    """All drift signals for one (model, fold) cell."""

    held_out_source: str
    model_name: str
    n_current: int
    per_feature: dict[str, FeatureDrift]
    prediction: PredictionDrift | None
    severity_counts: dict[SeverityBand, int]

    def top_drifted_features(self, k: int = 3) -> list[FeatureDrift]:
        """Return the top-``k`` features by PSI, descending. Stable on ties."""
        return sorted(
            self.per_feature.values(),
            key=lambda fd: (-fd.psi, fd.feature),
        )[:k]


def _bin_midpoints(edges: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bin midpoints for the (n_bins + 1)-edge vector.

    Outer bins have ``±inf`` edges; we synthesise a finite midpoint by
    extending past the next inner edge by the inner-data span. The
    exact value barely matters for KS — the statistic is rank-based on
    the empirical CDF, so the midpoint only sets the ordering of the
    synthesised samples within their bin.
    """
    finite = np.isfinite(edges)
    inner = edges[finite]
    if inner.size == 0:
        return np.array([0.0], dtype=np.float64)
    span = max(float(inner[-1] - inner[0]), 1.0)
    extended = edges.copy()
    if not np.isfinite(extended[0]):
        extended[0] = float(inner[0]) - span
    if not np.isfinite(extended[-1]):
        extended[-1] = float(inner[-1]) + span
    return ((extended[:-1] + extended[1:]) / 2.0).astype(np.float64)


def _drift_for_numeric(
    *,
    feature: str,
    reference_edges: npt.NDArray[np.float64],
    reference_counts: npt.NDArray[np.int64],
    reference_n: int,
    current_values: npt.NDArray[np.float64],
) -> FeatureDrift:
    """PSI + KS for one numeric feature against a pre-binned reference.

    The reference bin counts are reused directly from the persisted
    :class:`~cardiorisk.monitoring.reference.NumericReference`. The KS
    sanity-check, however, needs raw samples — we approximate those by
    the bin midpoints weighted by the reference counts. ADR-014
    documents the trade-off (the alternative would be persisting full
    reference samples, which several-x's the reference-artefact size
    on disk for marginal sanity-check gain).
    """
    n_missing = int(np.isnan(current_values).sum())
    cur_clean = current_values[~np.isnan(current_values)]

    cur_counts = bin_counts(cur_clean, reference_edges)
    psi_value = psi_from_counts(
        reference_counts=reference_counts,
        current_counts=cur_counts,
    )

    ks_value: KSResult | None = None
    if reference_n > 0 and cur_clean.size > 0 and reference_counts.size >= 1:
        ref_midpoints = _bin_midpoints(reference_edges)
        ref_synthetic = np.repeat(ref_midpoints, reference_counts)
        ks_value = ks_two_sample(reference=ref_synthetic, current=cur_clean)

    return FeatureDrift(
        feature=feature,
        kind="numeric",
        psi=psi_value,
        severity=severity_band(psi_value),
        n_ref=reference_n,
        n_cur=int(cur_clean.size),
        n_missing_cur=n_missing,
        ks_statistic=None if ks_value is None else ks_value.statistic,
        ks_p_value=None if ks_value is None else ks_value.p_value,
    )


def _drift_for_categorical(
    *,
    feature: str,
    reference_counts: dict[str, int],
    current_values: pd.Series,
) -> FeatureDrift:
    """PSI for one categorical feature."""
    n_missing = int(current_values.isna().sum())
    levels = current_values.dropna().astype(str)
    cur_counts: dict[str, int] = {str(k): int(v) for k, v in levels.value_counts().items()}
    psi_value = psi_categorical(reference_counts=reference_counts, current_counts=cur_counts)
    return FeatureDrift(
        feature=feature,
        kind="categorical",
        psi=psi_value,
        severity=severity_band(psi_value),
        n_ref=int(sum(reference_counts.values())),
        n_cur=int(levels.size),
        n_missing_cur=n_missing,
        ks_statistic=None,
        ks_p_value=None,
    )


def _drift_for_prediction(
    *,
    model_name: str,
    reference_edges: npt.NDArray[np.float64],
    reference_counts: npt.NDArray[np.int64],
    reference_n: int,
    reference_mean: float,
    current_proba: npt.NDArray[np.float64],
) -> PredictionDrift:
    """PSI on a model's ``predict_proba`` against the reference percentile bins."""
    cur_counts = bin_counts(current_proba, reference_edges)
    psi_value = psi_from_counts(
        reference_counts=reference_counts,
        current_counts=cur_counts,
    )
    return PredictionDrift(
        model_name=model_name,
        psi=psi_value,
        severity=severity_band(psi_value),
        n_ref=reference_n,
        n_cur=int(current_proba.size),
        mean_ref=reference_mean,
        mean_cur=float(current_proba.mean()) if current_proba.size > 0 else float("nan"),
    )


def compute_drift(
    *,
    reference: FoldReference,
    X_current: pd.DataFrame,
    model: Any | None = None,
    model_name: str | None = None,
) -> DriftReport:
    """Compute per-feature + (optional) prediction drift for one (model, fold).

    Parameters
    ----------
    reference : FoldReference
        Built at training time on the in-fold training pool.
    X_current : pd.DataFrame
        The "current" slice. For the Phase-2.6 demo run this is the
        fold's held-out LODO source (i.e. the data the model was *not*
        trained on); see ADR-014.
    model : optional sklearn-like estimator
        If provided, ``model.predict_proba(X_current)[:, 1]`` is used
        for prediction-drift PSI against the reference's percentile
        bins for the corresponding ``model_name`` entry.
    model_name : optional str
        Required when ``model`` is provided; selects which
        :class:`~cardiorisk.monitoring.reference.PredictionReference`
        the prediction-drift PSI is scored against.
    """
    if model is not None and model_name is None:
        raise ValueError("model_name is required when model is provided")

    per_feature: dict[str, FeatureDrift] = {}

    for feature, num_ref in reference.numeric.items():
        if feature not in X_current.columns:
            continue
        cur_values = X_current[feature].to_numpy(dtype=np.float64, na_value=np.nan)
        per_feature[feature] = _drift_for_numeric(
            feature=feature,
            reference_edges=num_ref.edges,
            reference_counts=num_ref.counts,
            reference_n=num_ref.n,
            current_values=cur_values,
        )

    for feature, cat_ref in reference.categorical.items():
        if feature not in X_current.columns:
            continue
        per_feature[feature] = _drift_for_categorical(
            feature=feature,
            reference_counts=cat_ref.counts,
            current_values=X_current[feature],
        )

    prediction: PredictionDrift | None = None
    if model is not None and model_name is not None:
        if model_name not in reference.prediction:
            raise KeyError(
                f"reference has no prediction binning for model {model_name!r}; "
                f"available: {sorted(reference.prediction)}"
            )
        pred_ref = reference.prediction[model_name]
        proba = np.asarray(model.predict_proba(X_current), dtype=np.float64)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(
                f"model {model_name!r} did not return a 2-D (n, 2) predict_proba; "
                f"got shape {proba.shape}"
            )
        prediction = _drift_for_prediction(
            model_name=model_name,
            reference_edges=pred_ref.edges,
            reference_counts=pred_ref.counts,
            reference_n=pred_ref.n,
            reference_mean=pred_ref.mean,
            current_proba=proba[:, 1],
        )

    severity_counts: dict[SeverityBand, int] = {"stable": 0, "moderate": 0, "major": 0}
    for fd in per_feature.values():
        severity_counts[fd.severity] += 1

    return DriftReport(
        held_out_source=reference.held_out_source,
        model_name=model_name or "(no-model)",
        n_current=len(X_current),
        per_feature=per_feature,
        prediction=prediction,
        severity_counts=severity_counts,
    )
