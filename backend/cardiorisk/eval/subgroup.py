"""Subgroup performance + fairness-gap reporting (TRIPOD+AI §5.2).

For every headline metric, report the metric *stratified* by:

- ``Sex`` (Male / Female from the HFP schema)
- ``Age band`` (<50 / 50-69 / >=70 — the cut-points
  ``04-revised-design.md`` §5.2 commits to)

Plus the **fairness gap** (max - min across the strata of each
grouping). Per the design doc, a fairness gap > 5 percentage points on
sensitivity gets a paragraph in the model card.

This module is grouping-agnostic: it takes a ``strata`` array (any
hashable per row) and applies the same metric function to each subset.
The Phase-2.3b reporter calls it once per (model, grouping) pair and
collates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Default age-band cut-points from `04-revised-design.md` §5.2.
#: Lower bound inclusive, upper bound exclusive on each bin.
AGE_BANDS: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 50, "<50"),
    (50, 70, "50-69"),
    (70, 200, ">=70"),
)


@dataclass(frozen=True)
class SubgroupResult:
    """One stratum's metric value plus its sample count."""

    stratum: str
    n: int
    value: float


@dataclass(frozen=True)
class StratifiedReport:
    """Per-stratum metrics + the fairness gap across them."""

    metric_name: str
    grouping_name: str
    by_stratum: tuple[SubgroupResult, ...]
    fairness_gap: float

    def values_dict(self) -> dict[str, float]:
        return {r.stratum: r.value for r in self.by_stratum}

    def n_dict(self) -> dict[str, int]:
        return {r.stratum: r.n for r in self.by_stratum}


def assign_age_band(age: float | int) -> str:
    """Map a numeric age to one of the design-doc bands. NaN -> 'unknown'."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def stratified_metrics(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    strata: npt.ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str,
    grouping_name: str,
    min_stratum_size: int = 10,
) -> StratifiedReport:
    """Compute ``metric_fn`` per stratum + the fairness gap across strata.

    Parameters
    ----------
    y_true, y_proba : array-like
        Ground-truth labels and predicted probabilities.
    strata : array-like
        Per-row stratum label. Same length as ``y_true``.
    metric_fn : callable
        Takes (y_true, y_proba) numpy arrays, returns a scalar.
    metric_name : str
        Name to attach to the report (e.g. ``"auroc"``).
    grouping_name : str
        Name of the grouping (e.g. ``"sex"``, ``"age_band"``).
    min_stratum_size : int
        Strata with fewer rows than this are dropped from the gap
        calculation (the per-stratum result is still emitted, with a
        NaN value, so it's visible in the report). Default 10.

    Returns
    -------
    StratifiedReport
        Per-stratum metric values + the gap (max - min) computed over
        strata with >= ``min_stratum_size`` rows. Gap is NaN if fewer
        than two strata meet the size cutoff.
    """
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    s = np.asarray(strata).ravel()
    if not (y.shape == p.shape == s.shape):
        raise ValueError(
            f"y_true / y_proba / strata shape mismatch: {y.shape} / {p.shape} / {s.shape}"
        )
    if y.size == 0:
        raise ValueError("inputs are empty")

    by_stratum: list[SubgroupResult] = []
    valid_values: list[float] = []
    for stratum_label in pd.unique(pd.Series(s)):
        mask = s == stratum_label
        n = int(mask.sum())
        if n < min_stratum_size:
            value = float("nan")
        else:
            try:
                value = float(metric_fn(y[mask], p[mask]))
            except (ValueError, ZeroDivisionError):
                value = float("nan")
            if not np.isnan(value):
                valid_values.append(value)
        by_stratum.append(SubgroupResult(stratum=str(stratum_label), n=n, value=value))

    gap = float("nan") if len(valid_values) < 2 else max(valid_values) - min(valid_values)

    # Sort by stratum label for stable report ordering.
    by_stratum.sort(key=lambda r: r.stratum)
    return StratifiedReport(
        metric_name=metric_name,
        grouping_name=grouping_name,
        by_stratum=tuple(by_stratum),
        fairness_gap=gap,
    )


def fairness_gap(stratified_values: dict[str, float]) -> float:
    """Standalone helper: max - min over the values, ignoring NaN."""
    valid = [v for v in stratified_values.values() if not (isinstance(v, float) and np.isnan(v))]
    if len(valid) < 2:
        return float("nan")
    return max(valid) - min(valid)
