"""Non-parametric bootstrap confidence intervals.

Per :doc:`../../../docs/research/04-revised-design.md` §5.1, every
headline metric is reported with a **2,000-resample bootstrap 95% CI**.
This module provides the generic machinery; the per-metric callers live
in the Phase-2.3b training driver.

API:

>>> from cardiorisk.eval.bootstrap import bootstrap_ci
>>> from cardiorisk.eval.metrics import auroc
>>> ci = bootstrap_ci(auroc, y_true, y_proba, n_resamples=2000)
>>> ci.point, ci.lower, ci.upper
(0.87, 0.83, 0.91)

Sampling protocol: case-resampling bootstrap (rows are the resampling
unit, with replacement). The metric is recomputed on each resample. The
returned CI uses the **percentile** method (2.5th, 97.5th percentiles
of the resampled metric distribution) — a deliberate choice over BCa
documented in :doc:`../../../docs/research/07-eval-design.md`.

Determinism: pinned to ``SEED = 20260505`` by default (same constant as
the rest of the project), so a given (y_true, y_proba, n_resamples)
input always yields the same CI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Pinned RNG seed (matches `cardiorisk.features.cv.SEED`).
SEED: Final[int] = 20260505

#: Default resample count from `04-revised-design.md` §5.1.
DEFAULT_N_RESAMPLES: Final[int] = 2000

#: Default CI level (95%).
DEFAULT_ALPHA: Final[float] = 0.05


@dataclass(frozen=True)
class CI:
    """Bootstrap confidence interval for a scalar metric."""

    point: float
    lower: float
    upper: float
    n_resamples: int
    alpha: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def width(self) -> float:
        return self.upper - self.lower


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = SEED,
) -> CI:
    """Percentile-method bootstrap CI for a metric of (y_true, y_proba).

    Parameters
    ----------
    metric_fn : callable
        Takes (y_true, y_proba) numpy arrays, returns a scalar.
    y_true, y_proba : array-like
        Ground-truth labels and predicted probabilities.
    n_resamples : int
        Number of bootstrap resamples. Default 2,000 per the design doc.
    alpha : float
        Significance level. Default 0.05 (95% CI).
    seed : int
        RNG seed. Default :data:`SEED` for repo-wide determinism.

    Returns
    -------
    CI
        ``point`` is the metric on the original sample; ``lower`` / ``upper``
        are the alpha/2 and 1-alpha/2 percentiles of the resampled
        distribution, after dropping any NaN resamples (which can occur
        when a resample happens to be all-one-class).
    """
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100; got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    y = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_proba shape mismatch: {y.shape} vs {p.shape}")
    n = y.size
    if n == 0:
        raise ValueError("y_true is empty; need at least one row")

    point = float(metric_fn(y, p))

    rng = np.random.default_rng(seed)
    resampled: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            value = float(metric_fn(y[idx], p[idx]))
        except (ValueError, ZeroDivisionError):
            # Some metrics raise on degenerate resamples (e.g. all one class).
            # Treat those as NaN and drop from the percentile calc.
            value = float("nan")
        resampled.append(value)

    arr = np.array(resampled, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size < n_resamples * 0.5:
        raise RuntimeError(
            f"only {valid.size}/{n_resamples} bootstrap resamples produced a valid "
            "metric value (>50% required); the input may be too small or too "
            "imbalanced for percentile bootstrap on this metric"
        )

    lower = float(np.quantile(valid, alpha / 2.0))
    upper = float(np.quantile(valid, 1.0 - alpha / 2.0))
    return CI(point=point, lower=lower, upper=upper, n_resamples=n_resamples, alpha=alpha)
