"""Reliability diagrams (calibration plots).

A *reliability diagram* bins predictions by predicted probability,
computes the observed positive rate within each bin, and plots
observed-vs-predicted with a 45-degree perfect-calibration reference.
Bins above the diagonal indicate underconfidence; below indicates
overconfidence.

Two binning strategies:

- ``"uniform"``: equal-width bins on [0, 1]. Easy to interpret; bins in
  rare-probability regions can be empty / tiny.
- ``"quantile"``: equal-population bins. Every bin gets the same number
  of rows; bin widths vary. Better statistical reliability per bin,
  more cluttered visually.

We default to ``"quantile"`` with ``n_bins=10`` (deciles), which is the
modern convention ([Niculescu-Mizil & Caruana 2005](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf),
[scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html)).

The function returns the matplotlib ``Figure`` so the caller can save
it to PNG, embed it in a notebook, or further customise it. We never
``plt.show()`` — that's a UI decision the caller owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

#: Default bin count and strategy.
DEFAULT_N_BINS: Final[int] = 10
DEFAULT_STRATEGY: Final[Literal["uniform", "quantile"]] = "quantile"


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin counts, mean predicted probability, observed positive rate."""

    bin_edges: np.ndarray
    n_per_bin: np.ndarray
    mean_predicted: np.ndarray
    observed_rate: np.ndarray


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Compute bin edges per the chosen strategy."""
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # np.quantile with linspace gives equal-population bins (mod ties).
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # Pin the outer edges so anything 0 or 1 is included.
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        # De-duplicate ties (e.g. p mostly identical) so np.histogram works.
        return np.unique(edges)
    raise ValueError(f"strategy must be 'uniform' or 'quantile'; got {strategy!r}")


def reliability_bins(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: Literal["uniform", "quantile"] = DEFAULT_STRATEGY,
) -> ReliabilityBins:
    """Bin predictions and compute the per-bin (predicted, observed) pairs."""
    y = np.asarray(y_true).ravel().astype(np.int64, copy=False)
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_proba shape mismatch: {y.shape} vs {p.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    edges = _bin_edges(p, n_bins, strategy)
    bin_idx = np.digitize(p, edges[1:-1], right=False)

    actual_n_bins = len(edges) - 1
    counts = np.zeros(actual_n_bins, dtype=np.int64)
    mean_pred = np.full(actual_n_bins, np.nan, dtype=np.float64)
    obs_rate = np.full(actual_n_bins, np.nan, dtype=np.float64)
    for b in range(actual_n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        counts[b] = n
        if n > 0:
            mean_pred[b] = float(p[mask].mean())
            obs_rate[b] = float(y[mask].mean())

    return ReliabilityBins(
        bin_edges=edges,
        n_per_bin=counts,
        mean_predicted=mean_pred,
        observed_rate=obs_rate,
    )


def reliability_diagram(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: Literal["uniform", "quantile"] = DEFAULT_STRATEGY,
    title: str | None = None,
) -> Figure:
    """Build a reliability diagram and return the matplotlib ``Figure``.

    The figure has two stacked axes: the top is the calibration curve
    (mean-predicted vs observed-rate, with a y=x perfect-calibration
    reference); the bottom is a histogram of the predicted-probability
    distribution. Caller is responsible for ``fig.savefig(...)`` or
    ``plt.close(fig)``.
    """
    bins = reliability_bins(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    populated = bins.n_per_bin > 0
    ax_cal.plot(
        bins.mean_predicted[populated],
        bins.observed_rate[populated],
        marker="o",
        linewidth=1.5,
        label="model",
    )
    ax_cal.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="perfect")
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_ylabel("observed positive rate")
    ax_cal.legend(loc="upper left")
    if title:
        ax_cal.set_title(title)
    ax_cal.grid(alpha=0.3)

    p_arr = np.asarray(y_proba, dtype=np.float64).ravel()
    ax_hist.hist(p_arr, bins=bins.bin_edges, edgecolor="black", linewidth=0.5)
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("count")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    return fig
