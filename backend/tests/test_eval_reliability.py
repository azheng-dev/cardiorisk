"""Tests for cardiorisk.eval.reliability.

Covers:

- Bin counts always sum to n (no rows lost).
- Quantile binning gives roughly equal-population bins.
- Uniform binning gives equal-width bins.
- A perfectly calibrated input lands on the y=x line.
- The figure builder returns a matplotlib Figure with two axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe to import after matplotlib is configured
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from cardiorisk.eval.reliability import (
    DEFAULT_N_BINS,
    DEFAULT_STRATEGY,
    ReliabilityBins,
    reliability_bins,
    reliability_diagram,
)


@pytest.fixture
def calibrated_problem() -> tuple[np.ndarray, np.ndarray]:
    """500 rows where p[i] is the true probability and y[i] ~ Bernoulli(p[i]).
    Per-bin observed rates should track mean predicted within each bin."""
    rng = np.random.default_rng(0)
    n = 500
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    return y, p


# ---------------------------------------------------------------- ReliabilityBins


def test_bins_sum_to_n(calibrated_problem: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = calibrated_problem
    bins = reliability_bins(y, p, n_bins=10)
    assert int(bins.n_per_bin.sum()) == y.size


def test_quantile_bins_have_roughly_equal_population(
    calibrated_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_problem
    bins = reliability_bins(y, p, n_bins=10, strategy="quantile")
    # Bins differ by at most 1 row (n=500, 10 bins -> 50 each).
    assert bins.n_per_bin.max() - bins.n_per_bin.min() <= 1


def test_uniform_bins_have_equal_width() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (p > 0.5).astype(int)
    bins = reliability_bins(y, p, n_bins=10, strategy="uniform")
    widths = np.diff(bins.bin_edges)
    assert np.allclose(widths, 0.1)


def test_perfectly_calibrated_input_lands_on_diagonal(
    calibrated_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    """For a calibrated generator, observed rate per bin tracks mean
    predicted per bin within sampling noise. Tolerance widens with bin
    count because each bin has fewer rows."""
    y, p = calibrated_problem
    bins = reliability_bins(y, p, n_bins=5, strategy="quantile")
    # Drop empty bins (rare with n=500, n_bins=5, but be safe).
    populated = bins.n_per_bin > 0
    diff = np.abs(bins.observed_rate[populated] - bins.mean_predicted[populated])
    assert diff.max() < 0.10  # 10% sampling tolerance


def test_bins_dataclass_attribute_shapes() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(100)
    y = (p > 0.5).astype(int)
    bins = reliability_bins(y, p, n_bins=10, strategy="uniform")
    assert isinstance(bins, ReliabilityBins)
    assert len(bins.n_per_bin) == 10
    assert len(bins.mean_predicted) == 10
    assert len(bins.observed_rate) == 10
    assert len(bins.bin_edges) == 11


def test_default_strategy_is_quantile() -> None:
    """Documented choice in the module docstring; lock it in."""
    assert DEFAULT_STRATEGY == "quantile"
    assert DEFAULT_N_BINS == 10


# ---------------------------------------------------------------- reliability_diagram


def test_reliability_diagram_returns_figure_with_two_axes(
    calibrated_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_problem
    fig = reliability_diagram(y, p, n_bins=10)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # calibration curve + histogram
    finally:
        plt.close(fig)


def test_reliability_diagram_accepts_title(
    calibrated_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_problem
    fig = reliability_diagram(y, p, title="my model")
    try:
        # Top axis (calibration) carries the title.
        assert fig.axes[0].get_title() == "my model"
    finally:
        plt.close(fig)


def test_reliability_diagram_can_be_saved_to_png(
    calibrated_problem: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    y, p = calibrated_problem
    fig = reliability_diagram(y, p)
    out = tmp_path / "rel.png"
    try:
        fig.savefig(out)
    finally:
        plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG, not an empty stub


# ---------------------------------------------------------------- input validation


def test_reliability_bins_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        reliability_bins(np.array([0, 1]), np.array([0.5]))


def test_reliability_bins_rejects_too_few_bins() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="n_bins must be"):
        reliability_bins(y, p, n_bins=1)


def test_reliability_bins_rejects_unknown_strategy() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="strategy"):
        reliability_bins(y, p, strategy="ad-hoc")  # type: ignore[arg-type]
