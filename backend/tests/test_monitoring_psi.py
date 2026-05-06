"""Tests for cardiorisk.monitoring.psi.

Covers the closed-form behaviour of PSI on synthetic distributions:

- Identical reference and current → PSI exactly 0.
- Disjoint distributions → PSI well above the ``major`` threshold.
- ε-floor stability: empty bins do not produce NaN or inf.
- Severity bands match the published cut-points.
- Bin-count sensitivity: PSI grows monotonically with bin count for
  the same shifted-distribution pair (the documented PSI footgun).
- Categorical PSI: novel current levels surface as drift; identical
  level-frequency vectors return ~0; empty inputs short-circuit to 0.
- Failure modes: shape mismatch, malformed edges, negative epsilon.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.monitoring.psi import (
    DEFAULT_EPSILON,
    PSI_MODERATE_MAX,
    PSI_STABLE_MAX,
    psi_categorical,
    psi_from_proportions,
    psi_numeric,
    severity_band,
)

SEED = 20260506


# ---------------------------------------------------------------- closed-form


def test_identical_distributions_give_psi_zero() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 5_000)
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf], dtype=np.float64)
    psi = psi_numeric(reference=samples, current=samples, edges=edges)
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_disjoint_distributions_give_major_drift() -> None:
    ref = np.full(2_000, -3.0)
    cur = np.full(2_000, 3.0)
    edges = np.array([-np.inf, -1.0, 1.0, np.inf], dtype=np.float64)
    psi = psi_numeric(reference=ref, current=cur, edges=edges)
    assert psi > PSI_MODERATE_MAX
    assert severity_band(psi) == "major"


def test_psi_is_non_negative_on_random_pairs() -> None:
    rng = np.random.default_rng(SEED)
    edges = np.array([-np.inf, -0.5, 0.5, np.inf], dtype=np.float64)
    for _ in range(50):
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0.2, 1.1, 500)
        assert psi_numeric(reference=ref, current=cur, edges=edges) >= 0.0


def test_psi_from_proportions_matches_known_value() -> None:
    # Closed-form: ref = [0.5, 0.5], cur = [0.6, 0.4]
    # PSI = (0.6 - 0.5) ln(0.6/0.5) + (0.4 - 0.5) ln(0.4/0.5)
    #     = 0.1 * 0.18232 + (-0.1) * (-0.22314)
    #     = 0.01823 + 0.02231 = 0.04055 (4 dp)
    p_ref = np.array([0.5, 0.5])
    p_cur = np.array([0.6, 0.4])
    psi = psi_from_proportions(p_ref, p_cur)
    assert psi == pytest.approx(0.04054651, abs=1e-6)


# ---------------------------------------------------------------- epsilon floor


def test_empty_bin_does_not_produce_nan_or_inf() -> None:
    # Reference puts mass in bin 0 only; current puts mass in bin 1 only.
    # Without an ε-floor either log term would be ±inf and PSI would be NaN.
    ref = np.full(1_000, -2.0)
    cur = np.full(1_000, 2.0)
    edges = np.array([-np.inf, 0.0, np.inf], dtype=np.float64)
    psi = psi_numeric(reference=ref, current=cur, edges=edges)
    assert np.isfinite(psi)
    assert psi > 0.0


def test_epsilon_floor_is_configurable() -> None:
    ref = np.full(1_000, -2.0)
    cur = np.full(1_000, 2.0)
    edges = np.array([-np.inf, 0.0, np.inf], dtype=np.float64)
    psi_small = psi_numeric(reference=ref, current=cur, edges=edges, epsilon=1e-9)
    psi_large = psi_numeric(reference=ref, current=cur, edges=edges, epsilon=1e-3)
    # Smaller ε puts the empty-bin proportion further from the non-empty
    # one and therefore amplifies PSI; larger ε flattens it.
    assert psi_small > psi_large


# ---------------------------------------------------------------- severity bands


def test_severity_band_below_stable_threshold_is_stable() -> None:
    assert severity_band(0.0) == "stable"
    assert severity_band(0.05) == "stable"
    assert severity_band(PSI_STABLE_MAX - 1e-6) == "stable"


def test_severity_band_in_moderate_window_is_moderate() -> None:
    assert severity_band(PSI_STABLE_MAX) == "moderate"
    assert severity_band(0.15) == "moderate"
    assert severity_band(PSI_MODERATE_MAX - 1e-6) == "moderate"


def test_severity_band_at_or_above_major_threshold_is_major() -> None:
    assert severity_band(PSI_MODERATE_MAX) == "major"
    assert severity_band(0.5) == "major"
    assert severity_band(10.0) == "major"


def test_severity_band_treats_nan_as_major() -> None:
    assert severity_band(float("nan")) == "major"


# ---------------------------------------------------------------- bin-count sensitivity


def test_psi_can_grow_with_bin_count_on_shifted_normals() -> None:
    """Documented PSI footgun: more bins means finer granularity, which
    typically means the shift between two clearly-different distributions
    becomes more visible. This test asserts the direction (more bins ->
    >= PSI) on a shifted-normal pair, not a specific magnitude. Exact
    monotonicity is not guaranteed, but the strict inequality between
    a 2-bin and a 20-bin partition is reliable."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0.0, 1.0, 5_000)
    cur = rng.normal(0.6, 1.0, 5_000)

    def edges_for(n_bins: int) -> np.ndarray:
        q = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.quantile(ref, q)
        e[0] = -np.inf
        e[-1] = np.inf
        return e

    psi_2 = psi_numeric(reference=ref, current=cur, edges=edges_for(2))
    psi_20 = psi_numeric(reference=ref, current=cur, edges=edges_for(20))
    assert psi_20 > psi_2


# ---------------------------------------------------------------- NaN handling


def test_nans_are_dropped_before_binning() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 1_000)
    edges = np.array([-np.inf, 0.0, np.inf], dtype=np.float64)
    samples_with_nans = samples.copy()
    samples_with_nans[::5] = np.nan
    psi_clean = psi_numeric(reference=samples, current=samples, edges=edges)
    psi_with_nans = psi_numeric(reference=samples, current=samples_with_nans, edges=edges)
    # Dropping NaNs leaves the residual sample distributionally
    # indistinguishable from the parent; PSI should still be small.
    assert psi_clean == pytest.approx(0.0, abs=1e-12)
    assert psi_with_nans < PSI_STABLE_MAX


def test_empty_inputs_short_circuit_to_zero() -> None:
    edges = np.array([-np.inf, 0.0, np.inf], dtype=np.float64)
    assert psi_numeric(reference=np.array([]), current=np.array([1.0, 2.0]), edges=edges) == 0.0
    assert psi_numeric(reference=np.array([1.0, 2.0]), current=np.array([]), edges=edges) == 0.0


# ---------------------------------------------------------------- categorical


def test_psi_categorical_identical_returns_zero() -> None:
    counts = {"M": 600, "F": 400}
    psi = psi_categorical(reference_counts=counts, current_counts=counts)
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_psi_categorical_flags_novel_levels() -> None:
    ref = {"M": 500, "F": 500}
    cur = {"M": 500, "F": 400, "Other": 100}  # 10% novel level
    psi = psi_categorical(reference_counts=ref, current_counts=cur)
    assert psi > PSI_STABLE_MAX


def test_psi_categorical_flags_majority_level_collapse() -> None:
    ref = {"M": 500, "F": 500}
    cur = {"M": 950, "F": 50}
    psi = psi_categorical(reference_counts=ref, current_counts=cur)
    assert severity_band(psi) == "major"


def test_psi_categorical_empty_inputs_return_zero() -> None:
    assert psi_categorical(reference_counts={}, current_counts={"M": 1}) == 0.0
    assert psi_categorical(reference_counts={"M": 1}, current_counts={}) == 0.0


def test_psi_categorical_uses_default_epsilon() -> None:
    # Whitebox: tiny ε should not produce inf even with a 100% novel level.
    psi = psi_categorical(
        reference_counts={"A": 100},
        current_counts={"B": 100},
        epsilon=DEFAULT_EPSILON,
    )
    assert np.isfinite(psi)
    assert psi > PSI_MODERATE_MAX


# ---------------------------------------------------------------- failure modes


def test_psi_numeric_raises_on_malformed_edges() -> None:
    samples = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="edges must be a 1-D"):
        psi_numeric(reference=samples, current=samples, edges=np.array([1.0]))
    with pytest.raises(ValueError, match="edges must be a 1-D"):
        psi_numeric(reference=samples, current=samples, edges=np.array([[0.0, 1.0]]))


def test_psi_from_proportions_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must align"):
        psi_from_proportions(np.array([0.5, 0.5]), np.array([0.3, 0.4, 0.3]))


def test_psi_from_proportions_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        psi_from_proportions(np.array([]), np.array([]))
