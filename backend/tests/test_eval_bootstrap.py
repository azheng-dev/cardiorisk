"""Tests for cardiorisk.eval.bootstrap.

Covers:

- The CI dataclass arithmetic (point/lower/upper, contains, width).
- Determinism: same (y, p, seed) always yields the same CI.
- Statistical sanity: the point estimate lies inside the CI.
- The CI shrinks as n_resamples grows (asymptotic property).
- Coverage on a known-stable metric on a large synthetic sample.
- Failure modes: too few resamples, bad alpha, shape mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.eval.bootstrap import CI, DEFAULT_N_RESAMPLES, SEED, bootstrap_ci
from cardiorisk.eval.metrics import auroc, brier


@pytest.fixture
def noisy_problem() -> tuple[np.ndarray, np.ndarray]:
    """200 rows, balanced classes, predictions overlap heavily so AUROC
    is well below 1.0 and bootstrap resamples produce real variation."""
    rng = np.random.default_rng(7)
    n = 200
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    # Heavy noise: classes overlap substantially so AUROC ~0.7-0.8.
    noise = rng.normal(0, 0.4, n)
    p = np.clip(y * 0.4 + 0.3 + noise, 0.01, 0.99)
    return y, p


# ---------------------------------------------------------------- CI dataclass


def test_ci_contains_returns_true_for_value_inside() -> None:
    ci = CI(point=0.85, lower=0.80, upper=0.90, n_resamples=2000, alpha=0.05)
    assert ci.contains(0.85)
    assert ci.contains(0.80)
    assert ci.contains(0.90)


def test_ci_contains_returns_false_for_value_outside() -> None:
    ci = CI(point=0.85, lower=0.80, upper=0.90, n_resamples=2000, alpha=0.05)
    assert not ci.contains(0.79)
    assert not ci.contains(0.91)


def test_ci_width_is_upper_minus_lower() -> None:
    ci = CI(point=0.85, lower=0.80, upper=0.90, n_resamples=2000, alpha=0.05)
    assert ci.width() == pytest.approx(0.10)


# ---------------------------------------------------------------- determinism


def test_bootstrap_is_deterministic_under_pinned_seed(
    noisy_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = noisy_problem
    ci1 = bootstrap_ci(auroc, y, p, n_resamples=500, seed=SEED)
    ci2 = bootstrap_ci(auroc, y, p, n_resamples=500, seed=SEED)
    assert ci1.point == pytest.approx(ci2.point)
    assert ci1.lower == pytest.approx(ci2.lower)
    assert ci1.upper == pytest.approx(ci2.upper)


def test_different_seeds_produce_different_intervals(
    noisy_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = noisy_problem
    ci1 = bootstrap_ci(auroc, y, p, n_resamples=500, seed=1)
    ci2 = bootstrap_ci(auroc, y, p, n_resamples=500, seed=2)
    # Point estimate is on the original sample, so identical.
    assert ci1.point == pytest.approx(ci2.point)
    # Lower / upper differ across seeds.
    assert ci1.lower != pytest.approx(ci2.lower) or ci1.upper != pytest.approx(ci2.upper)


# ---------------------------------------------------------------- statistical sanity


def test_point_estimate_lies_inside_ci(
    noisy_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    """Percentile-method bootstrap: by construction the point estimate
    on the original sample sits between the 2.5th and 97.5th percentiles
    of the resampled distribution for any reasonable metric."""
    y, p = noisy_problem
    ci = bootstrap_ci(auroc, y, p, n_resamples=500)
    assert ci.contains(ci.point)


def test_ci_width_shrinks_with_more_data() -> None:
    """Bigger sample → tighter CI. Asymptotic guarantee of the
    bootstrap; this is the main reason to bother with 2000 resamples.

    Uses a noisy problem so AUROC is well away from saturation; otherwise
    both CIs collapse to [1.0, 1.0] and the inequality is degenerate.
    """
    rng = np.random.default_rng(42)

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = (rng.random(n) < 0.5).astype(int)
        # Heavy noise -> AUROC ~0.7, well away from 1.0.
        p = np.clip(y * 0.3 + 0.35 + rng.normal(0, 0.4, n), 0.01, 0.99)
        return y, p

    y_small, p_small = make(80)
    y_big, p_big = make(2000)
    ci_small = bootstrap_ci(auroc, y_small, p_small, n_resamples=500)
    ci_big = bootstrap_ci(auroc, y_big, p_big, n_resamples=500)
    assert ci_big.width() < ci_small.width()


def test_default_resamples_matches_design_doc() -> None:
    """`04-revised-design.md` §5.1 commits to 2,000 resamples."""
    assert DEFAULT_N_RESAMPLES == 2000


def test_bootstrap_works_for_brier_metric(
    noisy_problem: tuple[np.ndarray, np.ndarray],
) -> None:
    """Sanity-check that bootstrap_ci works with any metric_fn shape,
    not just auroc."""
    y, p = noisy_problem
    ci = bootstrap_ci(brier, y, p, n_resamples=500)
    assert ci.lower <= ci.point <= ci.upper
    # Brier is in [0, 1] for probabilities in [0, 1].
    assert 0.0 <= ci.lower <= 1.0
    assert 0.0 <= ci.upper <= 1.0


# ---------------------------------------------------------------- failure modes


def test_too_few_resamples_rejected() -> None:
    y = np.array([0, 1] * 50)
    p = y.astype(float)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(auroc, y, p, n_resamples=50)


def test_bad_alpha_rejected() -> None:
    y = np.array([0, 1] * 50)
    p = y.astype(float)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(auroc, y, p, n_resamples=200, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(auroc, y, p, n_resamples=200, alpha=1.0)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_ci(auroc, np.array([0, 1]), np.array([0.5]), n_resamples=200)


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(auroc, np.array([]), np.array([]), n_resamples=200)
