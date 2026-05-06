"""Tests for cardiorisk.monitoring.ks.

Covers the thin scipy wrapper:

- Same-distribution → high p-value.
- Shifted-distribution → tiny p-value, large statistic.
- Empty inputs → degenerate "no evidence" outcome with sample sizes preserved.
- NaN handling matches the PSI helpers (drop, do not propagate).
- Statistic and p-value are real floats (not numpy scalars), so JSON
  serialisation downstream is straightforward.
"""

from __future__ import annotations

import numpy as np

from cardiorisk.monitoring.ks import KSResult, ks_two_sample

SEED = 20260506


def test_ks_returns_high_p_value_for_same_distribution() -> None:
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 1_000)
    cur = rng.normal(0, 1, 1_000)
    res = ks_two_sample(reference=ref, current=cur)
    assert isinstance(res, KSResult)
    assert res.p_value > 0.05  # almost always; deterministic with this seed


def test_ks_returns_tiny_p_value_for_shifted_distribution() -> None:
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 1_000)
    cur = rng.normal(2.0, 1, 1_000)  # mean shift of 2 SDs
    res = ks_two_sample(reference=ref, current=cur)
    assert res.p_value < 1e-6
    assert res.statistic > 0.5


def test_ks_drops_nans_before_computing() -> None:
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 1_000)
    cur = rng.normal(0, 1, 1_000)
    cur_with_nans = cur.copy()
    cur_with_nans[::5] = np.nan
    res = ks_two_sample(reference=ref, current=cur_with_nans)
    assert res.n_cur == 800  # 200 NaNs dropped from 1000
    assert res.p_value > 0.05


def test_ks_empty_input_returns_no_evidence() -> None:
    res = ks_two_sample(reference=np.array([]), current=np.array([1.0, 2.0]))
    assert res.statistic == 0.0
    assert res.p_value == 1.0
    assert res.n_ref == 0
    assert res.n_cur == 2

    res = ks_two_sample(reference=np.array([1.0, 2.0]), current=np.array([]))
    assert res.p_value == 1.0
    assert res.n_cur == 0


def test_ks_returns_python_floats_not_numpy_scalars() -> None:
    """JSON-serialisability requirement: the orchestrator dumps the
    statistic + p-value into reports/v1/drift/per_fold.json without
    coercion, so they must already be plain Python floats."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 100)
    cur = rng.normal(0, 1, 100)
    res = ks_two_sample(reference=ref, current=cur)
    assert type(res.statistic) is float
    assert type(res.p_value) is float
    assert type(res.n_ref) is int
    assert type(res.n_cur) is int


def test_ks_statistic_is_bounded_in_unit_interval() -> None:
    """KS statistic is the sup of an empirical-CDF difference, so it
    must lie in [0, 1] regardless of how shifted the inputs are."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(100, 1, 500)  # absurd shift
    res = ks_two_sample(reference=ref, current=cur)
    assert 0.0 <= res.statistic <= 1.0
