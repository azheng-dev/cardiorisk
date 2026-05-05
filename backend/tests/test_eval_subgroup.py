"""Tests for cardiorisk.eval.subgroup.

Covers:

- assign_age_band: cut-points match `04-revised-design.md` §5.2.
- stratified_metrics: per-stratum n + value, fairness gap = max - min.
- min_stratum_size: undersized strata get NaN values, are excluded
  from the gap calculation but still reported.
- fairness_gap helper: NaN-aware max - min.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.eval.metrics import auroc
from cardiorisk.eval.subgroup import (
    AGE_BANDS,
    StratifiedReport,
    SubgroupResult,
    assign_age_band,
    fairness_gap,
    stratified_metrics,
)

# ---------------------------------------------------------------- assign_age_band


def test_age_band_cut_points_match_design_doc() -> None:
    """`04-revised-design.md` §5.2 commits to <50 / 50-69 / >=70."""
    bands = {label for _, _, label in AGE_BANDS}
    assert bands == {"<50", "50-69", ">=70"}


def test_assign_age_band_boundary_cases() -> None:
    assert assign_age_band(49) == "<50"
    assert assign_age_band(50) == "50-69"
    assert assign_age_band(69) == "50-69"
    assert assign_age_band(70) == ">=70"
    assert assign_age_band(0) == "<50"
    assert assign_age_band(120) == ">=70"


def test_assign_age_band_handles_nan() -> None:
    assert assign_age_band(float("nan")) == "unknown"


def test_assign_age_band_handles_negative_age() -> None:
    """Defensive: negative ages are unknown rather than silently mapped to <50."""
    assert assign_age_band(-1) == "unknown"


# ---------------------------------------------------------------- stratified_metrics


@pytest.fixture
def two_strata_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """200 rows split 100/100 between groups M and F. M gets a perfect
    predictor; F gets a random one. Fairness gap should be ~0.5."""
    rng = np.random.default_rng(42)
    n_per = 100
    y_m = np.array([0] * (n_per // 2) + [1] * (n_per // 2))
    p_m = y_m.astype(float) * 0.9 + 0.05  # perfect
    y_f = (rng.random(n_per) < 0.5).astype(int)
    p_f = rng.random(n_per)  # random
    y = np.concatenate([y_m, y_f])
    p = np.concatenate([p_m, p_f])
    s = np.array(["M"] * n_per + ["F"] * n_per)
    return y, p, s


def test_stratified_metrics_returns_one_entry_per_stratum(
    two_strata_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, p, s = two_strata_problem
    report = stratified_metrics(y, p, s, auroc, metric_name="auroc", grouping_name="sex")
    assert isinstance(report, StratifiedReport)
    assert len(report.by_stratum) == 2
    assert {r.stratum for r in report.by_stratum} == {"M", "F"}


def test_stratified_metrics_per_stratum_n_correct(
    two_strata_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, p, s = two_strata_problem
    report = stratified_metrics(y, p, s, auroc, metric_name="auroc", grouping_name="sex")
    n_dict = report.n_dict()
    assert n_dict["M"] == 100
    assert n_dict["F"] == 100


def test_stratified_metrics_perfect_stratum_gets_auroc_one(
    two_strata_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, p, s = two_strata_problem
    report = stratified_metrics(y, p, s, auroc, metric_name="auroc", grouping_name="sex")
    values = report.values_dict()
    assert values["M"] == pytest.approx(1.0)
    # F is random; AUROC ~0.5 with sampling noise.
    assert 0.2 < values["F"] < 0.8


def test_stratified_metrics_fairness_gap_is_max_minus_min(
    two_strata_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, p, s = two_strata_problem
    report = stratified_metrics(y, p, s, auroc, metric_name="auroc", grouping_name="sex")
    values = list(report.values_dict().values())
    assert report.fairness_gap == pytest.approx(max(values) - min(values))


def test_stratified_metrics_undersized_stratum_gets_nan_value() -> None:
    """A stratum with fewer than min_stratum_size rows is reported with
    NaN value but still appears in the by_stratum list."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    p = np.linspace(0.1, 0.9, 12)
    s = np.array(["A"] * 10 + ["B"] * 2)  # B has only 2 rows
    report = stratified_metrics(
        y, p, s, auroc, metric_name="auroc", grouping_name="grp", min_stratum_size=5
    )
    by = {r.stratum: r for r in report.by_stratum}
    assert "B" in by
    assert np.isnan(by["B"].value)
    assert by["B"].n == 2
    # gap is NaN because only A has a valid value.
    assert np.isnan(report.fairness_gap)


def test_stratified_metrics_sorts_strata_alphabetically(
    two_strata_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, p, s = two_strata_problem
    report = stratified_metrics(y, p, s, auroc, metric_name="auroc", grouping_name="sex")
    labels = [r.stratum for r in report.by_stratum]
    assert labels == sorted(labels)


def test_stratified_metrics_handles_metric_function_error_gracefully() -> None:
    """If a metric raises (e.g. only one class in a stratum), the
    stratum gets NaN rather than crashing the whole report."""
    y = np.zeros(20, dtype=int)
    p = np.linspace(0.1, 0.9, 20)
    s = np.array(["only_neg"] * 20)
    report = stratified_metrics(
        y, p, s, auroc, metric_name="auroc", grouping_name="grp", min_stratum_size=5
    )
    # auroc returns NaN for single-class input; report.fairness_gap is NaN
    # because there's only one stratum.
    by = {r.stratum: r for r in report.by_stratum}
    assert np.isnan(by["only_neg"].value)
    assert np.isnan(report.fairness_gap)


# ---------------------------------------------------------------- fairness_gap helper


def test_fairness_gap_helper_basic() -> None:
    assert fairness_gap({"a": 0.8, "b": 0.6, "c": 0.7}) == pytest.approx(0.2)


def test_fairness_gap_helper_ignores_nan() -> None:
    assert fairness_gap({"a": 0.8, "b": float("nan"), "c": 0.6}) == pytest.approx(0.2)


def test_fairness_gap_helper_returns_nan_with_under_two_valid() -> None:
    assert np.isnan(fairness_gap({"a": 0.8}))
    assert np.isnan(fairness_gap({"a": 0.8, "b": float("nan")}))


# ---------------------------------------------------------------- result dataclass


def test_subgroup_result_dataclass_fields() -> None:
    r = SubgroupResult(stratum="M", n=100, value=0.85)
    assert r.stratum == "M"
    assert r.n == 100
    assert r.value == 0.85
