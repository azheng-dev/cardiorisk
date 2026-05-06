"""Tests for :mod:`cardiorisk.explainability.subgroup_drift`."""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.explainability.subgroup_drift import (
    DEFAULT_MIN_STRATUM_SIZE,
    SubgroupDriftResult,
    compute_subgroup_drift,
)


def test_skips_low_n_strata_below_guard() -> None:
    """LongBeachVA F=6 case: stratum below guard is skipped, recorded."""
    rng = np.random.default_rng(0)
    n = 50
    shap = rng.normal(size=(n, 3))
    grouping = np.array(["M"] * 44 + ["F"] * 6, dtype=object)
    out = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=grouping,
        shap_values_raw=shap,
        raw_feature_names=("a", "b", "c"),
        min_stratum_size=DEFAULT_MIN_STRATUM_SIZE,
    )
    assert isinstance(out, SubgroupDriftResult)
    audited = {s.stratum for s in out.by_stratum}
    skipped = {s for s, _ in out.skipped_strata}
    assert "M" in audited
    assert "F" in skipped


def test_emits_per_stratum_means() -> None:
    rng = np.random.default_rng(0)
    n = 100
    shap = rng.normal(size=(n, 3))
    grouping = np.array(["M"] * 60 + ["F"] * 40, dtype=object)
    out = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=grouping,
        shap_values_raw=shap,
        raw_feature_names=("a", "b", "c"),
        min_stratum_size=30,
    )
    audited = {s.stratum: s for s in out.by_stratum}
    assert set(audited.keys()) == {"M", "F"}
    assert audited["M"].n == 60
    assert audited["F"].n == 40
    for s in audited.values():
        assert set(s.mean_abs_per_feature.keys()) == {"a", "b", "c"}


def test_overall_mean_uses_all_rows() -> None:
    n = 50
    shap = np.ones((n, 2)) * 2.0  # mean |SHAP| should be 2.0 for both features
    grouping = np.array(["M"] * 50, dtype=object)
    out = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=grouping,
        shap_values_raw=shap,
        raw_feature_names=("a", "b"),
        min_stratum_size=10,
    )
    assert out.overall_mean_abs_per_feature == {"a": 2.0, "b": 2.0}


def test_delta_is_stratum_mean_minus_overall_mean() -> None:
    """If a stratum's |SHAP| is uniformly larger than overall, delta is positive."""
    n = 60
    # First 30 rows (stratum A) have |SHAP|=4, last 30 (stratum B) have |SHAP|=0.
    shap = np.zeros((n, 1))
    shap[:30, 0] = 4.0
    grouping = np.array(["A"] * 30 + ["B"] * 30, dtype=object)
    out = compute_subgroup_drift(
        grouping_name="x",
        grouping_values=grouping,
        shap_values_raw=shap,
        raw_feature_names=("only_feature",),
        min_stratum_size=10,
    )
    by = {s.stratum: s for s in out.by_stratum}
    # overall mean |SHAP| = (30*4 + 30*0) / 60 = 2.0
    assert by["A"].mean_abs_per_feature["only_feature"] == 4.0
    assert by["A"].delta_per_feature["only_feature"] == pytest.approx(2.0)
    assert by["B"].delta_per_feature["only_feature"] == pytest.approx(-2.0)


def test_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="rows"):
        compute_subgroup_drift(
            grouping_name="sex",
            grouping_values=np.array(["M", "F"], dtype=object),
            shap_values_raw=np.zeros((3, 2)),
            raw_feature_names=("a", "b"),
        )


def test_rejects_feature_count_mismatch() -> None:
    with pytest.raises(ValueError, match="cols"):
        compute_subgroup_drift(
            grouping_name="sex",
            grouping_values=np.array(["M"] * 4, dtype=object),
            shap_values_raw=np.zeros((4, 3)),
            raw_feature_names=("a", "b"),
        )


def test_strata_iteration_is_lexicographically_stable() -> None:
    """Per-fold JSONs need stable stratum order across runs."""
    n = 100
    grouping = np.array(["b"] * 50 + ["a"] * 50, dtype=object)
    shap = np.random.default_rng(0).normal(size=(n, 1))
    out = compute_subgroup_drift(
        grouping_name="x",
        grouping_values=grouping,
        shap_values_raw=shap,
        raw_feature_names=("only",),
        min_stratum_size=10,
    )
    assert tuple(s.stratum for s in out.by_stratum) == ("a", "b")
