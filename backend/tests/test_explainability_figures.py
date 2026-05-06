"""Smoke tests for :mod:`cardiorisk.explainability.figures`.

We only assert that each renderer returns a :class:`matplotlib.figure.Figure`
and produces non-trivial axes; pixel-level visual regression is out of
scope (and would be brittle across matplotlib minor versions).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cardiorisk.explainability.archetypes import Archetype
from cardiorisk.explainability.cross_model_agreement import (
    compute_cross_model_agreement,
)
from cardiorisk.explainability.figures import (
    archetype_features_to_dataframe,
    cross_model_agreement_heatmap,
    global_importance_bar,
    global_importance_beeswarm,
    lr_summed_vs_basis_bar,
    subgroup_drift_bar,
    treeshap_vs_kernelshap_scatter,
    waterfall,
)
from cardiorisk.explainability.subgroup_drift import (
    compute_subgroup_drift,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all open matplotlib figures after each test."""
    yield
    plt.close("all")


def test_global_importance_bar_returns_figure() -> None:
    fig = global_importance_bar(
        mean_abs_per_feature={"a": 0.3, "b": 0.5, "c": 0.1},
        title="test",
        top_k=2,
    )
    assert isinstance(fig, Figure)


def test_global_importance_beeswarm_returns_figure() -> None:
    rng = np.random.default_rng(0)
    fig = global_importance_beeswarm(
        shap_values_raw=rng.normal(size=(20, 4)),
        raw_feature_names=("a", "b", "c", "d"),
        title="test",
        top_k=3,
    )
    assert isinstance(fig, Figure)


def test_global_importance_beeswarm_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="cols"):
        global_importance_beeswarm(
            shap_values_raw=np.zeros((5, 3)),
            raw_feature_names=("a", "b"),
            title="test",
        )


def test_waterfall_returns_figure() -> None:
    arch = Archetype(label="tp_high", test_index=2, y_true=1, y_proba=0.85)
    fig = waterfall(
        shap_row=np.array([0.1, -0.2, 0.05, 0.3]),
        raw_feature_names=("a", "b", "c", "d"),
        expected_value=0.5,
        archetype=arch,
        top_k=4,
    )
    assert isinstance(fig, Figure)


def test_waterfall_rejects_shape_mismatch() -> None:
    arch = Archetype(label="tp_high", test_index=0, y_true=1, y_proba=0.6)
    with pytest.raises(ValueError, match="mismatched"):
        waterfall(
            shap_row=np.array([0.1, 0.2]),
            raw_feature_names=("a", "b", "c"),
            expected_value=0.5,
            archetype=arch,
        )


def test_cross_model_agreement_heatmap_returns_figure() -> None:
    agreement = compute_cross_model_agreement(
        mean_abs_per_model={
            "m1": {"a": 1.0, "b": 2.0, "c": 3.0},
            "m2": {"a": 3.0, "b": 2.0, "c": 1.0},
        }
    )
    fig = cross_model_agreement_heatmap(agreement=agreement, title="test")
    assert isinstance(fig, Figure)


def test_subgroup_drift_bar_returns_figure() -> None:
    rng = np.random.default_rng(0)
    grouping = np.array(["M"] * 50 + ["F"] * 40, dtype=object)
    drift = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=grouping,
        shap_values_raw=rng.normal(size=(90, 4)),
        raw_feature_names=("a", "b", "c", "d"),
        min_stratum_size=20,
    )
    fig = subgroup_drift_bar(drift=drift, title="test")
    assert isinstance(fig, Figure)


def test_subgroup_drift_bar_handles_all_skipped_strata() -> None:
    """When every stratum is below the guard, render a placeholder figure."""
    rng = np.random.default_rng(0)
    grouping = np.array(["A"] * 5, dtype=object)
    drift = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=grouping,
        shap_values_raw=rng.normal(size=(5, 3)),
        raw_feature_names=("a", "b", "c"),
        min_stratum_size=10,
    )
    assert drift.by_stratum == ()
    fig = subgroup_drift_bar(drift=drift, title="test")
    assert isinstance(fig, Figure)


def test_treeshap_vs_kernelshap_scatter_returns_figure() -> None:
    fig = treeshap_vs_kernelshap_scatter(
        treeshap_per_raw={"a": 0.5, "b": 0.3, "c": 0.1},
        kernelshap_per_raw={"a": 0.55, "b": 0.28, "c": 0.12},
        title="test",
    )
    assert isinstance(fig, Figure)


def test_treeshap_vs_kernelshap_scatter_rejects_no_overlap() -> None:
    with pytest.raises(ValueError, match="share no feature names"):
        treeshap_vs_kernelshap_scatter(
            treeshap_per_raw={"a": 0.5},
            kernelshap_per_raw={"b": 0.5},
            title="test",
        )


def test_lr_summed_vs_basis_bar_returns_figure() -> None:
    fig = lr_summed_vs_basis_bar(
        summed_per_raw={"Age": 0.5, "Sex": 0.3, "Cholesterol": 0.4},
        per_basis={
            "x0": 0.2,
            "x0_rcs1": 0.15,
            "x0_rcs2": 0.15,
            "Sex_M": 0.3,
            "Sex_F": 0.0,
        },
        title="test",
        top_k=3,
    )
    assert isinstance(fig, Figure)


def test_archetype_features_to_dataframe_returns_one_row() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    arch = Archetype(label="tp_high", test_index=1, y_true=1, y_proba=0.8)
    out = archetype_features_to_dataframe(archetype=arch, X_test=df)
    assert len(out) == 1
    assert out.iloc[0]["a"] == 2
    assert out.iloc[0]["b"] == "y"
