"""Matplotlib renderers for the Phase 2.5 explainability deliverables.

ADR-013 §"Output surface" enumerates the figure inventory:

- Per (model x fold): global bar + global beeswarm of mean |SHAP|.
- Per (model x fold x archetype): waterfall plot.
- Per fold: cross-model agreement heatmap + aggregate-across-folds heatmap.
- Per (model x fold x stratum): subgroup-drift bar (auditable strata only).
- Per fold: TreeSHAP-vs-KernelSHAP scatter (XGBoost sanity).
- Per fold: LR per-spline-basis vs summed-back bar (LR detail).

Total: ~140 PNGs for the full LODO sweep. Smoke run produces a
small subset under ``reports/v1/figures/explainability/smoke/``
(gitignored per the same convention as Phase 2.3b figure output).

Renderers return :class:`matplotlib.figure.Figure` handles. The
caller (the orchestrator) is responsible for ``fig.savefig(...)`` and
``plt.close(fig)``. This mirrors the
:func:`cardiorisk.eval.reliability.reliability_diagram` convention
the eval harness uses.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from cardiorisk.explainability.archetypes import Archetype
from cardiorisk.explainability.cross_model_agreement import AgreementResult
from cardiorisk.explainability.subgroup_drift import SubgroupDriftResult

#: Default top-K for bar charts (most-important features only).
DEFAULT_TOP_K: Final[int] = 12

#: Colormap used by the cross-model agreement heatmap. Diverging
#: because Spearman ranges in [-1, 1].
HEATMAP_CMAP: Final[str] = "RdBu_r"


def global_importance_bar(
    *,
    mean_abs_per_feature: dict[str, float],
    title: str,
    top_k: int = DEFAULT_TOP_K,
) -> Figure:
    """Horizontal bar chart: top-K features by mean |SHAP|."""
    items = sorted(mean_abs_per_feature.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [k for k, _ in reversed(items)]
    values = [v for _, v in reversed(items)]

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.5))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def global_importance_beeswarm(
    *,
    shap_values_raw: npt.NDArray[np.float64],
    raw_feature_names: tuple[str, ...],
    title: str,
    top_k: int = DEFAULT_TOP_K,
) -> Figure:
    """Beeswarm-style scatter: per-row SHAP distribution per feature.

    A simple matplotlib analogue of ``shap.summary_plot(plot_type="dot")``
    that doesn't require shap's internal plotting machinery (which has a
    history of breaking on non-default Agg backends and emitting its own
    warnings).
    """
    if shap_values_raw.shape[1] != len(raw_feature_names):
        raise ValueError(
            f"shap_values_raw cols={shap_values_raw.shape[1]} vs "
            f"{len(raw_feature_names)} feature names"
        )

    means = np.mean(np.abs(shap_values_raw), axis=0)
    order = np.argsort(means)[-top_k:]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    for plot_row, j in enumerate(order):
        vals = shap_values_raw[:, j]
        jitter = rng.uniform(-0.18, 0.18, size=vals.shape)
        ax.scatter(vals, np.full_like(vals, plot_row, dtype=float) + jitter, s=6, alpha=0.55)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([raw_feature_names[j] for j in order])
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on probability)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def waterfall(
    *,
    shap_row: npt.NDArray[np.float64],
    raw_feature_names: tuple[str, ...],
    expected_value: float,
    archetype: Archetype,
    top_k: int = DEFAULT_TOP_K,
) -> Figure:
    """Waterfall plot of one row's SHAP contributions.

    Bars are ordered by |SHAP| descending and clipped to ``top_k``;
    contributions outside the top-K are aggregated into an "other"
    bucket so the visual remains additive.
    """
    if shap_row.shape != (len(raw_feature_names),):
        raise ValueError(
            f"shap_row shape {shap_row.shape} mismatched against "
            f"{len(raw_feature_names)} feature names"
        )

    abs_vals = np.abs(shap_row)
    order = np.argsort(abs_vals)[::-1]
    top = order[:top_k]
    rest = order[top_k:]

    rows = [(raw_feature_names[i], float(shap_row[i])) for i in top]
    if rest.size:
        rows.append(("(other)", float(shap_row[rest].sum())))

    cumulative = expected_value
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.8))

    y_positions = np.arange(len(rows))[::-1]
    for y, (_name, contrib) in zip(y_positions, rows, strict=True):
        color = "indianred" if contrib > 0 else "steelblue"
        ax.barh(y, contrib, left=cumulative, color=color, edgecolor="black", linewidth=0.4)
        cumulative += contrib

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{name}" for name, _ in rows])
    ax.axvline(expected_value, color="grey", linestyle="--", linewidth=0.8, label="E[f(x)]")
    ax.axvline(cumulative, color="black", linestyle="-", linewidth=0.9, label="f(x)")
    ax.set_xlabel("predicted P(CVD-positive) contribution")
    ax.set_title(
        f"{archetype.label}  (test_index={archetype.test_index}, "
        f"y_true={archetype.y_true}, p={archetype.y_proba:.3f})"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def cross_model_agreement_heatmap(
    *,
    agreement: AgreementResult,
    title: str,
) -> Figure:
    """Heatmap of the Spearman rank-correlation matrix."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(
        agreement.spearman_matrix,
        cmap=HEATMAP_CMAP,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )
    ax.set_xticks(range(len(agreement.model_names)))
    ax.set_yticks(range(len(agreement.model_names)))
    ax.set_xticklabels(agreement.model_names, rotation=30, ha="right")
    ax.set_yticklabels(agreement.model_names)

    # Annotate cells with the Spearman value.
    for i in range(len(agreement.model_names)):
        for j in range(len(agreement.model_names)):
            val = agreement.spearman_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:+.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.6 else "white",
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rank correlation")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def subgroup_drift_bar(
    *,
    drift: SubgroupDriftResult,
    title: str,
    top_k: int = DEFAULT_TOP_K,
) -> Figure:
    """Grouped horizontal bars: per-stratum |SHAP| delta vs overall.

    One bar per (feature, stratum) pair, sorted by |delta| of the
    largest stratum.
    """
    if not drift.by_stratum:
        # All strata below guard -- emit a tiny "no audit" placeholder
        # rather than crash the orchestrator.
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.text(
            0.5,
            0.5,
            f"All {drift.grouping_name} strata below min-n guard",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return fig

    feature_names = list(drift.overall_mean_abs_per_feature.keys())
    # Rank features by max |delta| across audited strata (descending).
    max_abs_delta = {
        f: max(abs(s.delta_per_feature[f]) for s in drift.by_stratum) for f in feature_names
    }
    feature_names = sorted(feature_names, key=lambda f: max_abs_delta[f], reverse=True)[:top_k]

    n_features = len(feature_names)
    n_strata = len(drift.by_stratum)
    bar_height = 0.8 / n_strata
    fig, ax = plt.subplots(figsize=(7, 0.4 * n_features + 2.0))
    y_base = np.arange(n_features)
    for k, s in enumerate(drift.by_stratum):
        offsets = (k - (n_strata - 1) / 2) * bar_height
        deltas = [s.delta_per_feature[f] for f in feature_names]
        ax.barh(
            y_base + offsets,
            deltas,
            height=bar_height,
            label=f"{s.stratum} (n={s.n})",
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(list(reversed(feature_names))[::-1])
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean |SHAP value| - overall mean")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def treeshap_vs_kernelshap_scatter(
    *,
    treeshap_per_raw: dict[str, float],
    kernelshap_per_raw: dict[str, float],
    title: str,
) -> Figure:
    """Scatter of mean |SHAP| per raw feature: TreeSHAP vs KernelSHAP.

    A close-to-y=x cloud means the two algorithms agree on per-
    feature importance. The orchestrator's JSON output additionally
    records the Spearman rank correlation between the two so the
    figure has a one-number quantitative summary.
    """
    common = sorted(set(treeshap_per_raw.keys()) & set(kernelshap_per_raw.keys()))
    if not common:
        raise ValueError("treeshap and kernelshap dicts share no feature names")

    tree_vals = np.array([treeshap_per_raw[k] for k in common])
    kernel_vals = np.array([kernelshap_per_raw[k] for k in common])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(tree_vals, kernel_vals, s=30, alpha=0.7, color="steelblue")
    for x, y, name in zip(tree_vals, kernel_vals, common, strict=True):
        ax.annotate(name, (x, y), fontsize=7, alpha=0.7)
    lim = max(float(tree_vals.max()), float(kernel_vals.max())) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("TreeSHAP mean |log-odds SHAP|")
    ax.set_ylabel("KernelSHAP mean |probability SHAP|")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def lr_summed_vs_basis_bar(
    *,
    summed_per_raw: dict[str, float],
    per_basis: dict[str, float],
    title: str,
    top_k: int = DEFAULT_TOP_K,
) -> Figure:
    """Two-panel bar: summed-back per-feature view (left) + per-basis view (right)."""
    # Use constrained_layout so the suptitle doesn't fight tight_layout.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 0.4 * top_k + 2.0),
        gridspec_kw={"wspace": 0.4},
        constrained_layout=True,
    )

    # Summed-back panel.
    summed = sorted(summed_per_raw.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    summed_names = [k for k, _ in reversed(summed)]
    summed_values = [v for _, v in reversed(summed)]
    axes[0].barh(summed_names, summed_values, color="steelblue")
    axes[0].set_xlabel("mean |log-odds SHAP|, summed back")
    axes[0].set_title("Summed-back per-feature")
    axes[0].grid(axis="x", alpha=0.3)

    # Per-basis panel.
    basis = sorted(per_basis.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    basis_names = [k for k, _ in reversed(basis)]
    basis_values = [v for _, v in reversed(basis)]
    axes[1].barh(basis_names, basis_values, color="indianred")
    axes[1].set_xlabel("mean |log-odds SHAP|, per spline-basis")
    axes[1].set_title("Per-basis (LR detail)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(title)
    return fig


def archetype_features_to_dataframe(
    *,
    archetype: Archetype,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience: extract the archetype's feature row as a 1-row DataFrame.

    Used by the orchestrator when serialising local explanations to JSON.
    """
    return X_test.iloc[[archetype.test_index]].reset_index(drop=True)
