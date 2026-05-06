"""Single-PNG-per-cell drift dashboard.

One :func:`render_drift_dashboard` call produces one self-contained
matplotlib :class:`~matplotlib.figure.Figure` for one (model, fold) cell:

- **Top:** PSI bar across every feature, sorted descending. Bars are
  coloured by severity band (stable / moderate / major) using the
  ADR-014 cut-points. Reference horizontal lines mark the band
  boundaries so the eye can place each bar without consulting a legend.
- **Bottom-left:** ECDF overlay (reference vs current) for the top-3
  numeric drifted features. Picks numerics only — categorical features
  don't have an ordered CDF.
- **Bottom-right:** Predict-proba histogram overlay (reference vs
  current bin counts) when prediction-drift was computed; otherwise an
  explanatory placeholder.

Headless-safe (Agg backend); the orchestrator registers ``matplotlib.use("Agg")``
before importing this module, just like the explainability layer does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cardiorisk.monitoring.drift import DriftReport, FeatureDrift
from cardiorisk.monitoring.psi import PSI_MODERATE_MAX, PSI_STABLE_MAX
from cardiorisk.monitoring.reference import FoldReference, NumericReference

#: Severity-band colours. Chosen to be matplotlib-default-friendly and
#: distinguishable in the dashboard: green-ish / amber / red.
_SEVERITY_COLORS: Final[Mapping[str, str]] = {
    "stable": "#4daf4a",
    "moderate": "#ff7f00",
    "major": "#e41a1c",
}


def _ecdf(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sorted-x + step-CDF y for a 1-D sample. Empty input returns empty arrays."""
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(clean)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
    return xs, ys


def _reconstruct_reference_samples(num_ref: NumericReference) -> npt.NDArray[np.float64]:
    """Reconstruct synthetic reference samples from bin midpoints + counts.

    Same trick as :mod:`cardiorisk.monitoring.drift` uses for KS:
    each bin contributes ``count`` copies of its midpoint. The ECDF
    overlay this produces is a faithful step-function approximation
    of the true reference CDF given the same quantile binning that was
    applied at reference-build time.
    """
    edges = num_ref.edges
    finite = np.isfinite(edges)
    inner = edges[finite]
    if inner.size == 0:
        return np.array([], dtype=np.float64)
    span = max(float(inner[-1] - inner[0]), 1.0)
    extended = edges.copy()
    if not np.isfinite(extended[0]):
        extended[0] = float(inner[0]) - span
    if not np.isfinite(extended[-1]):
        extended[-1] = float(inner[-1]) + span
    midpoints = (extended[:-1] + extended[1:]) / 2.0
    return np.repeat(midpoints, num_ref.counts).astype(np.float64)


def render_drift_dashboard(
    *,
    report: DriftReport,
    reference: FoldReference,
    current_numeric: Mapping[str, npt.NDArray[np.float64]],
    current_proba: npt.NDArray[np.float64] | None = None,
    title: str | None = None,
) -> Figure:
    """One dashboard PNG for one (model, fold) cell.

    Parameters
    ----------
    report : DriftReport
        Output of :func:`cardiorisk.monitoring.drift.compute_drift`.
    reference : FoldReference
        The same reference used to compute ``report``. Needed for the
        ECDF overlay (we reconstruct synthetic reference samples from
        the persisted bin midpoints + counts).
    current_numeric : mapping of feature -> 1-D numeric current values
        Raw current-slice values for the numeric features. Only the
        top-3 drifted numerics are actually plotted; the orchestrator
        passes the full mapping so this function can pick.
    current_proba : optional 1-D array
        Calibrated ``predict_proba(X_current)[:, 1]``. Required for the
        bottom-right histogram overlay; if ``None`` (no model passed),
        a placeholder text axis is drawn instead.
    title : optional str
        Figure-level title. Defaults to ``"<source> | <model> | drift dashboard"``.
    """
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    _draw_psi_bar(ax=ax_top, drifts=list(report.per_feature.values()))
    _draw_ecdf_overlays(
        ax=ax_bl,
        report=report,
        reference=reference,
        current_numeric=current_numeric,
    )
    _draw_proba_overlay(
        ax=ax_br,
        report=report,
        reference=reference,
        current_proba=current_proba,
    )

    fig.suptitle(
        title or f"{report.held_out_source} | {report.model_name} | drift dashboard",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def _draw_psi_bar(*, ax: Axes, drifts: list[FeatureDrift]) -> None:
    drifts_sorted = sorted(drifts, key=lambda fd: (-fd.psi, fd.feature))
    names = [fd.feature for fd in drifts_sorted]
    psis = [fd.psi for fd in drifts_sorted]
    colors = [_SEVERITY_COLORS[fd.severity] for fd in drifts_sorted]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, psis, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, names)
    ax.invert_yaxis()
    ax.axvline(PSI_STABLE_MAX, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(PSI_MODERATE_MAX, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("PSI (higher = more drift)")
    ax.set_title(
        "Per-feature PSI vs reference (severity bands: <0.10 stable / 0.10-0.25 moderate / >=0.25 major)"
    )
    # Right-pad so the rightmost bar's value text doesn't get clipped.
    if psis:
        ax.set_xlim(0.0, max(max(psis), PSI_MODERATE_MAX) * 1.15)
    ax.grid(axis="x", alpha=0.3)


def _draw_ecdf_overlays(
    *,
    ax: Axes,
    report: DriftReport,
    reference: FoldReference,
    current_numeric: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """Top-3 numeric drifted features as ECDF overlays. Categoricals are skipped."""
    numeric_drifts = [fd for fd in report.per_feature.values() if fd.kind == "numeric"]
    numeric_drifts.sort(key=lambda fd: (-fd.psi, fd.feature))
    top = numeric_drifts[:3]

    if not top:
        ax.text(
            0.5,
            0.5,
            "No numeric features to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    ref_color = "#377eb8"
    cur_color = "#e41a1c"
    for fd in top:
        if fd.feature not in reference.numeric or fd.feature not in current_numeric:
            continue
        ref_samples = _reconstruct_reference_samples(reference.numeric[fd.feature])
        cur_samples = current_numeric[fd.feature]
        ref_x, ref_y = _ecdf(ref_samples)
        cur_x, cur_y = _ecdf(cur_samples)
        if ref_x.size:
            ax.step(
                ref_x,
                ref_y,
                where="post",
                color=ref_color,
                alpha=0.45,
                linewidth=1.2,
            )
        if cur_x.size:
            ax.step(
                cur_x,
                cur_y,
                where="post",
                color=cur_color,
                alpha=0.65,
                linewidth=1.2,
                label=f"{fd.feature} (PSI={fd.psi:.2f})",
            )

    # Build a shared legend: ref vs cur lines are duplicated per feature
    # in the loop above; only keep one of each plus the per-feature labels.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=8)
    ax.set_xlabel("feature value")
    ax.set_ylabel("cumulative probability")
    ax.set_title("ECDF: reference (blue) vs current (red), top-3 drifted numerics")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)


def _draw_proba_overlay(
    *,
    ax: Axes,
    report: DriftReport,
    reference: FoldReference,
    current_proba: npt.NDArray[np.float64] | None,
) -> None:
    """Predict-proba histogram overlay; placeholder text if no model was passed."""
    if report.prediction is None or current_proba is None:
        ax.text(
            0.5,
            0.5,
            "Prediction drift skipped (no model passed to compute_drift)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    pred_ref = reference.prediction.get(report.prediction.model_name)
    if pred_ref is None:
        ax.text(
            0.5,
            0.5,
            "Reference has no prediction binning",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    edges_for_plot = pred_ref.edges.copy()
    if not np.isfinite(edges_for_plot[0]):
        edges_for_plot[0] = 0.0
    if not np.isfinite(edges_for_plot[-1]):
        edges_for_plot[-1] = 1.0

    # Histogram bin counts as proportions so the ref + cur curves are
    # commensurate even when n_ref != n_cur (and they always differ).
    ref_props = pred_ref.counts.astype(np.float64) / max(pred_ref.counts.sum(), 1)
    cur_proba_clean = current_proba[~np.isnan(current_proba)]
    cur_counts = np.histogram(np.clip(cur_proba_clean, 0.0, 1.0), bins=edges_for_plot)[0]
    cur_props = cur_counts.astype(np.float64) / max(cur_counts.sum(), 1)

    centers = (edges_for_plot[:-1] + edges_for_plot[1:]) / 2.0
    width = float(np.diff(edges_for_plot).min()) * 0.45
    ax.bar(
        centers - width / 2.0,
        ref_props,
        width=width,
        label="reference",
        color="#377eb8",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        centers + width / 2.0,
        cur_props,
        width=width,
        label="current",
        color="#e41a1c",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("proportion of rows")
    pd_block = report.prediction
    ax.set_title(
        f"Prediction drift: PSI={pd_block.psi:.3f} ({pd_block.severity}) | "
        f"mean_ref={pd_block.mean_ref:.3f} -> mean_cur={pd_block.mean_cur:.3f}"
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
