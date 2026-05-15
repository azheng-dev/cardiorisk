"""Matplotlib renderers for the Phase-4 agent eval dashboard.

Three figures, mirroring the Phase 3.3 figure pattern:

- ``per_stage_pass_rate.png`` — bar chart of pass rate per stage.
- ``risk_band_confusion.png`` — 3x3 expected-vs-observed confusion.
- ``per_tag_pass_rate.png`` — stacked bar of pass rate per stage,
  grouped by tag.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .scorer import BANDS, AggregateReport


def render_per_stage_pass_rate(report: AggregateReport, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    stages = ("triage", "risk_band_match", "guideline", "letter")
    rates = (
        report.triage_pass_rate,
        report.risk_band_match_rate,
        report.guideline_pass_rate,
        report.letter_pass_rate,
    )
    colors = ("#2563eb", "#0ea5e9", "#10b981", "#f59e0b")
    bars = ax.bar(stages, rates, color=colors, edgecolor="#1f2937")
    for b, v in zip(bars, rates, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.01,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title(f"Phase 4 agent-eval per-stage pass rate (n={report.n_cases})")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_risk_band_confusion(report: AggregateReport, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    matrix = np.array(
        [[report.confusion_matrix.get(e, {}).get(o, 0) for o in BANDS] for e in BANDS],
        dtype=int,
    )
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(BANDS)), BANDS)
    ax.set_yticks(range(len(BANDS)), BANDS)
    ax.set_xlabel("observed risk band")
    ax.set_ylabel("expected risk band")
    ax.set_title("Phase 4 agent-eval risk-band confusion")
    for i in range(len(BANDS)):
        for j in range(len(BANDS)):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                fontsize=11,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_per_tag_pass_rate(report: AggregateReport, out_path: Path) -> Path:
    tags = list(report.per_tag.keys())
    if not tags:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no per-tag data", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    metrics = (
        "triage_pass_rate",
        "risk_band_match_rate",
        "guideline_pass_rate",
        "letter_pass_rate",
    )
    values = np.array([[float(report.per_tag[t][m]) for t in tags] for m in metrics])
    fig, ax = plt.subplots(figsize=(max(6, 1.0 + 1.6 * len(tags)), 4.5))
    x = np.arange(len(tags))
    width = 0.2
    colors = ("#2563eb", "#0ea5e9", "#10b981", "#f59e0b")
    for i, (m, c) in enumerate(zip(metrics, colors, strict=True)):
        ax.bar(x + (i - 1.5) * width, values[i], width, label=m, color=c, edgecolor="#1f2937")
    ax.set_xticks(x, tags, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("Phase 4 agent-eval per-tag pass rates")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_all(report: AggregateReport, out_dir: Path) -> dict[str, Path]:
    """Render all three figures into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "per_stage_pass_rate": render_per_stage_pass_rate(
            report, out_dir / "per_stage_pass_rate.png"
        ),
        "risk_band_confusion": render_risk_band_confusion(
            report, out_dir / "risk_band_confusion.png"
        ),
        "per_tag_pass_rate": render_per_tag_pass_rate(report, out_dir / "per_tag_pass_rate.png"),
    }


__all__ = [
    "render_all",
    "render_per_stage_pass_rate",
    "render_per_tag_pass_rate",
    "render_risk_band_confusion",
]
