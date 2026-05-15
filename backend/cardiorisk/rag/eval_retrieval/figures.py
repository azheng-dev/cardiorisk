"""Figure renderers for the Phase 3.2 retrieval eval.

Three matplotlib outputs (per ADR-016 §7):

- ``hit_at_5_by_cell.png`` — bar chart with bootstrap-CI error bars
  across all 6 cells.
- ``mrr_by_cell.png`` — analogous bar chart for MRR.
- ``per_tag_winning_cell.png`` — per-tag bars for the winning cell.

All figures use matplotlib defaults. The renderers accept a list of
``CellResult`` (one per cell) and write a single PNG. The
orchestrator caller closes figures after writing to keep memory
under control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from .scorer import EvalReport


@dataclass(frozen=True)
class CellResult:
    """One eval-matrix cell's results, as the orchestrator passes them in."""

    cell_label: str
    chunker: str
    with_rerank: bool
    report: EvalReport


def _cell_label(cell: CellResult) -> str:
    rerank = "rerank" if cell.with_rerank else "no-rerank"
    return f"{cell.chunker}\n{rerank}"


def render_hit_at_5_by_cell(
    cells: list[CellResult],
    *,
    out_path: Path,
    title: str = "Retrieval hit@5 by cell (95% bootstrap CI)",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [_cell_label(c) for c in cells]
    points = [c.report.hit_at_5 for c in cells]
    err_lo = [c.report.hit_at_5 - c.report.ci_hit_at_5.lower for c in cells]
    err_hi = [c.report.ci_hit_at_5.upper - c.report.hit_at_5 for c in cells]
    bars = ax.bar(range(len(cells)), points, yerr=[err_lo, err_hi], capsize=5)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("hit@5")
    ax.set_title(title)
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.97, points[i] + 0.02),
            f"{points[i]:.2f}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_mrr_by_cell(
    cells: list[CellResult],
    *,
    out_path: Path,
    title: str = "Retrieval MRR by cell (95% bootstrap CI)",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [_cell_label(c) for c in cells]
    points = [c.report.mrr for c in cells]
    err_lo = [c.report.mrr - c.report.ci_mrr.lower for c in cells]
    err_hi = [c.report.ci_mrr.upper - c.report.mrr for c in cells]
    ax.bar(range(len(cells)), points, yerr=[err_lo, err_hi], capsize=5)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("MRR")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_per_tag_for_winning_cell(
    winning_cell: CellResult,
    *,
    out_path: Path,
) -> None:
    per_tag = winning_cell.report.per_tag
    if not per_tag:
        return
    tags_sorted = sorted(per_tag)
    points = [per_tag[t]["hit_at_5"] for t in tags_sorted]
    counts = [int(per_tag[t]["n"]) for t in tags_sorted]
    labels = [f"{t}\n(n={c})" for t, c in zip(tags_sorted, counts, strict=True)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(tags_sorted)), points)
    ax.set_xticks(range(len(tags_sorted)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("hit@5")
    ax.set_title(
        f"Per-tag hit@5 — winning cell: {winning_cell.chunker} "
        f"({'with' if winning_cell.with_rerank else 'no'} rerank)"
    )
    for i, p in enumerate(points):
        ax.text(i, min(0.97, p + 0.02), f"{p:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
