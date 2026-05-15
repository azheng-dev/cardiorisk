"""Figure renderers for the Phase 3.3 generation eval.

Two matplotlib outputs (per ADR-017 §"Reporting"):

- ``citation_precision_by_tag.png`` — bar chart of citation
  precision per clinical tag.
- ``hallucination_rate_by_tag.png`` — bar chart of hallucination
  rate per clinical tag (lower is better; bars use a different
  colour to make that obvious in a scan).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from .scorer import EvalReport


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def render_citation_precision_by_tag(
    report: EvalReport,
    *,
    out_path: Path,
    title: str = "Citation precision by tag",
) -> None:
    per_tag = report.per_tag
    tags_sorted = [t for t in sorted(per_tag) if _is_finite(per_tag[t]["citation_precision"])]
    if not tags_sorted:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    points = [per_tag[t]["citation_precision"] for t in tags_sorted]
    counts = [int(per_tag[t]["n"]) for t in tags_sorted]
    labels = [f"{t}\n(n={c})" for t, c in zip(tags_sorted, counts, strict=True)]
    ax.bar(range(len(tags_sorted)), points, color="#3a7d44")
    ax.set_xticks(range(len(tags_sorted)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Citation precision")
    ax.set_title(title)
    for i, p in enumerate(points):
        ax.text(i, min(1.02, p + 0.02), f"{p:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_hallucination_rate_by_tag(
    report: EvalReport,
    *,
    out_path: Path,
    title: str = "Hallucination rate by tag (lower is better)",
) -> None:
    per_tag = report.per_tag
    tags_sorted = [t for t in sorted(per_tag) if _is_finite(per_tag[t]["hallucination_rate"])]
    if not tags_sorted:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    points = [per_tag[t]["hallucination_rate"] for t in tags_sorted]
    counts = [int(per_tag[t]["n"]) for t in tags_sorted]
    labels = [f"{t}\n(n={c})" for t, c in zip(tags_sorted, counts, strict=True)]
    ax.bar(range(len(tags_sorted)), points, color="#b3372b")
    ax.set_xticks(range(len(tags_sorted)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, max(0.05, max(points) + 0.05))
    ax.set_ylabel("Hallucination rate")
    ax.set_title(title)
    for i, p in enumerate(points):
        ax.text(i, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
