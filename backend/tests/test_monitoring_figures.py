"""Tests for cardiorisk.monitoring.figures.

Covers:

- ``render_drift_dashboard`` returns a matplotlib Figure.
- The figure has the expected three-axes layout (top wide bar + two
  bottom axes).
- The figure saves to disk as a non-empty PNG.
- The "no model passed" branch produces a placeholder bottom-right axis
  rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cardiorisk.monitoring.drift import compute_drift
from cardiorisk.monitoring.figures import render_drift_dashboard
from cardiorisk.monitoring.reference import build_fold_reference

SEED = 20260506


@pytest.fixture
def fitted_reference_and_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    n = 400
    ref = pd.DataFrame(
        {
            "Age": rng.normal(55, 10, n),
            "RestingBP": rng.normal(130, 20, n),
            "MaxHR": rng.normal(140, 25, n),
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], size=n),
        }
    )
    cur = ref.copy()
    cur["MaxHR"] = cur["MaxHR"] - 30.0  # clear shift on one feature
    return ref, cur


class _StubModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        p = rng.uniform(0.1, 0.9, size=len(X))
        return np.column_stack([1.0 - p, p])


def test_render_returns_figure(
    fitted_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = fitted_reference_and_current
    model = _StubModel()
    reference = build_fold_reference(held_out_source="x", X_train=ref_df, models={"stub": model})
    report = compute_drift(reference=reference, X_current=cur_df, model=model, model_name="stub")
    fig = render_drift_dashboard(
        report=report,
        reference=reference,
        current_numeric={
            c: cur_df[c].to_numpy(dtype=np.float64) for c in ("Age", "RestingBP", "MaxHR")
        },
        current_proba=np.asarray(model.predict_proba(cur_df))[:, 1],
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_render_dashboard_has_three_axes(
    fitted_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = fitted_reference_and_current
    model = _StubModel()
    reference = build_fold_reference(held_out_source="x", X_train=ref_df, models={"stub": model})
    report = compute_drift(reference=reference, X_current=cur_df, model=model, model_name="stub")
    fig = render_drift_dashboard(
        report=report,
        reference=reference,
        current_numeric={
            c: cur_df[c].to_numpy(dtype=np.float64) for c in ("Age", "RestingBP", "MaxHR")
        },
        current_proba=np.asarray(model.predict_proba(cur_df))[:, 1],
    )
    assert len(fig.axes) == 3
    plt.close(fig)


def test_render_writes_non_empty_png_to_disk(
    fitted_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    ref_df, cur_df = fitted_reference_and_current
    model = _StubModel()
    reference = build_fold_reference(held_out_source="x", X_train=ref_df, models={"stub": model})
    report = compute_drift(reference=reference, X_current=cur_df, model=model, model_name="stub")
    fig = render_drift_dashboard(
        report=report,
        reference=reference,
        current_numeric={
            c: cur_df[c].to_numpy(dtype=np.float64) for c in ("Age", "RestingBP", "MaxHR")
        },
        current_proba=np.asarray(model.predict_proba(cur_df))[:, 1],
    )
    out = tmp_path / "dashboard.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 5_000  # a real PNG is at least a few KB


def test_render_no_model_branch_does_not_crash(
    fitted_reference_and_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref_df, cur_df = fitted_reference_and_current
    reference = build_fold_reference(held_out_source="x", X_train=ref_df)
    report = compute_drift(reference=reference, X_current=cur_df)
    fig = render_drift_dashboard(
        report=report,
        reference=reference,
        current_numeric={
            c: cur_df[c].to_numpy(dtype=np.float64) for c in ("Age", "RestingBP", "MaxHR")
        },
        current_proba=None,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
