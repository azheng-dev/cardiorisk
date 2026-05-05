"""End-to-end smoke test for the Phase 2.3b + 2.4 training driver.

Drives ``backend/scripts/train_v1.py`` in ``--smoke`` mode against the
deterministic synthetic dataset that smoke mode generates. Verifies:

- All ``len(MODEL_NAMES)`` models (Phase 2.3b: tabicl/xgboost/lr;
  Phase 2.4 adds ensemble) complete fit -> calibrate -> predict ->
  persist.
- Per-fold JSON has the documented schema.
- Aggregate JSON has the documented schema and includes the
  ``n_ensemble_epochs`` Phase-2.4 config knob.
- Bootstrap CIs land for every metric we bootstrap.
- All artefacts land in ``models/v1/smoke/``.
- All figures land in ``reports/v1/figures/smoke/``.
- Output JSONs are strict-JSON parseable (no NaN literals).

This test runs the *real* driver, not a mock — failure here means the
end-to-end pipeline is actually broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from cardiorisk.data.paths import FIXTURE_PATH
from cardiorisk.models.base import MODEL_NAMES
from cardiorisk.models.ensemble import SMOKE_N_EPOCHS as ENSEMBLE_SMOKE_N_EPOCHS


@pytest.fixture
def driver_outputs(
    tmp_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], Path, Path]:
    """Run the driver in --smoke mode and return (aggregate, per_fold, models_dir, figures_dir)."""
    from cardiorisk.training import train_v1

    cfg = train_v1.RunConfig(
        smoke=True,
        n_trials=train_v1.SMOKE_N_TRIALS,
        n_resamples=train_v1.SMOKE_N_RESAMPLES,
        n_ensemble_epochs=ENSEMBLE_SMOKE_N_EPOCHS,
        # Smoke path: passing FIXTURE_PATH triggers the synthetic-on-the-fly
        # branch; the file itself is never read in smoke mode.
        data_path=FIXTURE_PATH,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        seed=20260505,
        n_folds_cap=1,
    )
    train_v1.run(cfg)
    aggregate = json.loads((cfg.reports_dir / "metrics_aggregate.json").read_text())
    per_fold = json.loads((cfg.reports_dir / "metrics_per_fold.json").read_text())
    return aggregate, per_fold, cfg.models_dir, cfg.figures_dir


def test_per_fold_has_one_block_per_model(driver_outputs):
    _, per_fold, _, _ = driver_outputs
    assert len(per_fold) == len(MODEL_NAMES)
    assert {b["model"] for b in per_fold} == set(MODEL_NAMES)


def test_per_fold_block_schema(driver_outputs):
    _, per_fold, _, _ = driver_outputs
    required_keys = {
        "model",
        "held_out_source",
        "n_test",
        "prevalence",
        "fit_seconds",
        "headline",
        "headline_ci",
        "subgroup_auroc_by_sex",
        "subgroup_auroc_by_age_band",
        "dca",
    }
    for block in per_fold:
        assert required_keys.issubset(block.keys()), f"missing keys in {block['model']}"
        for metric in (
            "auroc",
            "auprc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "sensitivity_at_85_spec",
            "sensitivity_at_90_spec",
        ):
            assert metric in block["headline"]


def test_bootstrap_cis_present(driver_outputs):
    _, per_fold, _, _ = driver_outputs
    expected_ci_metrics = {
        "auroc",
        "auprc",
        "brier",
        "sensitivity_at_85_spec",
        "sensitivity_at_90_spec",
    }
    for block in per_fold:
        cis = block["headline_ci"]
        assert expected_ci_metrics.issubset(cis.keys())
        for ci in cis.values():
            assert {"point", "lower", "upper", "n_resamples", "alpha"}.issubset(ci.keys())
            # Smoke mode runs at SMOKE_N_RESAMPLES.
            assert ci["n_resamples"] == 100


def test_dca_at_auscvdrisk_thresholds(driver_outputs):
    _, per_fold, _, _ = driver_outputs
    for block in per_fold:
        assert "5pct" in block["dca"]
        assert "10pct" in block["dca"]
        for at in block["dca"].values():
            assert {"model", "treat_all", "treat_none"}.issubset(at.keys())


def test_aggregate_schema(driver_outputs):
    aggregate, _, _, _ = driver_outputs
    assert "config" in aggregate
    assert "by_model" in aggregate
    assert set(aggregate["by_model"].keys()) == set(MODEL_NAMES)
    for _model_name, rows in aggregate["by_model"].items():
        for _metric, stats in rows.items():
            assert {"mean", "std", "n_folds"}.issubset(stats.keys())


def test_artefacts_persisted(driver_outputs):
    _, _, models_dir, _ = driver_outputs
    artefacts = sorted(models_dir.glob("*.joblib"))
    # 3 models x 1 LODO fold (smoke n_folds_cap=1) = 3 artefacts.
    assert len(artefacts) == len(MODEL_NAMES)
    # Each loads as something with predict_proba.
    for path in artefacts:
        clf = joblib.load(path)
        assert hasattr(clf, "predict_proba")


def test_figures_saved(driver_outputs):
    _, _, _, figures_dir = driver_outputs
    figures = sorted(figures_dir.glob("*.png"))
    # 3 models x 1 fold x 2 figure types (reliability + dca) = 6 PNGs.
    assert len(figures) == len(MODEL_NAMES) * 2
    for fig_path in figures:
        # Each PNG should be at least a few KB (real plot, not empty file).
        assert fig_path.stat().st_size > 1_000


def test_strict_json_no_nan_literal(driver_outputs):
    """Aggregate + per-fold JSON must parse with allow_nan=False."""
    aggregate, per_fold, _, _ = driver_outputs
    json.dumps(aggregate, allow_nan=False)
    json.dumps(per_fold, allow_nan=False)


# ----------------------------------------------------- Phase 2.4 specifics


def test_ensemble_row_present_per_fold(driver_outputs):
    """Phase 2.4: the Honours-Ensemble row must land in the per-fold JSON."""
    _, per_fold, _, _ = driver_outputs
    ensemble_blocks = [b for b in per_fold if b["model"] == "ensemble"]
    assert len(ensemble_blocks) == 1
    block = ensemble_blocks[0]
    # Same headline-metric surface as the other models.
    for metric in ("auroc", "auprc", "brier"):
        assert metric in block["headline"]


def test_ensemble_row_present_in_aggregate(driver_outputs):
    """Phase 2.4: the Honours-Ensemble row must appear in by_model aggregates."""
    aggregate, _, _, _ = driver_outputs
    assert "ensemble" in aggregate["by_model"]
    rows = aggregate["by_model"]["ensemble"]
    for metric in ("auroc", "auprc", "brier"):
        assert metric in rows
        assert {"mean", "std", "n_folds"}.issubset(rows[metric].keys())


def test_aggregate_records_n_ensemble_epochs(driver_outputs):
    """Phase 2.4: the smoke-mode ensemble epoch budget is recorded in config."""
    aggregate, _, _, _ = driver_outputs
    assert aggregate["config"]["n_ensemble_epochs"] == 1


def test_ensemble_artefact_persisted(driver_outputs):
    """Phase 2.4: a calibrated ensemble joblib lands alongside the v1 trio."""
    _, _, models_dir, _ = driver_outputs
    ensemble_paths = list(models_dir.glob("ensemble_*.joblib"))
    assert len(ensemble_paths) == 1
    clf = joblib.load(ensemble_paths[0])
    # Calibration applied (sigmoid/Platt per ADR-012).
    assert hasattr(clf, "predict_proba")
