"""End-to-end smoke test for :mod:`cardiorisk.monitoring.orchestrator`.

Asserts the drift orchestrator produces the expected JSON files +
dashboard PNGs for a 1-fold smoke run, given pre-trained smoke models
under ``models/v1/smoke/``. We train a tiny smoke model set inside the
test fixture so the suite is self-contained (matches the explainability
orchestrator's test pattern).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cardiorisk.monitoring.orchestrator import (
    DriftConfig,
    full_config,
    run,
    smoke_config,
)
from cardiorisk.training.train_v1 import RunConfig as TrainRunConfig
from cardiorisk.training.train_v1 import run as train_run


@pytest.fixture(scope="module")
def driver_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train smoke models then run the drift orchestrator once."""
    from cardiorisk.data.paths import FIXTURE_PATH
    from cardiorisk.models.base import SEED
    from cardiorisk.models.ensemble import SMOKE_N_EPOCHS
    from cardiorisk.monitoring.reference import DEFAULT_N_BINS
    from cardiorisk.training.train_v1 import SMOKE_N_RESAMPLES

    tmp = tmp_path_factory.mktemp("p26-orch-smoke")
    models_dir = tmp / "models"
    train_reports = tmp / "train_reports"
    train_figures = tmp / "train_figures"
    drift_reports = tmp / "drift_reports"
    drift_figures = tmp / "drift_figures"
    for d in (models_dir, train_reports, train_figures, drift_reports, drift_figures):
        d.mkdir(parents=True, exist_ok=True)

    train_run(
        TrainRunConfig(
            smoke=True,
            n_trials=1,
            n_resamples=SMOKE_N_RESAMPLES,
            n_ensemble_epochs=SMOKE_N_EPOCHS,
            data_path=FIXTURE_PATH,
            models_dir=models_dir,
            reports_dir=train_reports,
            figures_dir=train_figures,
            seed=SEED,
            n_folds_cap=1,
        )
    )

    cfg = DriftConfig(
        smoke=True,
        n_bins=DEFAULT_N_BINS,
        data_path=FIXTURE_PATH,
        models_dir=models_dir,
        reports_dir=drift_reports,
        figures_dir=drift_figures,
        seed=SEED,
        n_folds_cap=1,
    )
    run(cfg)
    return {
        "models_dir": models_dir,
        "drift_reports": drift_reports,
        "drift_figures": drift_figures,
    }


def test_per_fold_json_written(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["drift_reports"] / "per_fold.json"
    assert p.exists()
    blocks = json.loads(p.read_text())
    assert isinstance(blocks, list)
    assert len(blocks) == 4  # 4 models x 1 fold (smoke)


def test_aggregate_json_written(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["drift_reports"] / "aggregate.json"
    assert p.exists()
    blob = json.loads(p.read_text())
    assert blob["n_cells"] == 4
    assert blob["config"]["smoke"] is True
    assert "by_model" in blob
    for model_name in ("tabicl", "xgboost", "lr", "ensemble"):
        assert model_name in blob["by_model"]


def test_per_fold_block_has_expected_keys(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["drift_reports"] / "per_fold.json"
    blocks = json.loads(p.read_text())
    expected = {
        "held_out_source",
        "model",
        "n_current",
        "severity_counts",
        "per_feature",
        "prediction",
        "compute_seconds",
    }
    for b in blocks:
        assert set(b.keys()) >= expected


def test_per_feature_block_has_expected_keys(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["drift_reports"] / "per_fold.json"
    blocks = json.loads(p.read_text())
    expected = {
        "feature",
        "kind",
        "psi",
        "severity",
        "n_ref",
        "n_cur",
        "n_missing_cur",
        "ks_statistic",
        "ks_p_value",
    }
    sample_block = blocks[0]
    for fd in sample_block["per_feature"]:
        assert set(fd.keys()) >= expected
        assert fd["kind"] in ("numeric", "categorical")
        assert fd["severity"] in ("stable", "moderate", "major")


def test_prediction_block_populated_for_every_cell(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["drift_reports"] / "per_fold.json"
    blocks = json.loads(p.read_text())
    for b in blocks:
        assert b["prediction"] is not None
        assert b["prediction"]["model_name"] == b["model"]


def test_dashboard_png_per_cell(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["drift_figures"]
    for model_name in ("tabicl", "xgboost", "lr", "ensemble"):
        p = figs / f"{model_name}_smoke_a_dashboard.png"
        assert p.exists(), f"missing dashboard for {model_name}"
        assert p.stat().st_size > 5_000


def test_smoke_drift_flags_at_least_one_feature(driver_outputs: dict[str, Path]) -> None:
    """Sanity guarantee that the script *can* surface drift: at least one
    feature in the smoke fold should land outside ``stable`` for at least
    one model. The smoke synthetic generator produces two distinct sources
    so the held-out source is genuinely different from the training pool."""
    p = driver_outputs["drift_reports"] / "per_fold.json"
    blocks = json.loads(p.read_text())
    flagged_anywhere = any(
        b["severity_counts"]["moderate"] + b["severity_counts"]["major"] >= 1 for b in blocks
    )
    assert flagged_anywhere, "smoke run produced no drifted features across any model"


def test_smoke_config_uses_smoke_paths() -> None:
    cfg = smoke_config()
    assert cfg.smoke is True
    assert cfg.n_folds_cap == 1
    assert "smoke" in str(cfg.reports_dir)
    assert "smoke" in str(cfg.figures_dir)


def test_full_config_uses_repo_paths() -> None:
    cfg = full_config()
    assert cfg.smoke is False
    assert cfg.n_folds_cap is None
    assert "smoke" not in str(cfg.reports_dir)
