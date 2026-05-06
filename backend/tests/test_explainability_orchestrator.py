"""End-to-end smoke test for :mod:`cardiorisk.explainability.orchestrator`.

Asserts the orchestrator produces the expected JSON files + figure
inventory for a 1-fold smoke run, given pre-trained smoke models live
under ``models/v1/smoke/``. The training driver smoke run produces
those, so this test re-uses the same orchestrator path the CLI does.

We do *not* re-train inside the test (the training driver has its own
smoke test under ``test_train_v1.py``); instead we run a tiny train +
explain pipeline once per session and assert on the output files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cardiorisk.data.paths import (
    MODELS_V1_DIR,
    REPORTS_V1_EXPLAIN,
    REPORTS_V1_EXPLAIN_FIGURES,
)
from cardiorisk.explainability.orchestrator import (
    ExplainConfig,
    full_config,
    run,
    smoke_config,
)
from cardiorisk.training.train_v1 import RunConfig as TrainRunConfig
from cardiorisk.training.train_v1 import run as train_run


@pytest.fixture(scope="module")
def driver_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train smoke models then run the explainability orchestrator once.

    Outputs are written to a per-test temporary directory so the suite
    doesn't trample the developer's local Phase 2.5 outputs (and so a
    parallel pytest-xdist run wouldn't race).
    """
    from cardiorisk.data.paths import FIXTURE_PATH
    from cardiorisk.explainability.kernel_shap import (
        SMOKE_BACKGROUND_K,
        SMOKE_NSAMPLES,
    )
    from cardiorisk.models.base import SEED
    from cardiorisk.models.ensemble import SMOKE_N_EPOCHS
    from cardiorisk.training.train_v1 import SMOKE_N_RESAMPLES

    tmp = tmp_path_factory.mktemp("p25-orch-smoke")
    models_dir = tmp / "models"
    train_reports = tmp / "train_reports"
    train_figures = tmp / "train_figures"
    explain_reports = tmp / "explain_reports"
    explain_figures = tmp / "explain_figures"
    for d in (models_dir, train_reports, train_figures, explain_reports, explain_figures):
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

    cfg = ExplainConfig(
        smoke=True,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
        min_stratum_size=10,
        max_test_rows=30,
        data_path=FIXTURE_PATH,
        models_dir=models_dir,
        reports_dir=explain_reports,
        figures_dir=explain_figures,
        seed=SEED,
        n_folds_cap=1,
    )
    run(cfg)
    return {
        "models_dir": models_dir,
        "explain_reports": explain_reports,
        "explain_figures": explain_figures,
    }


def test_per_cell_json_written(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["explain_reports"] / "explanations_per_cell.json"
    assert p.exists()
    blocks = json.loads(p.read_text())
    assert isinstance(blocks, list)
    assert len(blocks) == 4  # 4 models x 1 fold (smoke)


def test_aggregate_json_written(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["explain_reports"] / "explanations_aggregate.json"
    assert p.exists()
    blob = json.loads(p.read_text())
    assert blob["n_cells"] == 4
    assert blob["config"]["smoke"] is True


def test_cross_model_agreement_json_written(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["explain_reports"] / "cross_model_agreement.json"
    assert p.exists()
    blob = json.loads(p.read_text())
    assert "per_fold" in blob
    assert "aggregate" in blob
    assert len(blob["per_fold"]) == 1


def test_per_cell_block_has_expected_keys(driver_outputs: dict[str, Path]) -> None:
    p = driver_outputs["explain_reports"] / "explanations_per_cell.json"
    blocks = json.loads(p.read_text())
    expected_keys = {
        "model",
        "held_out_source",
        "fit_seconds",
        "n_test_full",
        "n_test_explained",
        "expected_value",
        "global_importance",
        "subgroup_drift_sex",
        "subgroup_drift_age_band",
        "archetypes",
        "sanity",
    }
    for b in blocks:
        assert set(b.keys()) >= expected_keys


def test_global_bar_figure_exists_per_model(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["explain_figures"]
    for model_name in ("tabicl", "xgboost", "lr", "ensemble"):
        p = figs / f"{model_name}_smoke_a_global_bar.png"
        assert p.exists(), f"missing global bar for {model_name}"


def test_cross_model_heatmap_per_fold_and_aggregate(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["explain_figures"]
    assert (figs / "smoke_a_cross_model_agreement_heatmap.png").exists()
    assert (figs / "aggregate_cross_model_agreement_heatmap.png").exists()


def test_xgboost_treeshap_sanity_figure_exists(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["explain_figures"]
    assert (figs / "xgboost_smoke_a_treeshap_vs_kernelshap.png").exists()


def test_lr_summed_vs_basis_figure_exists(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["explain_figures"]
    assert (figs / "lr_smoke_a_summed_vs_basis.png").exists()


def test_archetype_waterfalls_present(driver_outputs: dict[str, Path]) -> None:
    figs = driver_outputs["explain_figures"]
    # At minimum tp_high should exist for every model on the smoke fold.
    for model_name in ("tabicl", "xgboost", "lr", "ensemble"):
        for arch_label in ("tp_high",):
            p = figs / f"{model_name}_smoke_a_{arch_label}_waterfall.png"
            assert p.exists(), f"missing {arch_label} waterfall for {model_name}"


def test_smoke_config_uses_smoke_paths() -> None:
    cfg = smoke_config()
    assert cfg.smoke is True
    assert cfg.n_folds_cap == 1
    assert "smoke" in str(cfg.reports_dir)
    assert "smoke" in str(cfg.figures_dir)
    assert "smoke" in str(cfg.models_dir)


def test_full_config_uses_full_paths() -> None:
    cfg = full_config()
    assert cfg.smoke is False
    assert cfg.n_folds_cap is None
    assert cfg.reports_dir == REPORTS_V1_EXPLAIN
    assert cfg.figures_dir == REPORTS_V1_EXPLAIN_FIGURES
    assert cfg.models_dir == MODELS_V1_DIR


def test_resolve_config_max_test_rows_override_full() -> None:
    from cardiorisk.explainability.orchestrator import build_argparser, resolve_config

    args = build_argparser().parse_args(["--max-test-rows", "12"])
    cfg = resolve_config(args)
    assert cfg.smoke is False
    assert cfg.max_test_rows == 12


def test_resolve_config_max_test_rows_override_smoke() -> None:
    from cardiorisk.explainability.orchestrator import build_argparser, resolve_config

    args = build_argparser().parse_args(["--smoke", "--max-test-rows", "5"])
    cfg = resolve_config(args)
    assert cfg.smoke is True
    assert cfg.max_test_rows == 5


def test_resolve_config_max_test_rows_zero_means_explain_everything() -> None:
    from cardiorisk.explainability.orchestrator import build_argparser, resolve_config

    args = build_argparser().parse_args(["--max-test-rows", "0"])
    cfg = resolve_config(args)
    # 0 maps to a sentinel >> any realistic test-slice size, which makes
    # _stratified_test_sample short-circuit to np.arange(len(y_test)).
    assert cfg.max_test_rows >= 10**8


def test_resolve_config_max_test_rows_negative_rejected() -> None:
    from cardiorisk.explainability.orchestrator import build_argparser, resolve_config

    args = build_argparser().parse_args(["--max-test-rows", "-1"])
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_config(args)


def test_resolve_config_max_test_rows_default_preserves_full() -> None:
    from cardiorisk.explainability.kernel_shap import DEFAULT_BACKGROUND_K
    from cardiorisk.explainability.orchestrator import (
        DEFAULT_MAX_TEST_ROWS,
        build_argparser,
        resolve_config,
    )

    args = build_argparser().parse_args([])
    cfg = resolve_config(args)
    assert cfg.max_test_rows == DEFAULT_MAX_TEST_ROWS
    assert cfg.background_k == DEFAULT_BACKGROUND_K
