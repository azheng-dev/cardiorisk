"""Phase 2.6 orchestrator: per (model x fold) drift report + dashboard.

End-to-end driver:

1. Load processed parquet (or generate the same synthetic dataset
   ``train_v1.py --smoke`` and ``compute_explanations.py --smoke`` use,
   in smoke mode).
2. ``clean_for_modelling`` (idempotent Phase 2.2 cleaning).
3. For each LODO fold (4 outer iterations on the four UCI sources):
   - Reproduce the same within-fold split + held-out test slice the
     training driver used. The fold's training pool is used to
     **build** the reference snapshot at orchestrator-time (rather
     than loading a persisted reference) — this keeps Phase 2.6 a
     fast, self-contained post-processing step that doesn't depend on
     anyone having pre-run ``backend/scripts/build_reference.py``.
     The standalone ``build_reference.py`` script is still shipped for
     the production-monitoring use case where the reference lives on
     disk.
   - For each model in (tabicl, xgboost, lr, ensemble):
     - Load the calibrated artefact from ``models/v1/<model>_<fold>.joblib``.
     - Use the **held-out source** as the "current" slice (per ADR-014
       §"Demo current slice").
     - :func:`cardiorisk.monitoring.drift.compute_drift` ->
       per-feature PSI + KS + prediction-drift PSI.
     - :func:`cardiorisk.monitoring.figures.render_drift_dashboard`
       -> single PNG dashboard.
4. Write per-fold JSON + an aggregate JSON.

Modes:

- ``--smoke``: 1 fold, smoke models, gitignored output dirs. CI uses
  this. Runs in ~30s, reusing the smoke-trained artefacts produced by
  the upstream ``train_v1.py --smoke`` step in the same CI job.
- ``--full``: all folds, the calibrated artefacts under ``models/v1/``,
  output under ``reports/v1/drift/`` (committed). The full sweep on the
  real data is fast — ~1 min CPU — because there's no SHAP-style
  perturbation: PSI is closed-form on the binned histograms.

The orchestrator deliberately does **not** retrain models. It loads
pre-trained calibrated artefacts from ``models/v1/`` (per ADR-010). If
those are missing it errors out telling the caller to run
``train_v1.py --full`` first. Same contract as the explainability
orchestrator.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cardiorisk.data.paths import (
    COMBINED_PARQUET,
    FIXTURE_PATH,
    MODELS_V1_DIR,
    REPORTS_V1_DRIFT,
    REPORTS_V1_DRIFT_FIGURES,
)
from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    clean_for_modelling,
)
from cardiorisk.features.cv import (
    SOURCE_COLUMN,
    TARGET_COLUMN,
    iter_lodo_folds,
    within_fold_split,
)
from cardiorisk.models.base import MODEL_NAMES, SEED
from cardiorisk.monitoring.drift import DriftReport, compute_drift
from cardiorisk.monitoring.figures import render_drift_dashboard
from cardiorisk.monitoring.reference import DEFAULT_N_BINS, build_fold_reference

logger = logging.getLogger("cardiorisk.monitoring.orchestrator")


# ----------------------------------------------------------------- config


@dataclass(frozen=True)
class DriftConfig:
    """Resolved configuration for one orchestrator invocation."""

    smoke: bool
    n_bins: int
    data_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    seed: int
    n_folds_cap: int | None


def smoke_config(*, data_path: Path | None = None) -> DriftConfig:
    """Smoke-mode defaults. 1 fold, smoke models, gitignored output dirs."""
    return DriftConfig(
        smoke=True,
        n_bins=DEFAULT_N_BINS,
        data_path=data_path or FIXTURE_PATH,
        models_dir=MODELS_V1_DIR / "smoke",
        reports_dir=REPORTS_V1_DRIFT / "smoke",
        figures_dir=REPORTS_V1_DRIFT_FIGURES / "smoke",
        seed=SEED,
        n_folds_cap=1,
    )


def full_config(*, data_path: Path | None = None) -> DriftConfig:
    """Full-mode defaults from ADR-014."""
    return DriftConfig(
        smoke=False,
        n_bins=DEFAULT_N_BINS,
        data_path=data_path or COMBINED_PARQUET,
        models_dir=MODELS_V1_DIR,
        reports_dir=REPORTS_V1_DRIFT,
        figures_dir=REPORTS_V1_DRIFT_FIGURES,
        seed=SEED,
        n_folds_cap=None,
    )


# ----------------------------------------------------------------- IO


_SMOKE_ROWS_PER_SOURCE: Final[int] = 100
_SMOKE_SOURCES: Final[tuple[str, ...]] = ("smoke_a", "smoke_b")


def _generate_smoke_dataframe(seed: int) -> pd.DataFrame:
    """Multi-source synthetic dataset matching ``train_v1`` smoke behaviour."""
    from cardiorisk.data.synthetic import generate_fixture

    frames: list[pd.DataFrame] = []
    for i, src in enumerate(_SMOKE_SOURCES):
        rows = generate_fixture(n=_SMOKE_ROWS_PER_SOURCE, seed=seed + i)
        df = pd.DataFrame(rows)
        df["source"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_dataframe(data_path: Path, smoke: bool, seed: int) -> pd.DataFrame:
    if smoke and data_path == FIXTURE_PATH:
        logger.info("smoke mode: generating multi-source synthetic dataset")
        return _generate_smoke_dataframe(seed)
    if data_path.suffix == ".parquet" and data_path.exists():
        logger.info("loading combined parquet from %s", data_path)
        return pd.read_parquet(data_path)
    raise FileNotFoundError(
        f"No data at {data_path}. Run train_v1.py --full first; or use --smoke."
    )


def _load_calibrated(model_name: str, source: str, models_dir: Path) -> Any:
    """Load the calibrated artefact for one (model, fold)."""
    p = models_dir / f"{model_name}_{source}.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"missing calibrated artefact {p}. "
            "Run `uv run --project backend python backend/scripts/train_v1.py` first."
        )
    return joblib.load(p)


# ----------------------------------------------------------------- core


_NUMERIC_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS


def _to_json_safe(obj: object) -> object:
    """Recursively coerce NaN / inf to None for strict JSON output."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _drift_report_to_dict(report: DriftReport) -> dict[str, object]:
    """Serialise a :class:`DriftReport` into the JSON schema."""
    per_feature_block: list[dict[str, object]] = []
    for fd in sorted(report.per_feature.values(), key=lambda x: (-x.psi, x.feature)):
        per_feature_block.append(
            {
                "feature": fd.feature,
                "kind": fd.kind,
                "psi": fd.psi,
                "severity": fd.severity,
                "n_ref": fd.n_ref,
                "n_cur": fd.n_cur,
                "n_missing_cur": fd.n_missing_cur,
                "ks_statistic": fd.ks_statistic,
                "ks_p_value": fd.ks_p_value,
            }
        )
    prediction_block: dict[str, object] | None = None
    if report.prediction is not None:
        pd_block = report.prediction
        prediction_block = {
            "model_name": pd_block.model_name,
            "psi": pd_block.psi,
            "severity": pd_block.severity,
            "n_ref": pd_block.n_ref,
            "n_cur": pd_block.n_cur,
            "mean_ref": pd_block.mean_ref,
            "mean_cur": pd_block.mean_cur,
        }
    return {
        "held_out_source": report.held_out_source,
        "model": report.model_name,
        "n_current": report.n_current,
        "severity_counts": dict(report.severity_counts),
        "per_feature": per_feature_block,
        "prediction": prediction_block,
    }


def _save_fig(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _drift_one_cell(
    *,
    model_name: str,
    held_out_source: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: DriftConfig,
) -> dict[str, object]:
    """Build reference, compute drift, render dashboard for one (model, fold)."""
    t0 = time.perf_counter()
    logger.info("[%s | %s] loading calibrated artefact", held_out_source, model_name)
    calibrated = _load_calibrated(model_name, held_out_source, cfg.models_dir)

    logger.info("[%s | %s] building per-fold reference", held_out_source, model_name)
    reference = build_fold_reference(
        held_out_source=held_out_source,
        X_train=X_train,
        models={model_name: calibrated},
        n_bins=cfg.n_bins,
    )

    logger.info("[%s | %s] computing drift on held-out source", held_out_source, model_name)
    report = compute_drift(
        reference=reference,
        X_current=X_test,
        model=calibrated,
        model_name=model_name,
    )

    fig_stem = f"{model_name}_{held_out_source}"
    current_numeric = {
        col: X_test[col].to_numpy(dtype=np.float64, na_value=np.nan)
        for col in _NUMERIC_COLUMNS
        if col in X_test.columns
    }
    current_proba = np.asarray(calibrated.predict_proba(X_test), dtype=np.float64)[:, 1]
    fig = render_drift_dashboard(
        report=report,
        reference=reference,
        current_numeric=current_numeric,
        current_proba=current_proba,
    )
    _save_fig(fig, cfg.figures_dir / f"{fig_stem}_dashboard.png")

    block = _drift_report_to_dict(report)
    block["compute_seconds"] = round(time.perf_counter() - t0, 2)
    logger.info(
        "[%s | %s] done in %.1fs (severity counts: %s)",
        held_out_source,
        model_name,
        block["compute_seconds"],
        report.severity_counts,
    )
    return block


def run(cfg: DriftConfig) -> dict[str, object]:
    """End-to-end drift driver. Returns the aggregate report dict."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(cfg.data_path, smoke=cfg.smoke, seed=cfg.seed)
    df = clean_for_modelling(df)
    logger.info("dataset shape after clean: %s", df.shape)

    y_full = df[TARGET_COLUMN].to_numpy()
    X_full = df.drop(columns=[TARGET_COLUMN, SOURCE_COLUMN])

    per_cell_blocks: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(iter_lodo_folds(df)):
        if cfg.n_folds_cap is not None and fold_idx >= cfg.n_folds_cap:
            logger.info("smoke mode: stopping after %d fold(s)", cfg.n_folds_cap)
            break

        logger.info(
            "fold %d: held out source=%s (n_train=%d, n_test=%d)",
            fold_idx + 1,
            fold.held_out_source,
            fold.train_idx.size,
            fold.test_idx.size,
        )

        wf = within_fold_split(fold.train_idx, y_full, seed=cfg.seed)
        X_train = X_full.iloc[wf.train_idx]
        X_test = X_full.iloc[fold.test_idx]

        for model_name in MODEL_NAMES:
            cell_block = _drift_one_cell(
                model_name=model_name,
                held_out_source=fold.held_out_source,
                X_train=X_train,
                X_test=X_test,
                cfg=cfg,
            )
            per_cell_blocks.append(cell_block)

    aggregate = _aggregate(per_cell_blocks)
    per_fold_path = cfg.reports_dir / "per_fold.json"
    aggregate_path = cfg.reports_dir / "aggregate.json"

    per_fold_path.write_text(
        json.dumps(_to_json_safe(per_cell_blocks), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = {
        "config": {
            "smoke": cfg.smoke,
            "n_bins": cfg.n_bins,
            "seed": cfg.seed,
            "n_folds_cap": cfg.n_folds_cap,
        },
        "n_cells": len(per_cell_blocks),
        "by_model": aggregate,
    }
    aggregate_path.write_text(
        json.dumps(_to_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    logger.info("wrote %s and %s", per_fold_path, aggregate_path)
    return report


def _aggregate(per_cell_blocks: list[dict[str, object]]) -> dict[str, object]:
    """Cross-fold severity tally + mean prediction-drift PSI per model."""
    by_model: dict[str, list[dict[str, object]]] = {}
    for block in per_cell_blocks:
        by_model.setdefault(str(block["model"]), []).append(block)

    out: dict[str, object] = {}
    for model_name, blocks in by_model.items():
        sev_total = {"stable": 0, "moderate": 0, "major": 0}
        for b in blocks:
            counts = b["severity_counts"]
            if not isinstance(counts, dict):
                raise TypeError(f"unexpected severity_counts payload: {type(counts).__name__}")
            for k, v in counts.items():
                sev_total[str(k)] += int(v)
        pred_psis = [
            float(b["prediction"]["psi"])  # type: ignore[index]
            for b in blocks
            if b.get("prediction") is not None
        ]
        out[model_name] = {
            "n_folds": len(blocks),
            "severity_counts_total": sev_total,
            "prediction_psi_mean": float(np.mean(pred_psis)) if pred_psis else float("nan"),
            "prediction_psi_max": float(np.max(pred_psis)) if pred_psis else float("nan"),
        }
    return out


# ----------------------------------------------------------------- CLI


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 fold, smoke models, gitignored output dirs. CI default.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to combined parquet (default) or fixture CSV. "
            "If unset uses processed/combined.parquet in --full and "
            "the bundled fixture in --smoke."
        ),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def resolve_config(args: argparse.Namespace) -> DriftConfig:
    if args.smoke:
        return smoke_config(data_path=args.data_path)
    return full_config(data_path=args.data_path)


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = resolve_config(args)
    logger.info("config: %s", cfg)
    run(cfg)


# Re-export for the categorical column name list — used by tests that
# build a synthetic "current" slice and want to know which columns the
# orchestrator considers categorical when reasoning about drift.
__all__ = [
    "CATEGORICAL_COLUMNS",
    "DriftConfig",
    "build_argparser",
    "full_config",
    "main",
    "resolve_config",
    "run",
    "smoke_config",
]
