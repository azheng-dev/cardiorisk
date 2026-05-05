"""Phase 2.3b training driver: fit + calibrate + evaluate the v1 stack.

Orchestrates the full Phase 2.3b run end-to-end:

1. Load combined HFP parquet (or a synthetic dataset in ``--smoke``).
2. ``clean_for_modelling`` (Phase 2.2 idempotent cleaning).
3. For each LODO fold (4 outer iterations on the four UCI sources):
   - ``within_fold_split`` -> 80/10/10 train / val / calib (val is reserved
     for the per-model inner CV inside each wrapper; calib is the post-hoc
     calibration slice).
   - For each model in (tabicl, xgboost, lr):
     - Fit on train slice (per-model wrapper handles its own hyperparam
       search internally).
     - Apply post-hoc calibration on the calib slice via
       :func:`cardiorisk.calibration.calibrate_for_model`.
     - Predict on the held-out source.
     - Compute headline metrics + bootstrap CIs.
     - Stratified subgroup metrics (sex, age band).
     - Decision-curve analysis at AusCVDRisk thresholds (5%, 10%).
     - Save reliability + DCA figures.
     - Persist the calibrated estimator via joblib (ADR-010).
4. Write per-fold JSONs + an aggregate JSON.

Modes:

- ``--smoke``: single LODO fold, single Optuna trial, 100 bootstrap
  resamples, ~15s wall clock. Used by CI to smoke the end-to-end path.
- ``--full``: all folds, full Optuna budget, full bootstrap. Used
  locally to produce the actual headline numbers (~30-50 min CPU).

macOS OpenMP note: torch (TabICL) and XGBoost both link against
libomp on macOS; loading both in one process deadlocks unless
``OMP_NUM_THREADS=1`` + ``KMP_DUPLICATE_LIB_OK=TRUE``. The CLI
wrapper at ``backend/scripts/train_v1.py`` sets these BEFORE
importing this module. Linux CI is unaffected; both libs use libgomp
consistently there.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required for headless / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cardiorisk.calibration import calibrate_for_model
from cardiorisk.data.combine import build_from_fixture
from cardiorisk.data.paths import (
    COMBINED_PARQUET,
    FIXTURE_PATH,
    MODELS_V1_DIR,
    REPORTS_V1_DIR,
    REPORTS_V1_FIGURES,
)
from cardiorisk.data.preprocess import clean_for_modelling
from cardiorisk.data.synthetic import generate_fixture
from cardiorisk.eval.bootstrap import bootstrap_ci
from cardiorisk.eval.dca import AUSCVDRISK_THRESHOLDS, decision_curve
from cardiorisk.eval.metrics import (
    auprc,
    auroc,
    brier,
    headline_metrics,
    sensitivity_at_specificity,
)
from cardiorisk.eval.reliability import reliability_diagram
from cardiorisk.eval.subgroup import assign_age_band, stratified_metrics
from cardiorisk.features.cv import (
    SOURCE_COLUMN,
    TARGET_COLUMN,
    iter_lodo_folds,
    within_fold_split,
)
from cardiorisk.models.base import MODEL_NAMES, SEED
from cardiorisk.models.lr import build_lr
from cardiorisk.models.tabicl import build_tabicl
from cardiorisk.models.xgboost_model import build_xgboost

logger = logging.getLogger("cardiorisk.training.train_v1")

#: Smoke-mode bootstrap resample count (vs default 2,000 in --full).
SMOKE_N_RESAMPLES: Final[int] = 100
FULL_N_RESAMPLES: Final[int] = 2_000

#: Smoke-mode XGBoost Optuna trial budget (vs default 50 in --full).
SMOKE_N_TRIALS: Final[int] = 1
FULL_N_TRIALS: Final[int] = 50

#: Smoke-mode synthetic dataset size (per pseudo-source). 100 rows x 2 sources
#: gives enough room for 80/10/10 within-fold splits and inner CV across
#: every model wrapper. The 20-row hfp_mini fixture is *too small* for the
#: full pipeline (LR's 5-fold inner CV needs ≥5 rows per fold per stratum).
SMOKE_ROWS_PER_SOURCE: Final[int] = 100  # 100 rows x 2 sources is enough for inner CV
SMOKE_SOURCES: Final[tuple[str, ...]] = ("smoke_a", "smoke_b")


# ----------------------------------------------------------------- types


@dataclass(frozen=True)
class RunConfig:
    """Resolved configuration for a single training-driver invocation."""

    smoke: bool
    n_trials: int
    n_resamples: int
    data_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    seed: int
    n_folds_cap: int | None  # None = all folds; smoke caps at 1


# ----------------------------------------------------------------- IO


def _generate_smoke_dataframe(seed: int) -> pd.DataFrame:
    """Build a multi-source synthetic dataset for the ``--smoke`` path.

    The committed hfp_mini.csv fixture is a single source ("fixture")
    of 20 rows, deliberately small for unit tests. That's too thin to
    exercise the LODO loop (needs ≥2 sources) or the LR's 5-fold inner
    CV (needs ≥5 rows per fold per stratum), so smoke mode generates
    its own dataset on the fly: ``SMOKE_ROWS_PER_SOURCE`` rows per
    pseudo-source in ``SMOKE_SOURCES``, all from the deterministic
    :func:`cardiorisk.data.synthetic.generate_fixture` generator.
    """
    frames: list[pd.DataFrame] = []
    for i, src in enumerate(SMOKE_SOURCES):
        rows = generate_fixture(n=SMOKE_ROWS_PER_SOURCE, seed=seed + i)
        df = pd.DataFrame(rows)
        df["source"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_dataframe(data_path: Path, smoke: bool, seed: int) -> pd.DataFrame:
    """Load combined HFP parquet; in smoke mode generate synthetic on-the-fly."""
    if smoke and data_path == FIXTURE_PATH:
        logger.info("smoke mode: generating multi-source synthetic dataset")
        return _generate_smoke_dataframe(seed)
    if data_path.suffix == ".parquet" and data_path.exists():
        logger.info("loading combined parquet from %s", data_path)
        return pd.read_parquet(data_path)
    if data_path.suffix == ".csv" and data_path.exists():
        logger.info("loading CSV from %s", data_path)
        return build_from_fixture(fixture_path=data_path)
    raise FileNotFoundError(
        f"No data at {data_path}. Run "
        "`uv run python backend/scripts/fetch_hfp.py` then "
        "`uv run python backend/scripts/build_combined.py` first; or use "
        "--smoke to fall back to a deterministic synthetic dataset."
    )


def _build_model(name: str, n_trials: int) -> Any:
    """Construct a fresh wrapper for ``name``."""
    if name == "tabicl":
        return build_tabicl()
    if name == "xgboost":
        return build_xgboost(n_trials=n_trials)
    if name == "lr":
        return build_lr()
    raise ValueError(f"unknown model name: {name!r}")


# ----------------------------------------------------------------- core


def _per_fold_eval(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sex: np.ndarray,
    age_band: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Compute the full per-fold evaluation block for one (model, fold)."""
    headline = headline_metrics(y_true, y_proba)

    # Bootstrap CIs for the three primary discrimination metrics + Brier.
    # We don't bootstrap calibration slope / intercept (their MLE is
    # numerically unstable on small bootstrap resamples in a way that
    # makes the CI itself misleading; reported point-only).
    cis = {
        "auroc": asdict(bootstrap_ci(auroc, y_true, y_proba, n_resamples=n_resamples, seed=seed)),
        "auprc": asdict(bootstrap_ci(auprc, y_true, y_proba, n_resamples=n_resamples, seed=seed)),
        "brier": asdict(bootstrap_ci(brier, y_true, y_proba, n_resamples=n_resamples, seed=seed)),
        "sensitivity_at_85_spec": asdict(
            bootstrap_ci(
                lambda y, p: sensitivity_at_specificity(y, p, 0.85),
                y_true,
                y_proba,
                n_resamples=n_resamples,
                seed=seed,
            )
        ),
        "sensitivity_at_90_spec": asdict(
            bootstrap_ci(
                lambda y, p: sensitivity_at_specificity(y, p, 0.90),
                y_true,
                y_proba,
                n_resamples=n_resamples,
                seed=seed,
            )
        ),
    }

    subgroup_sex = stratified_metrics(
        y_true, y_proba, sex, auroc, metric_name="auroc", grouping_name="sex"
    )
    subgroup_age = stratified_metrics(
        y_true, y_proba, age_band, auroc, metric_name="auroc", grouping_name="age_band"
    )

    dca = decision_curve(y_true, y_proba)
    dca_at = {f"{int(t * 100)}pct": dca.at(t) for t in AUSCVDRISK_THRESHOLDS}

    return {
        "model": model_name,
        "n_test": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "headline": headline.as_dict(),
        "headline_ci": cis,
        "subgroup_auroc_by_sex": {
            "by_stratum": [
                {"stratum": r.stratum, "n": r.n, "value": r.value} for r in subgroup_sex.by_stratum
            ],
            "fairness_gap": subgroup_sex.fairness_gap,
        },
        "subgroup_auroc_by_age_band": {
            "by_stratum": [
                {"stratum": r.stratum, "n": r.n, "value": r.value} for r in subgroup_age.by_stratum
            ],
            "fairness_gap": subgroup_age.fairness_gap,
        },
        "dca": dca_at,
    }


def _save_reliability_figure(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a reliability diagram PNG."""
    fig = reliability_diagram(y_true, y_proba, title=title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_dca_figure(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a decision-curve PNG (model vs treat-all vs treat-none)."""
    dca = decision_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca.thresholds, dca.net_benefit_model, label="model", linewidth=1.8)
    ax.plot(
        dca.thresholds, dca.net_benefit_treat_all, label="treat all", linestyle="--", color="grey"
    )
    ax.plot(
        dca.thresholds, dca.net_benefit_treat_none, label="treat none", linestyle=":", color="black"
    )
    for t in AUSCVDRISK_THRESHOLDS:
        ax.axvline(t, color="orange", alpha=0.3, linewidth=1)
    # Sensible y-axis floor: net benefit can go very negative for treat-all
    # at high thresholds; clip the visible window so the model curve is readable.
    ax.set_ylim(-0.05, max(0.5, float(dca.net_benefit_model.max()) + 0.05))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _run_one_fold_one_model(
    model_name: str,
    held_out_source: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_calib: pd.DataFrame,
    y_calib: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sex_test: np.ndarray,
    age_band_test: np.ndarray,
    cfg: RunConfig,
) -> dict[str, object]:
    """End-to-end fit -> calibrate -> predict -> eval -> persist for one cell."""
    t0 = time.perf_counter()
    logger.info("[%s | %s] fitting", held_out_source, model_name)
    model = _build_model(model_name, n_trials=cfg.n_trials)
    model.fit(X_train, y_train)

    logger.info("[%s | %s] calibrating", held_out_source, model_name)
    calibrated = calibrate_for_model(model, X_calib, y_calib, model_name=model_name)

    proba = np.asarray(calibrated.predict_proba(X_test))[:, 1]

    fold_block = _per_fold_eval(
        model_name=model_name,
        y_true=y_test,
        y_proba=proba,
        sex=sex_test,
        age_band=age_band_test,
        n_resamples=cfg.n_resamples,
        seed=cfg.seed,
    )
    fold_block["held_out_source"] = held_out_source
    fold_block["fit_seconds"] = round(time.perf_counter() - t0, 2)

    fig_stem = f"{model_name}_{held_out_source}"
    _save_reliability_figure(
        y_test,
        proba,
        cfg.figures_dir / f"{fig_stem}_reliability.png",
        title=f"{model_name} | held out {held_out_source}",
    )
    _save_dca_figure(
        y_test,
        proba,
        cfg.figures_dir / f"{fig_stem}_dca.png",
        title=f"{model_name} | held out {held_out_source}",
    )

    artefact_path = cfg.models_dir / f"{model_name}_{held_out_source}.joblib"
    joblib.dump(calibrated, artefact_path)
    headline = cast(dict[str, float], fold_block["headline"])
    logger.info(
        "[%s | %s] done in %.1fs (auroc=%.3f, brier=%.3f)",
        held_out_source,
        model_name,
        fold_block["fit_seconds"],
        headline["auroc"],
        headline["brier"],
    )
    return fold_block


def _to_json_safe(obj: object) -> object:
    """Recursively coerce ``NaN`` / ``inf`` to ``None`` so output is valid JSON.

    Python's stdlib ``json`` defaults to ``allow_nan=True`` which emits
    bare ``NaN`` / ``Infinity`` literals — accepted by Python and
    JavaScript but rejected by every strict JSON parser. We coerce
    them up-front and then dump with ``allow_nan=False`` so the file
    parses cleanly anywhere.
    """
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


def _aggregate(per_fold_blocks: list[dict[str, object]]) -> dict[str, object]:
    """LODO mean ± std per (model, metric) across folds."""
    by_model: dict[str, list[dict[str, object]]] = {}
    for block in per_fold_blocks:
        by_model.setdefault(str(block["model"]), []).append(block)

    out: dict[str, object] = {}
    for model_name, blocks in by_model.items():
        rows: dict[str, dict[str, float]] = {}
        for metric in (
            "auroc",
            "auprc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "sensitivity_at_85_spec",
            "sensitivity_at_90_spec",
        ):
            vals = np.array(
                [float(b["headline"][metric]) for b in blocks],  # type: ignore[index]
                dtype=np.float64,
            )
            vals = vals[~np.isnan(vals)]
            rows[metric] = {
                "mean": float(np.mean(vals)) if vals.size else float("nan"),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
                "n_folds": int(vals.size),
            }
        out[model_name] = rows
    return out


def run(cfg: RunConfig) -> dict[str, object]:
    """End-to-end driver. Returns the aggregate report dict (also written to disk)."""
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(cfg.data_path, smoke=cfg.smoke, seed=cfg.seed)
    df = clean_for_modelling(df)
    logger.info("dataset shape after clean: %s", df.shape)

    y_full = df[TARGET_COLUMN].to_numpy()
    X_full = df.drop(columns=[TARGET_COLUMN, SOURCE_COLUMN])
    sex_full = df["Sex"].to_numpy() if "Sex" in df.columns else np.array(["unknown"] * len(df))
    age_full = (
        df["Age"].apply(assign_age_band).to_numpy()
        if "Age" in df.columns
        else np.array(["unknown"] * len(df))
    )

    per_fold_blocks: list[dict[str, object]] = []
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
        y_train = y_full[wf.train_idx]
        X_calib = X_full.iloc[wf.calib_idx]
        y_calib = y_full[wf.calib_idx]
        X_test = X_full.iloc[fold.test_idx]
        y_test = y_full[fold.test_idx]
        sex_test = sex_full[fold.test_idx]
        age_band_test = age_full[fold.test_idx]

        for model_name in MODEL_NAMES:
            fold_block = _run_one_fold_one_model(
                model_name=model_name,
                held_out_source=fold.held_out_source,
                X_train=X_train,
                y_train=y_train,
                X_calib=X_calib,
                y_calib=y_calib,
                X_test=X_test,
                y_test=y_test,
                sex_test=sex_test,
                age_band_test=age_band_test,
                cfg=cfg,
            )
            per_fold_blocks.append(fold_block)

    per_fold_path = cfg.reports_dir / "metrics_per_fold.json"
    aggregate_path = cfg.reports_dir / "metrics_aggregate.json"
    per_fold_path.write_text(
        json.dumps(_to_json_safe(per_fold_blocks), indent=2, sort_keys=True, allow_nan=False)
    )
    aggregate = _aggregate(per_fold_blocks)
    report = {
        "config": {
            "smoke": cfg.smoke,
            "n_trials": cfg.n_trials,
            "n_resamples": cfg.n_resamples,
            "seed": cfg.seed,
            "n_folds_cap": cfg.n_folds_cap,
        },
        "by_model": aggregate,
    }
    aggregate_path.write_text(
        json.dumps(_to_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
    )
    logger.info("wrote %s and %s", per_fold_path, aggregate_path)
    return report


# ----------------------------------------------------------------- CLI


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 fold, 1 Optuna trial, 100 bootstrap resamples. CI default.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to combined parquet (default) or synthetic fixture CSV. "
            "If unset, uses processed/combined.parquet in --full and "
            "the bundled fixture in --smoke."
        ),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def resolve_config(args: argparse.Namespace) -> RunConfig:
    if args.smoke:
        data_path = args.data_path or FIXTURE_PATH
        return RunConfig(
            smoke=True,
            n_trials=SMOKE_N_TRIALS,
            n_resamples=SMOKE_N_RESAMPLES,
            data_path=data_path,
            models_dir=MODELS_V1_DIR / "smoke",
            reports_dir=REPORTS_V1_DIR / "smoke",
            figures_dir=REPORTS_V1_FIGURES / "smoke",
            seed=args.seed,
            n_folds_cap=1,
        )
    data_path = args.data_path or COMBINED_PARQUET
    return RunConfig(
        smoke=False,
        n_trials=FULL_N_TRIALS,
        n_resamples=FULL_N_RESAMPLES,
        data_path=data_path,
        models_dir=MODELS_V1_DIR,
        reports_dir=REPORTS_V1_DIR,
        figures_dir=REPORTS_V1_FIGURES,
        seed=args.seed,
        n_folds_cap=None,
    )


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
