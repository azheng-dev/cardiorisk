"""Phase 2.5 orchestrator: per (model x fold) SHAP computation + figure dump.

End-to-end driver:

1. Load processed parquet (or generate the same synthetic dataset
   ``train_v1.py --smoke`` uses, in smoke mode).
2. ``clean_for_modelling`` (idempotent Phase 2.2 cleaning).
3. For each LODO fold (4 outer iterations on the four UCI sources):
   - Reproduce the same within-fold split + held-out test slice the
     training driver used (same seed, same splitter -- guaranteeing
     the models we load from ``models/v1/`` are explained against the
     test slice they were evaluated on in Phase 2.4).
   - Fit the shared encoder on the per-fold training slice.
   - For each model in (tabicl, xgboost, lr, ensemble):
     - Load the calibrated artefact from ``models/v1/<model>_<fold>.joblib``.
     - Run KernelSHAP on the test slice (cross-model headline).
     - Sanity-check passes:
       - TreeSHAP for XGBoost.
       - Analytic linear attribution for LR.
     - Pick 4 archetypes; render waterfall plots.
     - Render global bar + beeswarm.
   - Per-fold cross-model agreement matrix + heatmap.
   - Per-fold subgroup-drift bars (auditable strata only).
   - Per-fold sanity figures (TreeSHAP-vs-KernelSHAP scatter, LR
     summed-vs-basis bar).
4. Aggregate-across-folds cross-model agreement matrix + heatmap.
5. Write JSONs to ``reports/v1/explainability/`` and PNGs to
   ``reports/v1/figures/explainability/``.

Modes:

- ``--smoke``: 1 fold, smoke kernel-shap budget, ~30s wall clock.
  Used by CI to smoke the end-to-end path.
- ``--full``: all folds, full kernel-shap budget. Used locally to
  produce the actual reports + figures (~30-60 min CPU per the ADR-013
  estimate).

The orchestrator deliberately *does not* re-train models. It loads
pre-trained calibrated artefacts from ``models/v1/`` (per ADR-010).
If those are missing it errors out telling the caller to run
``train_v1.py --full`` first. This makes Phase 2.5 a fast post-
processing step rather than a re-train.

macOS OpenMP note: ``OMP_NUM_THREADS=1`` + ``KMP_DUPLICATE_LIB_OK=TRUE``
must be set before this module imports any model wrapper. The CLI
wrapper at ``backend/scripts/compute_explanations.py`` sets these.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required for headless / CI
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from cardiorisk.data.paths import (
    COMBINED_PARQUET,
    FIXTURE_PATH,
    MODELS_V1_DIR,
    REPORTS_V1_EXPLAIN,
    REPORTS_V1_EXPLAIN_FIGURES,
)
from cardiorisk.data.preprocess import clean_for_modelling
from cardiorisk.eval.subgroup import assign_age_band
from cardiorisk.explainability.archetypes import Archetype, pick_archetypes
from cardiorisk.explainability.cross_model_agreement import (
    AgreementResult,
    aggregate_across_folds,
    compute_cross_model_agreement,
)
from cardiorisk.explainability.encoder import EncodedFeatureSpace, fit_encoder
from cardiorisk.explainability.figures import (
    archetype_features_to_dataframe,
    cross_model_agreement_heatmap,
    global_importance_bar,
    global_importance_beeswarm,
    lr_summed_vs_basis_bar,
    subgroup_drift_bar,
    treeshap_vs_kernelshap_scatter,
    waterfall,
)
from cardiorisk.explainability.kernel_shap import (
    DEFAULT_BACKGROUND_K,
    DEFAULT_NSAMPLES,
    SMOKE_BACKGROUND_K,
    SMOKE_NSAMPLES,
    KernelSHAPResult,
    explain_with_kernel_shap,
)
from cardiorisk.explainability.linear_attribution import attribute_lr
from cardiorisk.explainability.subgroup_drift import (
    DEFAULT_MIN_STRATUM_SIZE,
    SubgroupDriftResult,
    compute_subgroup_drift,
)
from cardiorisk.explainability.tree_shap import explain_xgboost_with_tree_shap
from cardiorisk.features.cv import (
    SOURCE_COLUMN,
    TARGET_COLUMN,
    iter_lodo_folds,
    within_fold_split,
)
from cardiorisk.models.base import MODEL_NAMES, SEED
from cardiorisk.models.lr import LRModel

logger = logging.getLogger("cardiorisk.explainability.orchestrator")


# ----------------------------------------------------------------- config


#: Per (model x fold) maximum number of test rows to explain via KernelSHAP.
#: ADR-013's "Trigger to revisit" anticipates this cap: explaining all 303
#: Cleveland test rows x 4 models would push the full LODO sweep past the
#: 90-minute budget. We instead stratified-sample by ``y_test`` (preserving
#: per-fold prevalence) up to this cap, plus the 4 archetype rows by name
#: so the local-explanation gallery is always rendered against the actual
#: archetype patients.
DEFAULT_MAX_TEST_ROWS: Final[int] = 80
SMOKE_MAX_TEST_ROWS: Final[int] = 30


@dataclass(frozen=True)
class ExplainConfig:
    """Resolved configuration for one orchestrator invocation."""

    smoke: bool
    background_k: int
    nsamples: int
    min_stratum_size: int
    max_test_rows: int  # ADR-013 wall-clock guard; per (model x fold).
    data_path: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    seed: int
    n_folds_cap: int | None  # None = all folds; smoke caps at 1


def smoke_config(*, data_path: Path | None = None) -> ExplainConfig:
    """Smoke-mode defaults. Tiny budget, 1 fold, gitignored output dirs."""
    return ExplainConfig(
        smoke=True,
        background_k=SMOKE_BACKGROUND_K,
        nsamples=SMOKE_NSAMPLES,
        min_stratum_size=10,
        max_test_rows=SMOKE_MAX_TEST_ROWS,
        data_path=data_path or FIXTURE_PATH,
        models_dir=MODELS_V1_DIR / "smoke",
        reports_dir=REPORTS_V1_EXPLAIN / "smoke",
        figures_dir=REPORTS_V1_EXPLAIN_FIGURES / "smoke",
        seed=SEED,
        n_folds_cap=1,
    )


def full_config(*, data_path: Path | None = None) -> ExplainConfig:
    """Full-mode defaults from ADR-013."""
    return ExplainConfig(
        smoke=False,
        background_k=DEFAULT_BACKGROUND_K,
        nsamples=DEFAULT_NSAMPLES,
        min_stratum_size=DEFAULT_MIN_STRATUM_SIZE,
        max_test_rows=DEFAULT_MAX_TEST_ROWS,
        data_path=data_path or COMBINED_PARQUET,
        models_dir=MODELS_V1_DIR,
        reports_dir=REPORTS_V1_EXPLAIN,
        figures_dir=REPORTS_V1_EXPLAIN_FIGURES,
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


def _stratified_test_sample(
    *,
    y_test: np.ndarray,
    archetype_indices: list[int],
    max_rows: int,
    seed: int,
) -> npt.NDArray[np.int64]:
    """Sample test-row indices stratified by class, including all archetypes.

    Returns indices into the full per-fold test slice, sorted ascending so
    the SHAP value matrix lines up with downstream ``X_test.iloc[indices]``.
    Archetype rows are always included (so the local-explanation gallery is
    rendered against real archetype patients, not nearest-sample stand-ins).
    The remaining slots are filled by stratified random sampling on
    ``y_test`` to preserve per-fold prevalence.
    """
    if max_rows >= len(y_test):
        return np.arange(len(y_test), dtype=np.int64)

    rng = np.random.default_rng(seed)
    archetype_set = {int(i) for i in archetype_indices}
    remaining_budget = max(0, max_rows - len(archetype_set))

    pos_pool = [i for i in range(len(y_test)) if y_test[i] == 1 and i not in archetype_set]
    neg_pool = [i for i in range(len(y_test)) if y_test[i] == 0 and i not in archetype_set]
    overall_prevalence = float(np.mean(y_test == 1))
    n_pos_to_draw = min(round(remaining_budget * overall_prevalence), len(pos_pool))
    n_neg_to_draw = min(remaining_budget - n_pos_to_draw, len(neg_pool))

    sampled_pos = (
        rng.choice(pos_pool, size=n_pos_to_draw, replace=False).tolist()
        if n_pos_to_draw > 0
        else []
    )
    sampled_neg = (
        rng.choice(neg_pool, size=n_neg_to_draw, replace=False).tolist()
        if n_neg_to_draw > 0
        else []
    )

    indices = sorted({*archetype_set, *sampled_pos, *sampled_neg})
    return np.asarray(indices, dtype=np.int64)


def _explain_one_model_one_fold(
    *,
    model_name: str,
    held_out_source: str,
    encoded_space: EncodedFeatureSpace,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sex_test: np.ndarray,
    age_test: np.ndarray,
    cfg: ExplainConfig,
) -> dict[str, object]:
    """Explain one (model, fold) cell. Returns the per-cell JSON block."""
    t0 = time.perf_counter()
    logger.info("[%s | %s] loading calibrated artefact", held_out_source, model_name)
    calibrated = _load_calibrated(model_name, held_out_source, cfg.models_dir)

    # Pick archetypes against the FULL test slice (cheap: just predict_proba).
    proba_full = np.asarray(calibrated.predict_proba(X_test))[:, 1]
    archetypes_full = pick_archetypes(y_true=y_test, y_proba=proba_full)

    # KernelSHAP is the expensive step. ADR-013 §"Trigger to revisit" pins
    # an upper bound on how many test rows we explain per (model, fold);
    # we stratified-sample by class and force-include the archetype rows
    # so the local-explanation gallery still shows the real high-/low-risk
    # patients.
    sampled_idx = _stratified_test_sample(
        y_test=y_test,
        archetype_indices=[a.test_index for a in archetypes_full],
        max_rows=cfg.max_test_rows,
        seed=cfg.seed,
    )
    X_test_sampled = X_test.iloc[sampled_idx]
    y_test_sampled = y_test[sampled_idx]
    sex_test_sampled = sex_test[sampled_idx]
    age_test_sampled = age_test[sampled_idx]
    proba_sampled = proba_full[sampled_idx]

    # Re-pick archetypes against the SAMPLED slice so test_index lines up
    # with the SHAP-value matrix rows (the archetypes_full set was used to
    # ensure the sample includes them; here we get the in-sample positions).
    archetypes = pick_archetypes(y_true=y_test_sampled, y_proba=proba_sampled)

    logger.info(
        "[%s | %s] running KernelSHAP (background_k=%d, nsamples=%d, n_test=%d)",
        held_out_source,
        model_name,
        cfg.background_k,
        cfg.nsamples,
        len(X_test_sampled),
    )
    kshap = explain_with_kernel_shap(
        predict_proba=calibrated.predict_proba,
        encoded_space=encoded_space,
        X_train=X_train,
        X_test=X_test_sampled,
        background_k=cfg.background_k,
        nsamples=cfg.nsamples,
        seed=cfg.seed,
    )

    fig_stem = f"{model_name}_{held_out_source}"

    _save_fig(
        global_importance_bar(
            mean_abs_per_feature=kshap.mean_abs_per_raw_feature,
            title=f"{model_name} | {held_out_source} | global importance (KernelSHAP)",
        ),
        cfg.figures_dir / f"{fig_stem}_global_bar.png",
    )
    _save_fig(
        global_importance_beeswarm(
            shap_values_raw=kshap.shap_values_raw,
            raw_feature_names=kshap.raw_feature_names,
            title=f"{model_name} | {held_out_source} | beeswarm (KernelSHAP)",
        ),
        cfg.figures_dir / f"{fig_stem}_global_beeswarm.png",
    )
    for arch in archetypes:
        _save_fig(
            waterfall(
                shap_row=kshap.shap_values_raw[arch.test_index],
                raw_feature_names=kshap.raw_feature_names,
                expected_value=kshap.expected_value,
                archetype=arch,
            ),
            cfg.figures_dir / f"{fig_stem}_{arch.label}_waterfall.png",
        )

    sex_drift = compute_subgroup_drift(
        grouping_name="sex",
        grouping_values=sex_test_sampled,
        shap_values_raw=kshap.shap_values_raw,
        raw_feature_names=kshap.raw_feature_names,
        min_stratum_size=cfg.min_stratum_size,
    )
    age_drift = compute_subgroup_drift(
        grouping_name="age_band",
        grouping_values=age_test_sampled,
        shap_values_raw=kshap.shap_values_raw,
        raw_feature_names=kshap.raw_feature_names,
        min_stratum_size=cfg.min_stratum_size,
    )

    if sex_drift.by_stratum:
        _save_fig(
            subgroup_drift_bar(
                drift=sex_drift,
                title=f"{model_name} | {held_out_source} | sex drift",
            ),
            cfg.figures_dir / f"{fig_stem}_subgroup_drift_sex.png",
        )
    if age_drift.by_stratum:
        _save_fig(
            subgroup_drift_bar(
                drift=age_drift,
                title=f"{model_name} | {held_out_source} | age-band drift",
            ),
            cfg.figures_dir / f"{fig_stem}_subgroup_drift_age_band.png",
        )

    # Sanity passes for XGB + LR. Run against the SAMPLED test slice so the
    # TreeSHAP-vs-KernelSHAP scatter is on the same rows the headline
    # KernelSHAP block was computed on -- otherwise the comparison would
    # mix per-feature means computed over different samples.
    sanity_block: dict[str, object] = {}
    if model_name == "xgboost":
        sanity_block["treeshap"] = _run_xgboost_treeshap_sanity(
            calibrated=calibrated,
            X_test=X_test_sampled,
            kshap=kshap,
            held_out_source=held_out_source,
            cfg=cfg,
        )
    if model_name == "lr":
        sanity_block["lr_basis"] = _run_lr_basis_sanity(
            calibrated=calibrated,
            X_test=X_test_sampled,
            held_out_source=held_out_source,
            cfg=cfg,
        )

    fit_seconds = round(time.perf_counter() - t0, 2)
    logger.info(
        "[%s | %s] done in %.1fs",
        held_out_source,
        model_name,
        fit_seconds,
    )

    return {
        "model": model_name,
        "held_out_source": held_out_source,
        "fit_seconds": fit_seconds,
        "n_test_full": len(X_test),
        "n_test_explained": len(X_test_sampled),
        "expected_value": kshap.expected_value,
        "global_importance": kshap.mean_abs_per_raw_feature,
        "subgroup_drift_sex": _drift_to_dict(sex_drift),
        "subgroup_drift_age_band": _drift_to_dict(age_drift),
        "archetypes": [_archetype_to_dict(a, X_test_sampled) for a in archetypes],
        "sanity": sanity_block,
    }


def _run_xgboost_treeshap_sanity(
    *,
    calibrated: Any,
    X_test: pd.DataFrame,
    kshap: KernelSHAPResult,
    held_out_source: str,
    cfg: ExplainConfig,
) -> dict[str, object]:
    """TreeSHAP-vs-KernelSHAP scatter + per-feature mean |SHAP|."""
    tshap = explain_xgboost_with_tree_shap(calibrated_or_bare_xgb=calibrated, X_test=X_test)
    _save_fig(
        treeshap_vs_kernelshap_scatter(
            treeshap_per_raw=tshap.mean_abs_per_raw_feature,
            kernelshap_per_raw=kshap.mean_abs_per_raw_feature,
            title=f"xgboost | {held_out_source} | TreeSHAP vs KernelSHAP",
        ),
        cfg.figures_dir / f"xgboost_{held_out_source}_treeshap_vs_kernelshap.png",
    )
    return {
        "treeshap_mean_abs_per_raw": tshap.mean_abs_per_raw_feature,
        "treeshap_expected_value": tshap.expected_value,
    }


def _run_lr_basis_sanity(
    *,
    calibrated: Any,
    X_test: pd.DataFrame,
    held_out_source: str,
    cfg: ExplainConfig,
) -> dict[str, object]:
    """Analytic LR per-basis + summed-back attribution."""
    lr_model = _resolve_lr(calibrated)
    attr = attribute_lr(lr_model=lr_model, X_test=X_test)

    summed = attr.mean_abs_per_raw_feature
    per_basis = dict(
        zip(
            attr.basis_feature_names,
            (
                float(np.mean(np.abs(attr.shap_per_basis[:, j])))
                for j in range(len(attr.basis_feature_names))
            ),
            strict=True,
        )
    )

    _save_fig(
        lr_summed_vs_basis_bar(
            summed_per_raw=summed,
            per_basis=per_basis,
            title=f"lr | {held_out_source} | summed vs per-basis",
        ),
        cfg.figures_dir / f"lr_{held_out_source}_summed_vs_basis.png",
    )
    return {
        "summed_per_raw": summed,
        "per_basis": per_basis,
        "intercept": attr.intercept,
    }


def _resolve_lr(obj: Any) -> LRModel:
    """Walk calibration wrappers to find the underlying LRModel."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if isinstance(obj, LRModel):
        return obj
    if isinstance(obj, CalibratedClassifierCV):
        inner = obj.calibrated_classifiers_[0].estimator
        if isinstance(inner, FrozenEstimator):
            return _resolve_lr(inner.estimator)
        return _resolve_lr(inner)
    if isinstance(obj, FrozenEstimator):
        return _resolve_lr(obj.estimator)
    raise TypeError(f"could not extract LRModel from {type(obj).__name__}")


# ----------------------------------------------------------------- helpers


def _save_fig(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _drift_to_dict(drift: SubgroupDriftResult) -> dict[str, object]:
    return {
        "grouping_name": drift.grouping_name,
        "overall_n": drift.overall_n,
        "overall_mean_abs_per_feature": drift.overall_mean_abs_per_feature,
        "by_stratum": [
            {
                "stratum": s.stratum,
                "n": s.n,
                "mean_abs_per_feature": s.mean_abs_per_feature,
                "delta_per_feature": s.delta_per_feature,
            }
            for s in drift.by_stratum
        ],
        "skipped_strata": [{"stratum": s, "n": n} for s, n in drift.skipped_strata],
    }


def _archetype_to_dict(arch: Archetype, X_test: pd.DataFrame) -> dict[str, object]:
    feats_df = archetype_features_to_dataframe(archetype=arch, X_test=X_test)
    feats: dict[str, object] = {}
    for c in feats_df.columns:
        v = feats_df[c].iloc[0]
        if isinstance(v, float) and not np.isfinite(v):
            feats[c] = None
        elif isinstance(v, (np.floating, float)):
            feats[c] = float(v)
        elif isinstance(v, (np.integer, int)):
            feats[c] = int(v)
        else:
            feats[c] = str(v)
    return {
        "label": arch.label,
        "test_index": arch.test_index,
        "y_true": arch.y_true,
        "y_proba": arch.y_proba,
        "features": feats,
    }


def _agreement_to_dict(agreement: AgreementResult) -> dict[str, object]:
    return {
        "model_names": list(agreement.model_names),
        "feature_names": list(agreement.feature_names),
        "spearman_matrix": agreement.spearman_matrix.tolist(),
    }


def _to_json_safe(obj: object) -> object:
    """Coerce NaN/inf to None for strict JSON output."""
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


# ----------------------------------------------------------------- driver


def run(cfg: ExplainConfig) -> dict[str, object]:
    """End-to-end explainability driver. Returns the aggregate report dict."""
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

    per_cell_blocks: list[dict[str, object]] = []
    per_fold_agreement: list[AgreementResult] = []

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
        y_test = y_full[fold.test_idx]
        sex_test = sex_full[fold.test_idx]
        age_test = age_full[fold.test_idx]

        encoded_space = fit_encoder(X_train)

        per_model_means: dict[str, dict[str, float]] = {}
        for model_name in MODEL_NAMES:
            cell_block = _explain_one_model_one_fold(
                model_name=model_name,
                held_out_source=fold.held_out_source,
                encoded_space=encoded_space,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                sex_test=sex_test,
                age_test=age_test,
                cfg=cfg,
            )
            per_cell_blocks.append(cell_block)
            global_importance = cast("dict[str, float]", cell_block["global_importance"])
            per_model_means[model_name] = dict(global_importance)

        # Per-fold cross-model agreement matrix.
        agreement = compute_cross_model_agreement(mean_abs_per_model=per_model_means)
        per_fold_agreement.append(agreement)
        _save_fig(
            cross_model_agreement_heatmap(
                agreement=agreement,
                title=f"{fold.held_out_source} | cross-model agreement (Spearman)",
            ),
            cfg.figures_dir / f"{fold.held_out_source}_cross_model_agreement_heatmap.png",
        )

    aggregate_block: dict[str, object] = {}
    if per_fold_agreement:
        aggregate = aggregate_across_folds(per_fold=per_fold_agreement)
        aggregate_block = _agreement_to_dict(aggregate)
        _save_fig(
            cross_model_agreement_heatmap(
                agreement=aggregate,
                title="aggregate (mean across folds) | cross-model agreement (Spearman)",
            ),
            cfg.figures_dir / "aggregate_cross_model_agreement_heatmap.png",
        )

    per_cell_path = cfg.reports_dir / "explanations_per_cell.json"
    aggregate_path = cfg.reports_dir / "explanations_aggregate.json"
    cross_model_path = cfg.reports_dir / "cross_model_agreement.json"

    per_cell_path.write_text(
        json.dumps(_to_json_safe(per_cell_blocks), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    cross_model_path.write_text(
        json.dumps(
            _to_json_safe(
                {
                    "per_fold": [_agreement_to_dict(a) for a in per_fold_agreement],
                    "aggregate": aggregate_block,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    report = {
        "config": {
            "smoke": cfg.smoke,
            "background_k": cfg.background_k,
            "nsamples": cfg.nsamples,
            "min_stratum_size": cfg.min_stratum_size,
            "max_test_rows": cfg.max_test_rows,
            "seed": cfg.seed,
            "n_folds_cap": cfg.n_folds_cap,
        },
        "n_cells": len(per_cell_blocks),
        "cross_model_aggregate_spearman": aggregate_block,
    }
    aggregate_path.write_text(
        json.dumps(_to_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    logger.info("wrote %s, %s, %s", per_cell_path, aggregate_path, cross_model_path)
    return report


# ----------------------------------------------------------------- CLI


def build_argparser() -> Any:
    """argparse parser shared between the module's ``main`` and the CLI script."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 fold, tiny KernelSHAP budget. CI default.",
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
    p.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help=(
            "Override the per (model x fold) KernelSHAP test-row cap. "
            "Defaults to 80 (full mode) / 30 (smoke mode) per ADR-013 "
            "amendment. Set to 0 to explain every test row (much slower; "
            "expect ~4x wall-clock on full mode)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def resolve_config(args: Any) -> ExplainConfig:
    base = (
        smoke_config(data_path=args.data_path)
        if args.smoke
        else full_config(data_path=args.data_path)
    )
    if args.max_test_rows is None:
        return base
    if args.max_test_rows < 0:
        raise ValueError(
            f"--max-test-rows must be >= 0 (0 means 'explain everything'); got {args.max_test_rows}"
        )
    # 0 = caller wants every test row; we pass a sentinel large enough that
    # _stratified_test_sample's "max_rows >= len(y_test)" branch always trips.
    effective = 10**9 if args.max_test_rows == 0 else args.max_test_rows
    return ExplainConfig(
        smoke=base.smoke,
        background_k=base.background_k,
        nsamples=base.nsamples,
        min_stratum_size=base.min_stratum_size,
        max_test_rows=effective,
        data_path=base.data_path,
        models_dir=base.models_dir,
        reports_dir=base.reports_dir,
        figures_dir=base.figures_dir,
        seed=base.seed,
        n_folds_cap=base.n_folds_cap,
    )


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # shap's KernelExplainer logs intermediate coalition-weight arrays at
    # the root logger's INFO level, which floods the run log with ~50
    # lines per explanation. Suppressed unless --verbose is on.
    if not args.verbose:
        logging.getLogger("shap").setLevel(logging.WARNING)
    cfg = resolve_config(args)
    logger.info("config: %s", cfg)
    run(cfg)
