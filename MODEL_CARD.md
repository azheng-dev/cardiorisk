# Model Card — CardioRisk Co-Pilot v1 risk models

> **Reading order.** This card is the user-facing summary of [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) (Phase 2.3b + 2.4 LODO results), [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md) (Honours-baseline honesty discussion), and [`docs/research/10-explainability.md`](docs/research/10-explainability.md) (Phase 2.5 KernelSHAP + cross-model agreement). Read those for the per-fold tables, per-subgroup audit, decision-curve analysis, bootstrap CIs, and per-(model × fold) SHAP figures.
>
> **Status.** Phase 2.5 deliverable. v1 = the four-model risk-prediction stack the rest of the CardioRisk Co-Pilot system is built on. Numbers below are produced verbatim by `backend/scripts/train_v1.py` and `backend/scripts/compute_explanations.py` from `data/processed/combined.parquet` (the Heart Failure Prediction dataset's underlying UCI sources combined under the HFP schema).
>
> **TL;DR.** The cardiovascular-risk module ships **four** binary classifiers — TabICL (TFM), L1 LR with restricted-cubic-spline expansion, XGBoost, and a faithful PyTorch port of the Honours team's 4-net mean-averaged Ensemble — evaluated under Leave-One-Domain-Out CV across the four UCI sources, with post-hoc calibration on a within-fold calibration slice, bootstrap CIs, subgroup audits, decision-curve analysis at the AusCVDRisk thresholds, and KernelSHAP-headline cross-model explainability with TreeSHAP / analytic-LR sanity checks. **TabICL is the headline model by AUROC, AUPRC, Brier, and calibration slope. L1 LR is the strongest white-box.** XGBoost suffers from isotonic-on-small-slice calibration collapse (slope 0.21). The Honours-Ensemble is reproduced honestly — without the WOA feature-selection layer (because the WOA code is not in the supplied archive); see §3 below and [ADR-012](docs/adr/012-honours-baseline-reproduction.md). The four models **agree on feature importance** at aggregate Spearman ρ ≥ 0.81 across all six pairwise comparisons (§5).

---

## 1. Intended use

This is a **research artefact, not a clinical product.** It exists to demonstrate that the author can build, evaluate, and honestly report on a clinical-domain ML system. It is *not* approved as a medical device, has not been validated in a clinical setting, and must not be used for real patient care.

**Intended downstream consumers:**

- The CardioRisk Co-Pilot agentic system (this repo, Phase 3+), which uses one of the trained models as the risk-score component of a larger explainability + retrieval + drafting pipeline.
- Recruiters / contributors auditing this repo's modelling work.

**Not intended for:**

- Real patient care, EHR integration, or clinical decision support.
- Populations the LODO procedure does not cover (the four UCI sources: Cleveland, Hungarian, LongBeachVA, Switzerland) — particularly any non-European, non-North-American cohort, or a population with a substantially different prevalence profile.
- The LongBeachVA ≥70 stratum, which is structurally under-served by every v1 model under our LODO protocol (see §8 below).

## 2. Models

| Slot | Model | Purpose | Calibration |
|---|---|---|---|
| Headline | **TabICL 2.1** ([Inria Soda](https://github.com/soda-inria/tabicl), BSD-3) | Tabular Foundation Model; zero-shot in-context learning. | Native (passes through unwrapped — calibrated by training objective). |
| Workhorse | **XGBoost 3.x** + Optuna (50 trials / 10-min cap) | White-box gradient boosted trees; the standard tabular baseline. | Isotonic on within-fold calibration slice. |
| Transparency anchor | **L1 LR with RCS** (saga, GridSearchCV C∈{0.001..100}) | Restricted-cubic-spline expansion of continuous features → linear model with nonzero coefficients only. The clinically interpretable model. | Sigmoid (Platt). |
| Honours baseline | **Ensemble** (PyTorch port of Honours `Demos/Data_Pre-processing.ipynb` cell 55) | 4 parallel sub-networks (DNN + 1D CNN + LSTM + BiLSTM), mean-averaged sigmoid output. The Honours team's architecture, faithfully reproduced. | Sigmoid (Platt). |

Wrapper code: [`cardiorisk/models/`](backend/cardiorisk/models/). All four wrappers implement the same `ModelWrapper` Protocol so the training driver and downstream consumers don't branch per model.

## 3. Headline results — LODO across the four UCI sources

**Methodology.** Leave-One-Domain-Out CV: each of the four UCI sources (Cleveland, Hungarian, LongBeachVA, Switzerland) is held out in turn while the model trains on the other three. Within each LODO fold, the training pool is split 80/10/10 into train / calibration / [unused — held for inner-CV in LR + Optuna in XGBoost]. Headline metrics: AUROC, AUPRC, Brier ↓, calibration slope (ideal=1), sensitivity at 85% / 90% specificity. Bootstrap CIs (2,000 percentile resamples) are reported per fold in [`reports/v1/metrics_per_fold.json`](reports/v1/metrics_per_fold.json).

| Model | AUROC | AUPRC | Brier ↓ | Calib. slope | Sens@85% | Sens@90% |
|---|---|---|---|---|---|---|
| TabICL | **0.811 ± 0.085** | **0.891 ± 0.055** | **0.150 ± 0.016** | 0.97 ± 0.30 | 0.567 ± 0.215 | 0.437 ± 0.277 |
| LR (L1+RCS) | 0.804 ± 0.082 | 0.883 ± 0.063 | 0.194 ± 0.037 | 0.75 ± 0.34 | **0.589 ± 0.136** | **0.457 ± 0.226** |
| XGBoost | 0.779 ± 0.081 | 0.826 ± 0.102 | 0.218 ± 0.041 | 0.21 ± 0.30 | 0.186 ± 0.195 | 0.103 ± 0.183 |
| Ensemble *(Honours architecture)* | 0.792 ± 0.076 | 0.860 ± 0.071 | 0.197 ± 0.024 | **1.02 ± 0.48** | 0.585 ± 0.138 | 0.370 ± 0.258 |

Mean ± standard deviation across the four LODO folds, from `reports/v1/metrics_aggregate.json`. Per-fold values + bootstrap CIs are in `reports/v1/metrics_per_fold.json`.

### Per-source breakdown (AUROC)

| Held-out source | TabICL | LR | XGBoost | Ensemble |
|---|---|---|---|---|
| Cleveland (n=303, prev=0.46) | **0.877** | 0.863 | 0.838 | 0.832 |
| Hungarian (n=294, prev=0.36) | **0.893** | 0.886 | 0.859 | 0.877 |
| LongBeachVA (n=200, prev=0.75) | 0.740 | 0.733 | 0.702 | **0.745** |
| Switzerland (n=123, prev=0.94) | **0.736** | 0.733 | 0.717 | 0.714 |

Full per-fold tables (including AUPRC, Brier, calibration, sensitivity) are in [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) §2. The qualitative reading consistent across all four models:

- **Cleveland and Hungarian** are the easiest folds. TabICL leads, all four are within ~5pp.
- **LongBeachVA** — every model loses ~10–15pp AUROC vs. the easier folds. Highest cholesterol-missingness, prevalence inversion vs. the training pool. The Honours-Ensemble *just* edges TabICL on AUROC here (0.745 vs 0.740) and is the only model whose net benefit at the 10% threshold beats treat-all (DCA §7); both within bootstrap noise.
- **Switzerland** — degenerate-by-design (negative class is borderline absent). All metrics are noisy on this fold; we do not drop it from LODO.

## 4. Subgroup audit

Per-(model × stratum) AUROC for sex and age band, with `min_stratum_size` guarding against meaningless small strata. Full table in [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) §3. Headlines:

- **Sex (F vs M):** Cleveland and Hungarian are the only auditable folds (LongBeachVA has F=6, Switzerland F=10 — both below the guard). Within auditable folds, Cleveland favours F slightly across every model (gaps 0.04–0.08); Hungarian favours M across every model (gaps 0.04–0.14). The Honours-Ensemble has the **largest Hungarian sex gap (0.142)** of the four — TabICL 0.099, LR 0.054, XGBoost 0.037 — flagged as the strongest single subgroup-audit reason not to prefer the Ensemble for deployment.
- **Age band (<50 / 50–69 / ≥70):** the ≥70 stratum on LongBeachVA (n=16) is the structural weak spot for **every v1 model.** AUROC there: TabICL 0.464, LR 0.536, XGBoost 0.518, Ensemble 0.393. The Ensemble is the worst on this stratum and posts the largest cross-stratum gap on LongBeachVA (0.440). The Honours architecture does not close this gap — it widens it. **The LongBeachVA ≥70 stratum is therefore declared out-of-scope for any deployment use of these four models** (see §8).

## 5. Explainability (Phase 2.5)

KernelSHAP is the cross-model headline (per [ADR-013](docs/adr/013-explainability-strategy.md)) — same algorithm, same background distribution (`shap.kmeans(50)`), same coalition budget (128) across all four models so the resulting feature attributions are commensurable. Native fast-path attributions (TreeSHAP for XGBoost; analytic LR-coefficient sum-back) run as sanity checks. Full discussion + per-(model × fold) figures: [`docs/research/10-explainability.md`](docs/research/10-explainability.md).

### Top-5 cross-fold-averaged feature importance per model

Mean |SHAP value| (probability space), averaged across the four LODO folds. Numbers from `reports/v1/explainability/explanations_per_cell.json`.

| Rank | TabICL | XGBoost | LR | Ensemble |
|---|---|---|---|---|
| 1 | **ChestPainType (0.112)** | **ChestPainType (0.144)** | **ChestPainType (0.122)** | **ChestPainType (0.104)** |
| 2 | FastingBS (0.047) | FastingBS (0.072) | MaxHR (0.069) | ST_Slope (0.055) |
| 3 | ExerciseAngina (0.045) | Oldpeak (0.065) | Oldpeak (0.062) | Oldpeak (0.040) |
| 4 | ST_Slope (0.041) | MaxHR (0.053) | ExerciseAngina (0.045) | ExerciseAngina (0.039) |
| 5 | Oldpeak (0.037) | ExerciseAngina (0.044) | ST_Slope (0.043) | Sex (0.037) |

`ChestPainType` is universally rank-1 across all four models on all four folds (the asymptomatic level dominates the contribution — see the LR per-spline-basis figures). `Cholesterol` is bottom-half for every model, an artefact of Switzerland's 100% missingness on that field, not a model-emergent dismissal of the textbook risk factor.

### Cross-model agreement (Spearman rank correlation, aggregate over 4 folds)

|  | TabICL | XGBoost | LR | Ensemble |
|---|---|---|---|---|
| **TabICL** | 1.00 | 0.90 | 0.84 | 0.83 |
| **XGBoost** | 0.90 | 1.00 | 0.85 | 0.83 |
| **LR** | 0.84 | 0.85 | 1.00 | 0.81 |
| **Ensemble** | 0.83 | 0.83 | 0.81 | 1.00 |

All six pairwise correlations ≥ 0.81 — the four models substantially agree on which features matter, the precondition the Phase 3 agentic system needs in order to draft a coherent risk-driver narrative on top of any one of them. TabICL ↔ XGBoost (0.90) is the closest pair despite maximally different inductive biases; the Honours-Ensemble disagrees most with the others (0.81–0.83), driven by its higher emphasis on `ST_Slope` and the `ST_Slope_was_missing` indicator. Visualisation: [`reports/v1/figures/explainability/aggregate_cross_model_agreement_heatmap.png`](reports/v1/figures/explainability/aggregate_cross_model_agreement_heatmap.png).

### Sanity checks: KernelSHAP vs native attribution

Spearman rank correlation between the cross-model KernelSHAP feature ranking and the native fast-path ranking (TreeSHAP for XGBoost; LR-coefficient × spline expansion summed back to raw HFP features for LR). Cell-by-cell numbers from `explanations_per_cell.json`:

| Fold | XGBoost (KernelSHAP vs TreeSHAP) | LR (KernelSHAP vs analytic-summed) |
|---|---|---|
| Cleveland | 0.91 | 0.93 |
| Hungarian | 0.96 | 0.89 |
| LongBeachVA | 0.93 | 0.93 |
| Switzerland | 0.98 | 0.89 |
| **Mean** | **0.95** | **0.91** |

ADR-013's "trigger to revisit" was set at 0.7; the actual numbers are far above that. The cross-model KernelSHAP comparison is not an algorithm artefact.

### Subgroup-feature drift (auditable strata only)

Per ADR-013 §4 + [`07-eval-design.md`](docs/research/07-eval-design.md) §5, per-stratum mean |SHAP| deltas are computed only on strata with `n ≥ 30`. The structural finding: **no LODO fold has an auditable F sex-stratum.** F counts in the per-fold KernelSHAP test slices: Cleveland ≈ 19–24, Hungarian ≈ 17–22, LongBeachVA ≈ 2–4, Switzerland ≈ 7–8 — all below the guard. We do not report per-feature drift between sexes because we cannot do so honestly at this sample size. The discrimination-side analogue is §4 above (sex AUROC reported only for Cleveland and Hungarian); the explainability-side analogue is here. Where the guard passes (M-stratum always; 50–69 age band always; <50 age band on Hungarian only), per-(model × fold × stratum) bars are under [`reports/v1/figures/explainability/<model>_<fold>_subgroup_drift_*.png`](reports/v1/figures/explainability/) and are uniformly small (≤ 0.02 per-feature delta vs the per-fold average).

### Local explanations: 64 archetype waterfalls

Four representative test patients per (model × fold) — TP-high, TP-low, FN, FP, picked deterministically from the per-fold test slice — are rendered as SHAP waterfall plots. Total 64 PNGs (4 archetypes × 4 folds × 4 models) under [`reports/v1/figures/explainability/<model>_<fold>_<archetype>_waterfall.png`](reports/v1/figures/explainability/). The archetype rows are always included in the KernelSHAP test-row cap so the waterfalls reflect actual archetype patients, not stand-ins.

### Methodological caveats

The wall-clock cost of KernelSHAP forced a contingency at execution time (documented inline in [`10-explainability.md`](docs/research/10-explainability.md) §1 and in the [ADR-013 amendment](docs/adr/013-explainability-strategy.md)): the per-(model × fold) test slice was capped at 80 rows (stratified-sampled, archetype rows always preserved) instead of explaining every test row. This inflates the standard error of mean |SHAP| by roughly 1.2×–1.9× depending on fold; the rank-based cross-model agreement above is essentially unaffected. The full per-fold slice is recoverable via `compute_explanations.py --max-test-rows 0` (~4× wall-clock).

## 6. Calibration story (read carefully)

Phase 2.3b's empirical finding ([`08-v1-model-results.md`](docs/research/08-v1-model-results.md) §4): the within-fold calibration slice sits at ~50–100 rows per LODO fold. **Isotonic regression on this slice size collapses XGBoost** (cross-fold mean calibration slope = 0.21, vs ideal=1). The collapse manifests as a near-flat reliability curve and zero/near-zero sensitivity at high specificity. This is *not* a bug in our XGBoost wrapper — it is a known property of isotonic regression on small calibration sets.

Two consequences for the model card:

1. **XGBoost should not be used at clinically meaningful operating points (e.g. sens@85% / sens@90%).** Its uncalibrated discrimination is fine; its calibrated probabilities are not.
2. **Sigmoid (Platt) calibration is robust at this slice size.** Both LR and the Ensemble use it — the Ensemble explicitly per [ADR-012](docs/adr/012-honours-baseline-reproduction.md), to avoid replicating the XGBoost failure mode on the same slice size. Niculescu-Mizil & Caruana (2005) shows Platt scaling outperforms isotonic at n < ~1000 calibration rows.

If the deployment context allows for a larger calibration slice (e.g. an Australian-cohort retraining where ≥1000 calibration rows are available), isotonic for XGBoost becomes viable. The current artefact does not.

## 7. Decision-curve analysis at AusCVDRisk thresholds

DCA at 5% and 10% (the AusCVDRisk treatment thresholds, [`07-eval-design.md`](docs/research/07-eval-design.md) §6). Headline: models add net benefit over treat-all in **moderate-prevalence settings (Cleveland, Hungarian)** but not in **very-high-prevalence settings (Switzerland)**, where treat-all dominates because the negative-class base rate is too low for any model's predicted-low-risk subgroup to deliver clinical value vs. simply treating everyone. **LongBeachVA at 10%** is borderline: the Honours-Ensemble is the only model whose net benefit (+0.7178) exceeds treat-all (+0.7167); LR and XGBoost are *worse* than treat-all on this fold; TabICL ties. The gap is well within bootstrap noise.

This is honest information about the *data*, not a verdict on the *models*. Full per-fold DCA values in [`reports/v1/metrics_per_fold.json`](reports/v1/metrics_per_fold.json) under each `dca` block; visualisations under [`reports/v1/figures/`](reports/v1/figures/).

## 8. Drift monitoring (Phase 2.6)

Per-feature input-drift PSI + prediction-drift PSI for every (model × LODO fold) cell, computed by `backend/scripts/compute_drift.py` against the in-fold training-pool combined distribution as the reference. Each fold's held-out source is used as the "current" slice — i.e. the drift number quantifies how different the data is from what the fold's model was actually fit on. Severity bands per [ADR-014](docs/adr/014-drift-monitoring.md) (industry convention; not validated for this dataset): `< 0.10` stable / `0.10 – 0.25` moderate / `>= 0.25` major.

**Per-feature drift is identical across models within a fold** (input drift is a property of the data, not the model). Severity counts (out of 11 features per fold):

| Held-out source | stable | moderate | major | Top-3 drifted features |
|---|---:|---:|---:|---|
| Cleveland | 4 | 2 | 5 | `ST_Slope` (PSI=7.06), `RestingECG` (1.84), `ExerciseAngina` (1.13) |
| Hungarian | 3 | 2 | 6 | `Age` (2.23), `Oldpeak` (1.42), `ST_Slope` (1.41) |
| LongBeachVA | 2 | 1 | 8 | `ExerciseAngina` (1.77), `Age` (0.78), `MaxHR` (0.68) |
| Switzerland | 2 | 3 | 6 | `MaxHR` (0.60), `ChestPainType` (0.54), `Oldpeak` (0.42) |

**Prediction-drift PSI varies dramatically across models under the same input drift.** The two foundation-model-style learners (TabICL, Honours-Ensemble) translate cross-source covariate shift into 3–4× larger predicted-probability shifts than the calibrated tree (XGBoost) and linear (LR) models:

| Model | mean prediction-PSI (across 4 folds) | max prediction-PSI |
|---|---:|---:|
| **TabICL** | 1.57 | 2.94 (LongBeachVA) |
| **Honours-Ensemble** | 1.24 | 1.72 (Switzerland) |
| **XGBoost** | 0.44 | 0.68 (Switzerland) |
| **L1 LR** | 0.40 | 0.58 (Switzerland) |

Operational reading: if any of these models were deployed, **TabICL and the Ensemble would warrant more aggressive prediction-drift monitoring than XGBoost and LR**. The narrower XGBoost/LR ranges are partly because their post-hoc calibration mappings absorb some of the covariate-shift signal — a known effect, not a free pass on stability.

**The methodological caveats are honest about what this metric does and doesn't see** (full discussion in [`docs/research/11-drift-design.md`](docs/research/11-drift-design.md) §5):

- PSI is per-feature; joint-distribution shifts are invisible to it. Multivariate drift (MMD, domain-classifier) is deferred to a productionisation phase.
- Severity bands are industry convention, not derived from this dataset.
- 10 quantile bins fixed across the sweep; PSI is bin-count-sensitive.
- KS p-values in the JSON report are *sanity-only* — they use bin-midpoint reconstruction of the reference, not raw reference samples.
- No time component (single point-in-time PSI; no rolling-window series).
- No concept drift (would require labelled new data; deferred).

Reproduce: `uv run --project backend python backend/scripts/compute_drift.py` (full sweep ~30 s) writes [`reports/v1/drift/per_fold.json`](reports/v1/drift/per_fold.json), [`reports/v1/drift/aggregate.json`](reports/v1/drift/aggregate.json), and 16 dashboard PNGs (one per model × fold) to [`reports/v1/figures/drift/`](reports/v1/figures/drift/). Each dashboard shows the PSI bar across all features, an ECDF overlay for the top-3 drifted numerics, and a `predict_proba` histogram overlay.

## 9. Limitations & out-of-scope

- **Not validated in a clinical setting.** No clinician-in-the-loop study, no real-EHR integration, no deployment.
- **Trained on small, biased UCI sources.** ~920 rows total across four heterogeneous sources, none of which is contemporary Australian primary-care data. Generalisability to any cohort outside these four sources is *unverified*.
- **LongBeachVA ≥70 stratum is structurally under-served.** Every v1 model loses meaningful AUROC on that stratum. The model card flags this explicitly; any deployed surface must either exclude this stratum or carry a "low-confidence" UI affordance.
- **No external validation cohort.** The LODO protocol simulates a deployment-like scenario (test on a source the model has never seen) but does not replace external validation on a non-UCI cohort.
- **No fairness audit beyond sex + age band.** Race, ethnicity, socioeconomic status, geographic strata: not in the data, not auditable.
- **No auditable F sex-stratum subgroup-feature-drift on any LODO fold.** §5 above. We do not report sex-based feature-importance drift because every fold's F count is below the `n ≥ 30` honesty guard.
- **Calibration slice size limits XGBoost.** See §6.
- **KernelSHAP test-slice cap (80 rows per model × fold).** See §5 "Methodological caveats" — inflates the standard error of mean |SHAP| by ~1.2×–1.9×; rank-based cross-model agreement is essentially unaffected. Recoverable via `--max-test-rows 0`.
- **The Honours-Ensemble row is the architecture only, not the WOA-Ensemble pipeline** (see §3 of [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md)). The WOA feature-selection layer that produced the Honours report's headline number is not in the supplied archive; we did not invent one. Reading the Honours-Ensemble row as if it is "WOA-Ensemble under our protocol" is incorrect.

## 10. Honesty caveats specific to the Honours-Ensemble row

The Honours team's Final Report §7.2 Table 2.2 reports WOA-Ensemble on HFP at sensitivity 89.72%, specificity 83.12% (single 80/20 split, no CV, no per-source breakdown, no calibration). Our table reports a different number under a different protocol (4-fold LODO, calibrated, sensitivity at 85% / 90% specificity not at the default 0.5 threshold, no WOA layer). A direct numerical comparison ("89.72% vs our X%") is misleading. The right reading is qualitative: the Honours architecture's relative position against the v1 trio under a fair LODO protocol. Full discussion in [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md).

## 11. Reproducibility

- All code: this repo, MIT-licensed.
- Run training (full LODO, ~40 min on a recent CPU): `uv sync --project backend && uv run --project backend python backend/scripts/train_v1.py`. `--smoke` for a 1-fold synthetic-data smoke pass (~70s).
- Run explainability sweep (~2h 20m on a recent CPU): `uv run --project backend python backend/scripts/compute_explanations.py`. `--smoke` for the 1-fold smoke pass (~30s); `--max-test-rows 0` to explain every per-fold test row (~4× wall-clock).
- Run drift sweep (~30s on a recent CPU): `uv run --project backend python backend/scripts/compute_drift.py`. `--smoke` for the 1-fold smoke pass (~10s).
- Model artefacts: not in git (per [ADR-010](docs/adr/010-model-artefact-storage.md)); regenerated locally by the training script above.
- Reports: in git under [`reports/v1/`](reports/v1/) — `metrics_per_fold.json`, `metrics_aggregate.json`, `explainability/{explanations_per_cell,explanations_aggregate,cross_model_agreement}.json`, `drift/{per_fold,aggregate}.json`, and `figures/**/*.png`.
- Determinism: every wrapper pins `random_state` / `torch.manual_seed` to the project seed (20260505). XGBoost reproduces to ~1e-6 across runs; the PyTorch Ensemble reproduces to ~1e-5 (deterministic-algorithms flag deliberately not enabled — see [ADR-012](docs/adr/012-honours-baseline-reproduction.md)). KernelSHAP per-row values reproduce to ~1e-5; aggregate quantities (mean |SHAP|, Spearman ranks) to ~1e-6. Drift PSI is closed-form on binned histograms and reproduces exactly bit-for-bit modulo numpy floating-point ordering.
- CI: all three smoke runs are enforced on every PR (`train_v1.py --smoke`, `compute_explanations.py --smoke`, and `compute_drift.py --smoke` steps in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 12. References

- [TabICL (Inria Soda)](https://github.com/soda-inria/tabicl) — BSD-3-Clause.
- [XGBoost](https://xgboost.readthedocs.io/) — Apache-2.0.
- [scikit-learn](https://scikit-learn.org/) — BSD-3.
- [PyTorch](https://pytorch.org/) — BSD-style.
- Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. (Cited only as the WOA spec the Honours report claims to follow; not implemented in v1.)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. (Calibration recipe rationale.)
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. (DCA methodology.)
- Honours Final Report (2024S1-1698) — the prior study this v1 work supersedes.

Architecture decisions:

- [ADR-006](docs/adr/006-risk-model-architecture.md) — risk-model architecture.
- [ADR-008](docs/adr/008-preprocessing-pipeline.md) — preprocessing pipeline.
- [ADR-009](docs/adr/009-eval-harness.md) — eval harness.
- [ADR-010](docs/adr/010-model-artefact-storage.md) — model artefact storage.
- [ADR-011](docs/adr/011-tfm-tabicl-supersedes-tabpfn.md) — TabICL supersedes TabPFN.
- [ADR-012](docs/adr/012-honours-baseline-reproduction.md) — Honours-baseline reproduction (Phase 2.4).
- [ADR-013](docs/adr/013-explainability-strategy.md) — explainability strategy (Phase 2.5; with 2026-05-06 amendment recording the wall-clock contingency).
- [ADR-014](docs/adr/014-drift-monitoring.md) — drift / monitoring strategy (Phase 2.6).
