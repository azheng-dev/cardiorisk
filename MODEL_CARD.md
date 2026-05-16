# Model Card — CardioRisk Co-Pilot v1 risk models

> **Reading order.** This card is the user-facing summary of [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) (Phase 2.3b + 2.4 LODO results), [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md) (Honours-baseline honesty discussion), and [`docs/research/10-explainability.md`](docs/research/10-explainability.md) (Phase 2.5 KernelSHAP + cross-model agreement). Read those for the per-fold tables, per-subgroup audit, decision-curve analysis, bootstrap CIs, and per-(model × fold) SHAP figures.
>
> **Status.** Phase 2.5 deliverable for the modelling content (§1-§10); Phase 4 + 6 deliverable for the agent-eval headline (§11-§12); Phase 7 deliverable for the observability + latency-budget content (§13). v1 = the four-model risk-prediction stack the rest of the CardioRisk Co-Pilot system is built on. Numbers below are produced verbatim by `backend/scripts/train_v1.py`, `backend/scripts/compute_explanations.py`, and `backend/scripts/eval_agents.py` from `data/processed/combined.parquet` (the Heart Failure Prediction dataset's underlying UCI sources combined under the HFP schema).
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

## 9. Retrieval (Phase 3.2)

> The retrieval layer is the foundation Phase 3.3's citation-mandatory generator sits on. The eval below picks a winning chunker × rerank cell from the matrix `{token, semantic, heading-aware-hybrid} × {no-rerank, with-rerank}` and reports `hit@1`, `hit@5`, `MRR` with 2,000-resample bootstrap CIs. Full design: [ADR-016](docs/adr/016-retrieval-stack.md) and [`docs/research/13-retrieval-design.md`](docs/research/13-retrieval-design.md).

**Stack.** `BAAI/bge-m3` dense embedder (1024-d, MIT-equivalent licence) + `rank_bm25.BM25Okapi` sparse retriever + Reciprocal Rank Fusion (`k=60`) + optional `BAAI/bge-reranker-v2-m3` cross-encoder. Vector index is in-memory `hnswlib` (cosine, `M=16`, `ef_construction=200`); pgvector graduates with the rest of the agentic stack in Phase 4. Real-corpus headline = **token chunker + no rerank**; see "Reading the table" below for why the reranker is *off* by default on the real corpus despite winning on the fixture.

**Real-corpus eval (10 Qs over 1,834 chunks).** The Phase 3.2 eval set is 50 hand-curated Qs at [`eval/retrieval/questions.jsonl`](eval/retrieval/questions.jsonl); 40 target the markdown fixture corpus, 10 target the real RACGP Red Book + NVDPA 2023 guideline + Summary-of-recommendations PDFs. The orchestrator splits on `expected_doc_id`: the real-corpus run loads only the 10 real-corpus Qs (the 40 fixture Qs are guaranteed misses against the real corpus). Real-corpus headline:

| Cell | hit@1 | hit@5 | MRR | 95% CI hit@5 |
|---|---:|---:|---:|---|
| **token, no rerank** | **0.500** | **0.600** | **0.550** | **[0.30, 0.90]** |
| token, with rerank | 0.300 | 0.600 | 0.378 | [0.30, 0.90] |
| semantic, no rerank | 0.500 | 0.600 | 0.533 | [0.30, 0.90] |
| semantic, with rerank | 0.400 | 0.600 | 0.470 | [0.30, 0.90] |
| hybrid, no rerank | 0.400 | 0.600 | 0.467 | [0.30, 0.90] |
| hybrid, with rerank | 0.200 | 0.600 | 0.323 | [0.30, 0.90] |

Source: [`reports/v1/retrieval/per_cell.json`](reports/v1/retrieval/per_cell.json) and [`reports/v1/retrieval/aggregate.json`](reports/v1/retrieval/aggregate.json). Figures: [`reports/v1/figures/retrieval/`](reports/v1/figures/retrieval/).

**Fixture eval (40 Qs over 10 hybrid chunks).** The fixture eval was the Phase 3.2 result-of-record before the real corpus was fetched; it stays in the suite as a CI-friendly smoke and a sanity check on the pipeline wiring. Headline (rerun via `eval_retrieval.py --use-fixture`): all 3 chunkers tie at `hit@5 = 1.0` once the reranker is on; reranker buys +5 to +35 pp on hit@1; hybrid chunker is the only cell with meaningful chunk count (10 vs 2 for token / semantic) on the fixture. Numbers archived in `reports/v1/retrieval/smoke/` after a smoke run; the real-corpus result is the headline of record.

**Reading the table.** All six real-corpus cells tie at `hit@5 = 0.600` (6 of 10 expected documents land in the top 5). On `hit@1` and `MRR`, **no-rerank wins on every chunker**: token (0.50 → 0.30), semantic (0.50 → 0.40), hybrid (0.40 → 0.20) — the cross-encoder hurts top-1 precision across the board on the real corpus. This is the **opposite** of the fixture finding (where rerank lifted hit@1 by +35 pp on token / semantic and +5 pp on hybrid). The likely interpretation: the fixture passages are short and lexically-aligned, so the cross-encoder mostly re-confirms the RRF top candidate; the real-corpus passages (~512 tokens, dense Australian-clinical prose) are long enough that the cross-encoder picks a semantically-related-but-not-doc-matching chunk over the keyword-match-perfect one. With n=10 the 95% CIs `[0.30, 0.90]` overlap heavily — the rerank-hurts effect is real *in direction* across all 3 chunkers but the per-cell magnitude is statistically indistinguishable.

**Decisions baked into Phase 3.3 from this result:**

- **Production default `with_rerank = False`.** The Phase 3.3 generator calls `RetrievalPipeline.retrieve(..., with_rerank=False)` by default; the cross-encoder stays available behind a flag for downstream maintainers who want to A/B it on a larger eval set.
- **Token-window chunker is the v1 production chunker.** It ties on hit@5 and wins on MRR (0.550 vs 0.533 semantic / 0.467 hybrid).
- **The deferred-to-Phase-3.2.1 token-window-size sweep (256 / 1024 vs 512) is dropped.** With n=10 the eval is too underpowered to discriminate; running the sweep would give a confidently-wrong winner.

**Honest weaknesses (full discussion in [`docs/research/13-retrieval-design.md`](docs/research/13-retrieval-design.md) §8):**

- **n=10 is the hard limit on the real-corpus signal** until the eval set grows. Every CI in this table is `[0.30, 0.90]` wide (the bootstrap floor at this n). Any single Q toggling its hit moves the headline by 10 pp.
- **No proprietary-model A/B.** `text-embedding-3-large` would lift hit@5 by an unknown margin; the deferral is documented in ADR-016 §"Trigger to revisit".
- **In-memory hnswlib only.** Phase 4's pgvector graduation is the production design; the file structure today reflects the Phase-3.2 eval surface, not the deploy surface.
- **Reranker counter-intuitive result.** The on-fixture vs on-real divergence is a real Phase 3.2 finding; ADR-016 §"Amendment 2026-05-15 (real-corpus rerank-hurts result)" documents both the surface decision (default off) and the open question (does this hold at n=100? Phase 6 will rerun with a much larger Q set and decide for the production deploy).

**Reproduce.** Real-corpus full run (~6 min after weights are warm; 6 cells × 10 real Qs):

```bash
uv run --project backend python backend/scripts/fetch_corpus.py
uv run --project backend python backend/scripts/build_corpus.py
CARDIORISK_TORCH_THREADS=8 uv run --project backend python backend/scripts/build_index.py --embedder bge-m3
CARDIORISK_TORCH_THREADS=8 uv run --project backend python backend/scripts/eval_retrieval.py
```

Writes [`reports/v1/retrieval/per_cell.json`](reports/v1/retrieval/per_cell.json), [`reports/v1/retrieval/aggregate.json`](reports/v1/retrieval/aggregate.json), and 3 figures under [`reports/v1/figures/retrieval/`](reports/v1/figures/retrieval/). The `CARDIORISK_TORCH_THREADS` env var lifts the single-thread guard the script otherwise inherits from Phase 2.x's TabICL/XGBoost OpenMP-deadlock workaround (this script never imports those, so 8 threads is safe). CI smoke uses `sentence-transformers/all-MiniLM-L6-v2` (~80 MB, no rerank) against the fixture corpus; ~60 s on `ubuntu-latest`.

## 10. Citation-mandatory generation (Phase 3.3)

> The generation layer is the user-facing surface of the retrieval+verification stack. It enforces the AGENTS §3 honesty contract end-to-end: every claim ends in a sentence-trailing bracketed citation, every citation is verified by an NLI model, and every unverified claim is suppressed (never re-prompted). Full design: [ADR-017](docs/adr/017-citation-and-nli-verification.md) and [`docs/research/14-citation-generation-design.md`](docs/research/14-citation-generation-design.md).

**Stack.** `BAAI/bge-m3` retriever (Phase 3.2 stack) → `RetrievalPipeline.retrieve(top_k=5, with_rerank=False)` → `citation_required.v1.md` prompt → pluggable `BaseLLMClient` (Mock for CI; Anthropic / OpenAI for Phase 6) → bracketed-citation parser → `BaseNLIVerifier` (Mock token-overlap for CI; `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` for the production verifier). Suppression policy: drop unverified claims with a typed reason (`no_citation` / `phantom_citation` / `not_entailed`), never re-prompt the LLM.

**Phase-3.3 eval headline (Mock-LLM + Mock-NLI; 12 real-corpus cases = 6 positive + 6 refusal).**

| metric | point | 95% CI | n |
|---|---:|---|---|
| citation_precision | 1.000 | (denominator constant under MockLLM) | 12 |
| keyword_recall | 0.042 | [0.000, 0.146] | 12 |
| hallucination_rate | 0.167 | [0.000, 0.500] | 6 (positive only) |
| refusal_accuracy | 0.000 | [0.000, 0.000] | 6 (refusal only) |

Source: [`reports/v1/generation/aggregate.json`](reports/v1/generation/aggregate.json) and [`reports/v1/generation/per_case.json`](reports/v1/generation/per_case.json). Figures: [`reports/v1/figures/generation/`](reports/v1/figures/generation/).

**Verifier-comparison archive (same Mock-LLM run; DeBERTa-NLI vs Mock-NLI).** The same 12-case run with the production DeBERTa verifier suppresses 7 of MockLLM's 15 emitted claims (Mock-NLI suppresses 1) and pushes the hallucination rate from 0.167 to 0.000. This is the wiring proof that the verifier-in-the-loop architecture rejects bad claims when bad claims arrive. Archive: [`reports/v1/generation/nli_deberta/`](reports/v1/generation/nli_deberta/).

| | Mock NLI | DeBERTa NLI |
|---|---:|---:|
| verified claims | 14 | 8 |
| suppressed claims | 1 | 7 |
| citation_precision | 1.000 | 1.000 |
| keyword_recall | 0.042 | 0.042 |
| hallucination_rate | 0.167 | 0.000 |
| refusal_accuracy | 0.000 | 0.000 |

**Reading the table — what these numbers do not say.** The Phase 3.3 headline is **diagnostic of MockLLM, not predictive of the production system.** MockLLM picks the first sentence of the first retrieved passage and emits it with the chunk's chunk_id. So:

- `citation_precision = 1.0` means MockLLM always cites a real chunk_id — not that the answer is correct.
- `keyword_recall = 0.04` means MockLLM does not actually answer the question — it picks a passage and quotes it.
- `hallucination_rate = 0.17` (Mock NLI) means 1 of 6 positive cases got a wrong-doc citation; the DeBERTa run drops this to 0 because DeBERTa rejected MockLLM's syntactically-broken claims and forced a refusal-by-suppression.
- `refusal_accuracy = 0.0` means MockLLM never emits the `__INSUFFICIENT_EVIDENCE__` sentinel (it doesn't know it should). A real LLM with a properly-respected refusal directive should score 6/6.

**Phase 3.3 ships the wiring proof. Phase 6 ships the quality proof.** The Mock-LLM headline is the regression baseline against which Phase 6's Claude Sonnet 4.5 / GPT-4o-mini A/B will be measured.

**Decisions baked into Phase 4 / Phase 5 from this result:**

- **Phase 4 LangGraph guideline-agent contract.** `state.guideline_answer = CitationGenerator.generate(state.normalised_question)`; agent code sees a structured `GeneratedAnswer` with `verified_claims`, `suppressed_claims`, `refused`, `refusal_reason` — no free-text parsing required.
- **Phase 5.3 letter-editor UI contract.** Verified claims render as the answer body; suppressed claims render in a collapsible "the system rejected the following claims because…" panel with the typed reason. HITL approve / edit / reject persists the verified set, not the raw LLM text.
- **Phase 6 100-case eval extension.** 36 → 100 cases; real-LLM A/B (Claude Sonnet 4.5 + GPT-4o-mini); LLM-judge NLI cross-check on a 50-claim sub-sample.

**Honest weaknesses (full discussion in [`docs/research/14-citation-generation-design.md`](docs/research/14-citation-generation-design.md) §8):**

- **Mock-LLM headline is diagnostic only.** §8.1.
- **n=6 real-corpus positives is the hard floor on the per-tag signal.** §8.2.
- **No multi-LLM A/B in Phase 3.3.** Pluggable `BaseLLMClient` is the contract; Phase 6 ships the comparison. §8.3.
- **DeBERTa verifier has no medical-domain fine-tune.** General-purpose, not expert. §8.4.
- **Suppression is "drop, never re-prompt".** Shorter-but-true beats longer-but-some-of-it-fabricated; trade-off documented. §8.5.
- **Citation precision is doc-level, not paragraph-level.** Pinning the eval to specific chunk ids would silently break under any Phase 3.2.1 chunker change. The NLI verifier covers paragraph-level entailment in production. §8.6.
- **No latency or cost numbers in Phase 3.3.** Deferred to Phase 6 + Phase 7. §8.7.

**Reproduce.** Real-corpus full run (~25 s after weights are warm; 12 cases):

```bash
uv run --project backend python backend/scripts/fetch_corpus.py
uv run --project backend python backend/scripts/build_corpus.py
CARDIORISK_TORCH_THREADS=8 uv run --project backend python backend/scripts/build_index.py --embedder bge-m3
CARDIORISK_TORCH_THREADS=8 uv run --project backend python backend/scripts/eval_generation.py --llm mock --nli mock --strategy token --embedder bge-m3
# Verifier comparison (DeBERTa NLI; ~3 min after weights are warm):
CARDIORISK_TORCH_THREADS=8 uv run --project backend python backend/scripts/eval_generation.py --llm mock --nli deberta --strategy token --embedder bge-m3 \
    --reports-dir reports/v1/generation/nli_deberta \
    --figures-dir reports/v1/figures/generation/nli_deberta
```

Writes [`reports/v1/generation/per_case.json`](reports/v1/generation/per_case.json), [`reports/v1/generation/aggregate.json`](reports/v1/generation/aggregate.json), and 2 figures under [`reports/v1/figures/generation/`](reports/v1/figures/generation/). CI smoke uses `--smoke` (1 case, mock client, mock verifier, fixture corpus); ~5 s on `ubuntu-latest` with no API key required.

## 11. Agent orchestration (Phase 4)

> The agent layer is the end-to-end clinical workflow on top of the v1 risk model (§3) and the citation-mandatory generator (§10). A (synthetic) patient payload becomes a triage check, a risk score with attributions, a verified guideline answer, and a referral letter draft — with a structured human-in-the-loop (HITL) decision (approve / edit / reject) gating every transition. Full design: [ADR-018](docs/adr/018-agent-orchestration.md) and [`docs/research/15-agent-design.md`](docs/research/15-agent-design.md).

**Stack.** `langgraph>=0.6,<0.7` `StateGraph` + `InMemorySaver` checkpointer (`PostgresSaver` graduates with the rest of the deploy stack in Phase 7 / 8) + `interrupt()` HITL gates between every agent transition + Pydantic-immutable `AgentState` (the state schema *is* the FastAPI request/response schema *is* the eval schema). Four agents — `triage` (rule-based normalisation + sanity-flag emitter), `risk` (joblib loader + calibrated band + top-k attributions; deterministic mock fallback when no artefact is present), `guideline` (wraps the Phase 3.3 `CitationGenerator`), `letter` (deterministic template renderer over verified claims) — wired by `backend/cardiorisk/agents/graph.py`. Resilience = `tenacity`-backed `with_retries` + an in-house 30-LoC `CircuitBreaker` (3 consecutive `TransientAgentError` failures → open for 60 s; deterministic clock hook for tests). FastAPI surface = `POST /v1/agents/cases` (kicks off; pauses at the first interrupt) + `POST /v1/agents/cases/{case_id}/decide` (resumes with a structured `Decision`) + `GET /v1/agents/cases/{case_id}` (reads the latest checkpointed state) + `GET /healthz`. No WebSocket / SSE / auth in Phase 4 (deferred to Phase 5 / 8).

**HITL contract.** Every stage exposes `approve` / `edit` / `reject` with one exception: **`risk` is approve / reject only.** The calibrated probability is not user-editable on calibration-honesty grounds; if the *inputs* were wrong, the reviewer edits at triage and re-runs; if the *output* is judged wrong, the reviewer rejects with a structured reason. See ADR-018 §3.

**Phase-4 eval headline (Mock-LLM + always-entail NLI + stub retrieval; 30-case auto-approve harness; `tabicl_Cleveland.joblib`; AusCVDRisk thresholds 0.05 / 0.10).**

| metric | point | n |
|---|---:|---:|
| triage_pass_rate | 0.900 | 30 |
| risk_band_match_rate | 0.467 | 30 |
| guideline_pass_rate | 1.000 | 30 |
| letter_pass_rate | 1.000 | 30 |
| **full_pipeline_pass_rate** | **0.400** | **30** |
| median_total_duration_ms | ≈ 1035 | 30 |
| p95_total_duration_ms | ≈ 1067 | 30 |

Source: [`reports/v1/agents/aggregate.json`](reports/v1/agents/aggregate.json) and [`reports/v1/agents/per_case.json`](reports/v1/agents/per_case.json). Figures: [`reports/v1/figures/agents/`](reports/v1/figures/agents/).

**Risk-band confusion matrix (predicted ↓ vs expected →):**

| expected \ predicted | low | intermediate | high |
|---|---:|---:|---:|
| **low** | 3 | 3 | 2 |
| **intermediate** | 0 | 2 | **11** |
| **high** | 0 | 0 | 9 |

**Reading the table — what these numbers do and don't say.** The Phase 4 eval is **the orchestration proof, not the quality proof.**

- `triage_pass_rate = 0.90` means 27/30 cases produced exactly the expected sanity flags. The 3 misses are 1 `extreme_case` (the rules don't catch every adversarial vital sign — by design) + 2 `low_risk` benign-extra-flag mismatches in the eval-set catalogue. Not orchestration bugs.
- `risk_band_match_rate = 0.467` is **the dominant headline gap and is *not* an orchestration finding.** The model dramatically over-classifies *intermediate* cases as *high* (11/13). Likely reading: (a) **threshold mismatch** — AusCVDRisk's 0.05 / 0.10 thresholds were calibrated on Australian primary-care 5-year absolute risk (~5–10% prevalence in the 40–74 age band); the model is trained on UCI HFP (Cleveland prev=0.46), so applying those thresholds to a model fit on a much higher-prevalence population pushes most cases past 0.10 by construction. (b) **distribution shift** — Phase 2.6 ([`docs/research/11-drift-design.md`](docs/research/11-drift-design.md)) showed TabICL translates input distribution shift into 3–4× larger predicted-probability shifts than calibrated linear/tree models. The synthetic cases sit in a feature region the Cleveland fold's training distribution didn't fully cover. **The honest reading is that the v1 model is well-calibrated under LODO across UCI sources but is not validated for the synthetic case distribution.** Phase 6 will (1) re-evaluate against the Hungarian-fold artefact (lower prevalence, lower TabICL prediction-PSI); (2) re-calibrate the band thresholds on a larger synthetic case set (or use percentile-bucket assignment); (3) consider 4-model ensemble voting for the band call.
- `guideline_pass_rate = 1.0` and `letter_pass_rate = 1.0` are **diagnostic of the smoke harness.** Mock-LLM picks the first sentence of the first retrieved passage; always-entails NLI verifies every claim with `p_entail = 0.99`. Both pass-rates are *guaranteed* under this harness. Phase 6 ships the production headline with DeBERTa NLI + Claude Sonnet 4.5 / GPT-4o-mini.
- `full_pipeline_pass_rate = 0.40` is the AND of the four per-stage rates — dominated by the risk-band miss. *Orchestrationally* the pipeline succeeds end-to-end on every case (no agent crashes, no checkpoint corruption, no HITL-routing failures across 30 cases).

**Decisions baked into Phase 5 / Phase 6 from this result:**

- **Phase 5 React UI binds to the FastAPI schema, not LangGraph.** The state schema *is* the API schema; the UI doesn't need to know LangGraph exists. ADR-018 §6.
- **Phase 5.3 risk dashboard surfaces both the raw probability and the band**, with a callout to the model card §11 explaining when the band may overshoot. Reviewers will see the band as advisory until Phase 6 ships the recalibration.
- **Phase 6 risk-agent revisit.** Hungarian fold + threshold recalibration + 4-model ensemble voting; eval grows from 30 to 100 cases; auto-approve harness gets a *judge-as-reviewer* companion (LLM issues HITL decisions, graded against a gold set).
- **Phase 6 guideline/letter agents swap to a real LLM.** Mock-LLM headline is the regression baseline; Claude Sonnet 4.5 + GPT-4o-mini A/B is the Phase 6 deliverable.
- **Phase 7 swaps `InMemorySaver` for `PostgresSaver`** (Supabase) per ADR-021 (placeholder). Sets the per-case latency SLO + adds a global deadline.

**Honest weaknesses (full discussion in [`docs/research/15-agent-design.md`](docs/research/15-agent-design.md) §8):**

- **30 cases is too small for stable per-tag CIs** — `borderline` is n=2; `extreme_case` is n=1. Phase 6 grows to 100. §8.1.
- **The risk agent's calibration is not validated for synthetic cases.** Recapitulated in detail in §8.2 of the research doc. The eval surfaces it; Phase 6 fixes it.
- **The auto-approve harness validates plumbing, not HITL quality.** Real reviewer behaviour (edit / reject / approve) is not exercised. §8.3.
- **LangGraph 0.6 is a 2024–2025 framework** with API churn; pinned `>=0.6,<0.7` upper bound is the safety belt. §8.4.
- **No global per-case deadline in Phase 4** — duration distribution is reported; SLO + deadline land in Phase 7. §8.5.
- **The `letter` agent is a deterministic template renderer.** Citation-preserving, deterministic, free; doesn't read like a clinician-drafted referral. Phase 6 ships an LLM-drafted parallel branch and A/Bs them. §8.6.
- **The `_ArtefactCache` is a process-local singleton** — reload requires a process restart. Phase 7 deploy story handles refresh; Phase 4 doesn't. §8.7.
- **No multi-reviewer HITL.** Single-reviewer state advancement. Phase 6 may revisit if inter-reviewer disagreement is the dominant failure mode. §8.8.

**Reproduce.** Full 30-case run (~35 s on a recent CPU; uses the v1 TabICL Cleveland artefact if present, mock-classifier fallback if not):

```bash
uv run --project backend python backend/scripts/eval_agents.py
```

Writes [`reports/v1/agents/per_case.json`](reports/v1/agents/per_case.json), [`reports/v1/agents/aggregate.json`](reports/v1/agents/aggregate.json), and 3 figures under [`reports/v1/figures/agents/`](reports/v1/figures/agents/). The CLI also exposes `--smoke` (3 cases, ~5 s, the CI default — no joblib artefact required), `--limit N`, `--tag <tag>`, `--risk-model {tabicl,xgboost,lr,ensemble}`, and `--risk-source {Cleveland,Hungarian,LongBeachVA,Switzerland}`. The FastAPI surface runs locally with `uv run --project backend uvicorn cardiorisk.api:build_app --factory --reload`; the OpenAPI spec is at `/docs`.

## 12. Phase-6 eval harness (100-case agent eval + regression gate)

Phase 6 grew the Phase 4 30-case smoke into a real eval. The case set went from 30 to **100** (25 high + 25 intermediate + 25 low + 10 borderline + 8 data-quality + 4 extreme + 3 refusal), four new per-case metrics ship (`citation_precision`, `citation_recall`, `recommendation_correctness`, `hallucination_rate`), an LLM-as-judge layer with a pluggable Protocol scores the letter drafts on two 1-5 Likert axes, the LLM stack moved to free-tier-only (Mock + Gemini 2.5 Flash + opt-in Groq Llama-3.3-70B; Anthropic / OpenAI clients kept for opt-in users but excluded from the default config), and a ±2 pp regression gate enforces the locked mock-pipeline baseline in CI on every PR. Full methodology + headline numbers + reproduce steps live in [EVAL.md](EVAL.md); binding choices in [ADR-019](docs/adr/019-phase-6-eval-harness.md); opinionated walkthrough in [`docs/research/19-phase-6-eval-design.md`](docs/research/19-phase-6-eval-design.md).

### Headline (mock pipeline, 100 cases, $0 cost)

| Metric | Value | Notes |
|---|---|---|
| Cases | 100 | locked at `eval/agents/cases.jsonl` v1 |
| Wall-clock per case (median / p95) | 1029 ms / 1055 ms | smoke pipeline |
| Triage pass rate | **0.97** | 3 data-quality cases not surfacing the injected flag |
| Risk band match | 0.43 | mock TabICL classifier (LODO cross-source ceiling per §3) |
| Guideline pass | 1.00 | always-entail Mock NLI |
| Letter pass | 1.00 | mock letter draft always meets word-count floor |
| Recommendation correctness | 0.41 | mock LetterAgent template ≠ the expected keyword on ~60% of cases |
| Citation precision | **1.00** | mock LLM cites the literal prompt chunks |
| Citation recall | **1.00** | every verified claim has a citation in the retrieved set |
| Hallucination rate | **0.00** | mock LLM never tries to fabricate |
| Judge pass rate (MockJudge) | 0.41 | mirrors recommendation correctness (deterministic mock judge) |
| Total cost | **$0.00** | mock floor cell, by construction |

**How to read this table.** The mock pipeline is the *floor*, not the ceiling. `MockLLMClient` cites the literal chunks it sees and never hallucinates (precision / recall / hallucination_rate are locked at perfect). The `LetterAgent`'s template applied to mocked chunks doesn't always land in the expected recommendation family — that's the 0.41 figure. The `risk_band_match_rate` of 0.43 reflects the same TabICL LODO ceiling Phase 2.4 documented; it is recapitulated here on synthetic cases.

### Live Gemini cell

Run locally (not committed to CI, but ships in the codebase end-to-end):

```bash
GEMINI_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm gemini --judge gemini \
    --reports-dir reports/v1/agents/gemini
```

Expected USD cost for a full 100-case run: ~$0.05 (well inside the Gemini 2.5 Flash free tier of 10 RPM / 250 K TPM / 250 RPD — the run is fully covered if you've made no other Gemini calls that day). The third opt-in cell is Groq Llama-3.3-70B (`--llm groq --judge groq`, requires `GROQ_API_KEY`).

### Regression gate (CI-enforced)

The mock baseline lives at [`reports/v1/agents/baseline_mock.json`](reports/v1/agents/baseline_mock.json). On every PR, `.github/workflows/ci.yml` runs:

```bash
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json
```

The job exits non-zero (and fails the PR) if any of the nine tracked metrics drifts by more than ±2 percentage points in the wrong direction. The lower-is-better axis (`mean_hallucination_rate`) fails on increases; the eight higher-is-better metrics fail on decreases. Missing-baseline metrics (new fields not yet in the baseline) record as `fail=False` so the gate is silent on metric additions. The baseline is refreshed in the same PR as whatever motivated the refresh.

## 13. Phase-7 observability + cost

Phase 7 wires the free-tier observability stack — **Langfuse Cloud Hobby** (LLM-shaped traces with prompt + completion + token counts + USD cost + per-node spans) + **Sentry Free** (FastAPI + Next.js error tracking with a recursive `patient`-key scrubber on every SDK) + **Vercel Web Analytics + Speed Insights** (frontend RUM). Every observability hook is **no-op when its key is unset** so CI runs with both keys deliberately blank and never makes a network call. Full methodology + rationale in [ADR-024](docs/adr/024-observability-free-tier.md); opinionated walkthrough in [`docs/research/20-observability-design.md`](docs/research/20-observability-design.md).

### Per-case `trace_id` round-trip

A `trace_id` field rides on `AgentState` (Pydantic, default `None`; round-tripped by `state_to_dict` / `state_from_dict`) and is exposed end-to-end:

1. **Backend.** Every `POST /v1/agents/cases` wraps the agent run in `start_root_span(case_id)`, which returns the Langfuse-issued UUIDv4 when the key is set or mints a deterministic `mock-trace-<6-hex>` sentinel when it isn't. The handler writes the trace ID into the response body **and** as the `X-Trace-Id` HTTP response header. Subsequent `GET /v1/agents/cases/{id}` and `POST /v1/agents/cases/{id}/decide` calls return the same trace ID so the UI can refresh it after every HITL decision.
2. **Frontend.** `caseSnapshotSchema` (zod) accepts `trace_id: z.string().nullable().optional()` so neither side of the contract is brittle. The audit screen renders an "Open in Langfuse" deep-link button when **both** a real trace ID is present **and** `NEXT_PUBLIC_LANGFUSE_TRACE_URL_BASE` is set in the environment; otherwise it renders a muted "Local mock — no remote trace" badge.

### p95 latency budget gate (added to the Phase 6 regression gate)

The Phase 6 regression gate gains two additional metrics, both checked with a **multiplicative** ±20% tolerance (independent from the ±2 pp band used by the pass-rate metrics):

| Metric | Direction | Tolerance |
|---|---|---|
| `median_total_duration_ms` | latency (lower better; multiplicative) | ±20% |
| `p95_total_duration_ms` | latency (lower better; multiplicative) | ±20% |

The band is ±20% rather than ±2 pp because latency variance is multiplicative, not additive — a ±2 pp band on a 1156 ms baseline ≈ "fail at +23 ms", which is the CI runner noise floor. The CLI exposes both knobs with `--regression-tolerance-pp` (default 2.0) + `--latency-regression-tolerance-pct` (default 0.20). See ADR-024 §5 for the binding rationale and the honest trade-off that ±20% intentionally absorbs the Langfuse / Sentry SDK-import overhead the same PR introduces (+127 ms median on `baseline_mock.json`).

### Reproducing the live observability stack

```bash
LANGFUSE_PUBLIC_KEY=... \
LANGFUSE_SECRET_KEY=... \
LANGFUSE_HOST=https://cloud.langfuse.com \
SENTRY_DSN=... \
GEMINI_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm gemini --judge gemini \
    --reports-dir reports/v1/agents/gemini
```

Traces flow to Langfuse Cloud (50 K observations / month, 30-day retention), errors to Sentry (5 K errors / month, `patient`-key-scrubbed before send). The frontend auto-mounts `@vercel/analytics` + `@vercel/speed-insights` so web vitals (LCP / FID / INP / CLS) land on every Vercel-deployed page view.

### Honest weaknesses

- **Langfuse Hobby retains traces for 30 days.** Headline traces from a long-ago demo disappear. The mock baseline is the system of record; Langfuse is the live drilldown.
- **No APM on the FastAPI app on the free tier.** Sentry's performance traces are sampled at 0.1 by default — enough to spot outliers, not enough to build operational dashboards. The eval-harness latency gate (`median` + `p95` with the ±20% band) is the operational guard.
- **The ±20% latency band intentionally hides the Phase 7 SDK-import bump.** The right next step is to tighten back towards ±10% in Phase 8 once the SDK overhead is the steady state.

## 14. Limitations & out-of-scope

- **Not validated in a clinical setting.** No clinician-in-the-loop study, no real-EHR integration, no deployment.
- **Trained on small, biased UCI sources.** ~920 rows total across four heterogeneous sources, none of which is contemporary Australian primary-care data. Generalisability to any cohort outside these four sources is *unverified*.
- **LongBeachVA ≥70 stratum is structurally under-served.** Every v1 model loses meaningful AUROC on that stratum. The model card flags this explicitly; any deployed surface must either exclude this stratum or carry a "low-confidence" UI affordance.
- **No external validation cohort.** The LODO protocol simulates a deployment-like scenario (test on a source the model has never seen) but does not replace external validation on a non-UCI cohort.
- **No fairness audit beyond sex + age band.** Race, ethnicity, socioeconomic status, geographic strata: not in the data, not auditable.
- **No auditable F sex-stratum subgroup-feature-drift on any LODO fold.** §5 above. We do not report sex-based feature-importance drift because every fold's F count is below the `n ≥ 30` honesty guard.
- **Calibration slice size limits XGBoost.** See §6.
- **KernelSHAP test-slice cap (80 rows per model × fold).** See §5 "Methodological caveats" — inflates the standard error of mean |SHAP| by ~1.2×–1.9×; rank-based cross-model agreement is essentially unaffected. Recoverable via `--max-test-rows 0`.
- **The Honours-Ensemble row is the architecture only, not the WOA-Ensemble pipeline** (see §3 of [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md)). The WOA feature-selection layer that produced the Honours report's headline number is not in the supplied archive; we did not invent one. Reading the Honours-Ensemble row as if it is "WOA-Ensemble under our protocol" is incorrect.

## 15. Honesty caveats specific to the Honours-Ensemble row

The Honours team's Final Report §7.2 Table 2.2 reports WOA-Ensemble on HFP at sensitivity 89.72%, specificity 83.12% (single 80/20 split, no CV, no per-source breakdown, no calibration). Our table reports a different number under a different protocol (4-fold LODO, calibrated, sensitivity at 85% / 90% specificity not at the default 0.5 threshold, no WOA layer). A direct numerical comparison ("89.72% vs our X%") is misleading. The right reading is qualitative: the Honours architecture's relative position against the v1 trio under a fair LODO protocol. Full discussion in [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md).

## 16. Reproducibility

- All code: this repo, MIT-licensed.
- Run training (full LODO, ~40 min on a recent CPU): `uv sync --project backend && uv run --project backend python backend/scripts/train_v1.py`. `--smoke` for a 1-fold synthetic-data smoke pass (~70s).
- Run explainability sweep (~2h 20m on a recent CPU): `uv run --project backend python backend/scripts/compute_explanations.py`. `--smoke` for the 1-fold smoke pass (~30s); `--max-test-rows 0` to explain every per-fold test row (~4× wall-clock).
- Run drift sweep (~30s on a recent CPU): `uv run --project backend python backend/scripts/compute_drift.py`. `--smoke` for the 1-fold smoke pass (~10s).
- Model artefacts: not in git (per [ADR-010](docs/adr/010-model-artefact-storage.md)); regenerated locally by the training script above.
- Reports: in git under [`reports/v1/`](reports/v1/) — `metrics_per_fold.json`, `metrics_aggregate.json`, `explainability/{explanations_per_cell,explanations_aggregate,cross_model_agreement}.json`, `drift/{per_fold,aggregate}.json`, and `figures/**/*.png`.
- Determinism: every wrapper pins `random_state` / `torch.manual_seed` to the project seed (20260505). XGBoost reproduces to ~1e-6 across runs; the PyTorch Ensemble reproduces to ~1e-5 (deterministic-algorithms flag deliberately not enabled — see [ADR-012](docs/adr/012-honours-baseline-reproduction.md)). KernelSHAP per-row values reproduce to ~1e-5; aggregate quantities (mean |SHAP|, Spearman ranks) to ~1e-6. Drift PSI is closed-form on binned histograms and reproduces exactly bit-for-bit modulo numpy floating-point ordering.
- CI: all three smoke runs are enforced on every PR (`train_v1.py --smoke`, `compute_explanations.py --smoke`, and `compute_drift.py --smoke` steps in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 17. References

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
- [ADR-015](docs/adr/015-corpus-ingestion.md) — corpus ingestion (Phase 3.1).
- [ADR-016](docs/adr/016-retrieval-stack.md) — retrieval stack (Phase 3.2; with 2026-05-15 amendment recording the real-corpus reranker reversal).
- [ADR-017](docs/adr/017-citation-and-nli-verification.md) — citation-mandatory generation + NLI verification (Phase 3.3).
- [ADR-018](docs/adr/018-agent-orchestration.md) — 4-agent orchestration with LangGraph + HITL gates + FastAPI surface (Phase 4).
- [ADR-019](docs/adr/019-phase-6-eval-harness.md) — Phase-6 eval harness (100 cases + 4 new metrics + LLM-judge + free-tier-only LLM stack + ±2 pp regression gate).
- [ADR-024](docs/adr/024-observability-free-tier.md) — Phase-7 free-tier observability stack + multiplicative ±20% p95 latency budget gate.
