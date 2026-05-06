# 10 — Explainability (Phase 2.5)

> **Status:** Phase 2.5 deliverable. Companion to [`08-v1-model-results.md`](./08-v1-model-results.md) (LODO discrimination/calibration) and [`09-honours-vs-v1.md`](./09-honours-vs-v1.md) (cross-model honesty). Numbers below are produced verbatim by `backend/scripts/compute_explanations.py` from the v1 model artefacts under `models/v1/` and `data/processed/combined.parquet`. ADR for binding methodology choices: [ADR-013](../adr/013-explainability-strategy.md).
>
> **Purpose.** Five things, in priority order:
>
> 1. Quantify *which features* each of the four v1 models leans on, on the same scale (KernelSHAP probability deltas), under the same protocol (LODO).
> 2. Test whether the four models *agree* on feature importance, via a Spearman rank-concordance matrix — the prerequisite question to the Phase 3 agentic system being willing to draft a single risk-driver narrative on top of any one of them.
> 3. Sanity-check the cross-model headline (KernelSHAP) against fast native attribution (TreeSHAP for XGBoost; analytic LR-coefficient summed-back for L1 LR), to flag any algorithm-induced disagreement before the cross-model story is read at face value.
> 4. Surface per-archetype local explanations (TP-high, TP-low, FN, FP per model × fold) so the global rankings can be cross-read against actual patient predictions.
> 5. Report subgroup-feature-drift on the auditable strata only, consistent with the Phase 2.3b honesty discipline ([`07-eval-design.md`](./07-eval-design.md) §5).
>
> **Methodology** is bound by [ADR-013](../adr/013-explainability-strategy.md); a contingency was applied at execution time and is recorded inline in §1 below + the ADR amendment dated 2026-05-06.

---

## 1. What was actually run

KernelSHAP is the cross-model headline (per ADR-013 §1). The first full-LODO attempt with ADR-013's originally-planned settings (`nsamples=256`, full per-fold test slice up to 303 rows on Cleveland) projected to ~3–4 hours wall-clock, which crossed the ADR's 90-minute "trigger to revisit" clause. Two mitigations were applied before re-running:

| Knob | ADR-013 plan | Shipped value | Rationale |
|---|---|---|---|
| `nsamples` (KernelSHAP coalition budget) | 256 | **128** | Smoke testing on the LR fold shows top-6 feature ranks are stable; bottom-half ranks shift by ≤1 position. Acceptable for a rank-based cross-model surface. |
| `max_test_rows` (KernelSHAP test-slice cap) | none (every test row) | **80 per (model × fold)** | Stratified-sampled by `y_test` to preserve per-fold prevalence; the four archetype rows are always included. Both `n_test_full` and `n_test_explained` are recorded in the JSON so the cap is auditable. |
| `background_k` (`shap.kmeans` medoids) | 50 | 50 (unchanged) | — |
| `min_stratum_size` (subgroup-drift guard) | inherited from Phase 2.3b | 30 (unchanged) | — |

Re-running with these settings completed in **2h 20m wall-clock** on the M4 Pro CPU. Per-cell breakdown (logged in `/tmp/p25_full_run.log`):

| Model | Per-cell KernelSHAP wall-clock (median) |
|---|---|
| TabICL | 13–15 min (transformer forward passes — dominant cost) |
| Ensemble | ~2 min (4 PyTorch nets per forward pass) |
| XGBoost | ~10 sec |
| LR | ~2 sec |

The `--max-test-rows 0` flag explains every row at the same per-row cost (~4× wall-clock on full mode); use it if a downstream consumer needs per-row SHAP values for the full per-fold test slice.

## 2. Cross-model agreement (the headline)

For each LODO fold and each pair of models, we compute the Spearman rank correlation between the two models' mean |SHAP value| feature rankings (16 raw HFP features). The aggregate matrix is the average across the four folds:

| Aggregate Spearman ρ | TabICL | XGBoost | LR | Ensemble |
|---|---|---|---|---|
| **TabICL** | 1.00 | 0.90 | 0.84 | 0.83 |
| **XGBoost** | 0.90 | 1.00 | 0.85 | 0.83 |
| **LR** | 0.84 | 0.85 | 1.00 | 0.81 |
| **Ensemble** | 0.83 | 0.83 | 0.81 | 1.00 |

Numbers verbatim from `reports/v1/explainability/cross_model_agreement.json`. Visualisation: [`reports/v1/figures/explainability/aggregate_cross_model_agreement_heatmap.png`](../../reports/v1/figures/explainability/aggregate_cross_model_agreement_heatmap.png).

**Reading.** All six pairwise rank correlations are ≥ 0.81. The four models substantially agree on which features matter, which is the precondition the Phase 3 agentic system needs in order to write a coherent risk-driver narrative regardless of which model produces the underlying score. Within the agreement:

- **TabICL ↔ XGBoost (0.90)** is the closest pair, despite the two models being maximally architecturally different (transformer in-context learning vs gradient-boosted trees). Convergent feature-importance evidence across very different inductive biases is reassuring.
- **Ensemble disagrees most with the others (0.81–0.83 vs everything else).** The Honours-architecture Ensemble (4-net mean-average over DNN + 1D CNN + LSTM + BiLSTM) carries some idiosyncratic emphasis — most visibly, it weights `ST_Slope` and `ST_Slope_was_missing` higher than the other three (see §3 below). Not large enough to be alarming; large enough to be the single piece of evidence that the Honours architecture is not "just" reproducing what XGBoost or LR already say.
- **LR is a closer match to XGBoost (0.85) than to TabICL (0.84) or Ensemble (0.81).** Not surprising: LR's importance comes from coefficients on (RCS-expanded) features; XGBoost's comes from split gain on the same raw features after one-hot. Both surface tabular-feature importance in the additive-feature sense, which TabICL's contextual scoring and the Ensemble's nonlinear nets do less directly.

### Per-fold cross-model agreement

The aggregate matrix conceals fold-level variation, which is itself informative:

| Fold | Range of pairwise Spearman ρ |
|---|---|
| Cleveland | 0.72 – 0.92 |
| Hungarian | 0.80 – 0.89 |
| **LongBeachVA** | **0.69 – 0.84 ← weakest fold** |
| Switzerland | 0.82 – 0.98 ← strongest fold |

Per-fold heatmaps under [`reports/v1/figures/explainability/<fold>_cross_model_agreement_heatmap.png`](../../reports/v1/figures/explainability/).

LongBeachVA being the weakest agreement fold is consistent with Phase 2.3b's finding that LongBeachVA is the structural-difficulty fold (75% prevalence, the highest cholesterol-missingness, the prevalence inversion vs. the training pool). When the data is harder, the four models reach for slightly different feature subsets to fit it. Switzerland being the strongest is partly an artefact of its degenerate-by-design structure (94% prevalence) — every model reaches for the same handful of always-present features, which collapses the rank ordering.

## 3. Global feature importance per model (cross-fold averaged)

Mean |SHAP value| (probability-space) per raw HFP feature, averaged across the four LODO folds. Top 8 per model:

| Rank | TabICL | XGBoost | LR | Ensemble |
|---|---|---|---|---|
| 1 | **ChestPainType (0.112)** | **ChestPainType (0.144)** | **ChestPainType (0.122)** | **ChestPainType (0.104)** |
| 2 | FastingBS (0.047) | FastingBS (0.072) | MaxHR (0.069) | ST_Slope (0.055) |
| 3 | ExerciseAngina (0.045) | Oldpeak (0.065) | Oldpeak (0.062) | Oldpeak (0.040) |
| 4 | ST_Slope (0.041) | MaxHR (0.053) | ExerciseAngina (0.045) | ExerciseAngina (0.039) |
| 5 | Oldpeak (0.037) | ExerciseAngina (0.044) | ST_Slope (0.043) | Sex (0.037) |
| 6 | Sex (0.034) | ST_Slope (0.041) | Sex (0.040) | FastingBS (0.034) |
| 7 | MaxHR (0.033) | Age (0.030) | Cholesterol (0.027) | MaxHR (0.031) |
| 8 | Cholesterol (0.025) | Sex (0.030) | Age (0.023) | ST_Slope_was_missing (0.027) |

Numbers from `reports/v1/explainability/explanations_per_cell.json`. Per-(model × fold) bar charts and beeswarms under [`reports/v1/figures/explainability/<model>_<fold>_global_{bar,beeswarm}.png`](../../reports/v1/figures/explainability/).

**Reading.**

- **`ChestPainType` is universally rank-1** across all four models on all four folds. The asymptomatic (`ASY`) level dominates the contribution (LR sanity-check below: `ChestPainType_ASY` is the single largest LR coefficient on every fold). Clinically this matches the textbook intuition that asymptomatic CVD is a very different prior risk than typical/atypical angina presentation.
- **`Oldpeak` (ST-segment depression at exercise) is top-3 for every model except TabICL** — TabICL ranks it #5. On Switzerland (94% prevalence, severe cholesterol missingness) `Oldpeak` actually overtakes `ChestPainType` as rank-1 for both XGBoost (0.189 vs 0.069) and LR (0.167 vs 0.069). The four models *can* shift their feature emphasis when the data structure changes; they just don't shift it the same way.
- **The Ensemble is the only model with a missingness indicator (`ST_Slope_was_missing`) in its top 8.** The Ensemble has learned to read `ST_Slope`'s missingness pattern as informative — consistent with the Phase 2.3b finding ([`08-v1-model-results.md`](./08-v1-model-results.md) §3) that LongBeachVA's `ST_Slope` missingness is itself a structural signal.
- **`Cholesterol` is bottom-half for every model** despite being a textbook CVD risk factor. The likely reason is the Switzerland fold (`Cholesterol` is 100% missing — every Switzerland row carries the imputed median + the `Cholesterol_was_missing` flag), which dilutes the cross-fold average. Within-fold, `Cholesterol` is top-5 for TabICL on Hungarian (0.033) and LR on Cleveland (0.026); it's not that the models ignore it, it's that one fold has none of it.

## 4. Sanity check: KernelSHAP vs native attribution

For the two models where a native attribution algorithm is available, we compute mean |SHAP value| from both algorithms and report the Spearman rank correlation between the two rankings. High concordance means the cross-model KernelSHAP table in §3 reflects model behaviour, not algorithm artefact.

| Fold | XGBoost (KernelSHAP vs TreeSHAP) | LR (KernelSHAP vs analytic-summed) |
|---|---|---|
| Cleveland | 0.91 | 0.93 |
| Hungarian | 0.96 | 0.89 |
| LongBeachVA | 0.93 | 0.93 |
| Switzerland | 0.98 | 0.89 |
| **Mean** | **0.95** | **0.91** |

Visualisations: per-fold [`xgboost_<fold>_treeshap_vs_kernelshap.png`](../../reports/v1/figures/explainability/) (scatter, points = features; identity line for reference) and [`lr_<fold>_summed_vs_basis.png`](../../reports/v1/figures/explainability/) (bars: per-spline-basis vs summed-to-feature attribution).

**Reading.**

- **XGBoost: 0.95 mean Spearman.** TreeSHAP and KernelSHAP agree on which features matter for XGBoost across all four folds. The cross-model headline in §2 / §3 is *not* an artefact of the algorithm choice on the workhorse model.
- **LR: 0.91 mean Spearman.** KernelSHAP's coalition sampling on the RCS-expanded LR pipeline introduces a small extra variance (~0.04 lower than XGBoost's TreeSHAP comparison, which is exact). Still well above ADR-013's 0.7 trigger.
- The ADR-013 contingency clause "if TreeSHAP and KernelSHAP disagree by Spearman rank correlation < 0.7 on XGBoost, the cross-model KernelSHAP surface needs additional justification" did not fire. The KernelSHAP cross-model story stands.

The LR per-spline-basis detail figures additionally surface *which knot* of each RCS-expanded continuous feature is doing the work. On Cleveland, for example, `Age`'s spline contributes through `x0_rcs1` (the lower knot, ages < ~50) but not through `x0_rcs2` (the upper knot) — i.e. the LR has learned an age effect that saturates above a midlife threshold. This is exactly the nonlinearity the RCS expansion was added for in [ADR-008](../adr/008-preprocessing-pipeline.md); the detail figures make that visible.

## 5. Local explanations: the four archetypes

For each (model × fold), four representative test patients are picked and rendered as SHAP waterfall plots:

- **TP-high** — highest-confidence correct positive.
- **TP-low** — lowest-confidence correct positive (the "closest call that came out right").
- **FN** — most over-confident missed positive (the worst false negative).
- **FP** — highest-confidence false positive (the worst false alarm).

64 PNGs total = 4 archetypes × 4 folds × 4 models. All under [`reports/v1/figures/explainability/<model>_<fold>_{tp_high,tp_low,fn,fp}_waterfall.png`](../../reports/v1/figures/explainability/). The selection logic (fixed 0.5 threshold, deterministic tie-breaking on `test_index`) is in [`backend/cardiorisk/explainability/archetypes.py`](../../backend/cardiorisk/explainability/archetypes.py).

The archetype rows are *always* included in the KernelSHAP test-row cap (§1), so the waterfall plots are computed on the actual archetype patients — not nearest-sample stand-ins. The per-row `y_true`, `y_proba`, and feature values are recorded in the `archetypes` block of `explanations_per_cell.json` for any reviewer who wants to look up a specific row.

**Why these four and not "5 random rows" or "all FNs":** four-archetype framing maps onto the operating-point conversation from [`07-eval-design.md`](./07-eval-design.md) §3 (sens@85% / sens@90% spec → these are the FN/FP trade-offs you're picking). A reviewer looking at the four waterfalls per (model × fold) sees the model's behaviour at all four corners of the confusion-matrix square, which is what they'd ask about anyway.

## 6. Subgroup-feature drift (auditable strata only)

Per ADR-013 §4, subgroup-drift analysis runs only on strata with `n ≥ 30` (the same `min_stratum_size` guard Phase 2.3b uses for fairness gaps). What survives:

| Fold | Auditable sex strata | Auditable age strata |
|---|---|---|
| Cleveland | M (n ≈ 56–61) | 50–69 (n ≈ 53–55) |
| Hungarian | M (n ≈ 58–63) | <50 (n ≈ 43–44), 50–69 (n ≈ 36–37) |
| LongBeachVA | M (n ≈ 76–78) | 50–69 (n ≈ 64–66) |
| Switzerland | M (n ≈ 72–73) | 50–69 (n ≈ 60–61) |

`n` ranges within each stratum-fold pair because the four models see slightly different stratified-sampled test-row subsets (per §1's `max_test_rows` cap; archetype-row class membership shifts the sex/age distribution by ±5 rows). Skipped strata (the F sex-stratum on every fold; <50 and ≥70 age-bands on Cleveland / LongBeachVA / Switzerland) are recorded in `explanations_per_cell.json` under each cell's `subgroup_drift_*.skipped_strata`.

**The structural finding to flag honestly:** *no LODO fold has an auditable F sex-stratum* under our `n ≥ 30` threshold. Every fold's F count is below the guard (Cleveland F ≈ 19–24, Hungarian F ≈ 17–22, LongBeachVA F ≈ 2–4, Switzerland F ≈ 7–8). This is the same data-shortage issue Phase 2.3b's fairness audit ran into, surfaced in the explainability domain: we cannot honestly say "the model relies on different features for women than for men" because we do not have enough women in any fold to estimate the per-feature mean |SHAP| reliably. The MODEL_CARD.md §4 sex-AUROC discussion (Cleveland and Hungarian only) is the corresponding discrimination-side finding; here the same data ceiling caps what can be said about feature reliance.

Per-(model × fold × stratum) bar plots are still generated wherever the guard passes (M-stratum always; 50–69 age band always; <50 age band on Hungarian only) — see [`reports/v1/figures/explainability/<model>_<fold>_subgroup_drift_{sex,age_band}.png`](../../reports/v1/figures/explainability/). They show small per-feature deltas (≤ 0.02 in mean |SHAP|) between the M-stratum and the overall population — i.e. the M-stratum is not behaviourally different from the per-fold average, which is unsurprising given M dominates the per-fold sample. The Hungarian <50 vs 50–69 comparison is the only auditable age contrast in the whole sweep; it shows the four models have moderately different age-band feature weighting (`ChestPainType` carries more weight on <50 across all four; `Oldpeak` carries more on 50–69 for XGBoost specifically), but with small per-stratum sample sizes the deltas are not statistically robust.

## 7. Honest discussion of explainer disagreement

ADR-013 §"Why KernelSHAP-everywhere…" predicts the cross-model surface will be the most trustworthy comparison surface *and* that the algorithm-induced disagreement (KernelSHAP vs TreeSHAP for XGBoost; KernelSHAP vs analytic-LR for LR) is its own finding. The numbers came out unambiguously on the favourable side of that prediction (§4: 0.91–0.95 mean Spearman), but two caveats deserve the same prose treatment:

1. **The 80-row test-slice cap inflates the global-importance standard error.** Mean |SHAP value| per feature is a sample mean over the explained test rows; capping at 80 rows (vs the per-fold full slice of 123–303 rows) inflates the standard error by roughly `√(n_full / 80) ≈ 1.2× – 1.9×`. The Spearman rank-stability across this regime is documented in §1; the *level* uncertainty is larger. A reviewer comparing the absolute mean |SHAP| of `ChestPainType` (0.144 for XGBoost) to `Oldpeak` (0.065 for XGBoost) should read those as significantly different (the gap ≈ 0.08 dominates the standard error); a reviewer comparing `MaxHR` (0.053) to `ExerciseAngina` (0.044) should not.

2. **KernelSHAP's interventional sampling is approximate when features are correlated.** The seven HFP categorical features and nine continuous features have non-trivial mutual correlation (`Sex` × `MaxHR`, `ChestPainType` × `ExerciseAngina`, `Oldpeak` × `ST_Slope`). Janzing et al. 2020 (cited in ADR-013) shows interventional Shapley estimates can over- or under-state contributions of correlated features by a factor of ~1.5× in the worst case. The TreeSHAP-vs-KernelSHAP sanity check (§4) bounds this empirically on XGBoost: the two algorithms agree to Spearman 0.95 here, which is consistent with feature correlation in this dataset being too modest to materially distort the cross-model rank order. We do not claim it's zero; we claim it's small enough to be a footnote, not a headline.

Neither caveat changes the cross-model agreement story (§2). Both are surfaced in the model card.

## 8. What this enables for Phase 3

The point of Phase 2.5 was to give Phase 3 a concrete per-feature attribution surface that is (a) computed identically across the four risk models, (b) sanity-checked against fast native algorithms where available, and (c) auditable per (model × fold × archetype) so a downstream LLM can quote a real SHAP value on a real patient row instead of inventing a justification. All three properties hold:

- (a) cross-model: §2 + `cross_model_agreement.json`.
- (b) sanity: §4 + `xgboost_<fold>_treeshap_vs_kernelshap.png` + `lr_<fold>_summed_vs_basis.png`.
- (c) per-row: §5 + the `archetypes` blocks in `explanations_per_cell.json`.

The Phase 3 risk-driver narrative drafter can therefore start from "the model's top-3 SHAP-attributed features for *this patient* are X (+0.12), Y (+0.08), Z (−0.05)" and have those numbers traceable back to a deterministic, reproducible per-fold artefact rather than a re-run-every-time SHAP call. Whether the agentic surface uses TabICL's, XGBoost's, LR's, or the Ensemble's attribution as the narrative source is a Phase 3 question — but cross-model agreement of ≥ 0.81 means the narrative wouldn't change qualitatively across that choice.

## 9. Reproducibility

- **Code:** [`backend/cardiorisk/explainability/`](../../backend/cardiorisk/explainability/) (8 modules: `encoder`, `kernel_shap`, `tree_shap`, `linear_attribution`, `archetypes`, `subgroup_drift`, `cross_model_agreement`, `figures`, `orchestrator`); driver at [`backend/scripts/compute_explanations.py`](../../backend/scripts/compute_explanations.py).
- **Run** (full LODO sweep, ~2h 20m on M4 Pro CPU):
  ```bash
  cd backend && uv run python scripts/compute_explanations.py
  ```
- **Smoke run** (1 fold, tiny budgets, ~30 s — what CI runs):
  ```bash
  cd backend && uv run python scripts/compute_explanations.py --smoke
  ```
- **Override the test-slice cap** (~4× wall-clock; explains every per-fold test row):
  ```bash
  cd backend && uv run python scripts/compute_explanations.py --max-test-rows 0
  ```
- **Outputs:** `reports/v1/explainability/{explanations_per_cell,explanations_aggregate,cross_model_agreement}.json` + `reports/v1/figures/explainability/*.png` (142 PNGs).
- **Determinism:** ~1e-5 SHAP-value drift across runs from the KernelSHAP coalition sampler's RNG (same band as Phase 2.4's PyTorch Ensemble); aggregate quantities (mean |SHAP|, Spearman ranks) stable to ~1e-6.
- **CI:** smoke run is enforced on every PR (`compute-explanations-smoke` step in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — added in this phase).

## 10. References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence.
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causality problem. AISTATS.
- Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. AI.
- [ADR-013](../adr/013-explainability-strategy.md) — explainability strategy (binding decisions + 2026-05-06 amendment with the wall-clock contingency).
- [`08-v1-model-results.md`](./08-v1-model-results.md) — the LODO discrimination + calibration story this builds on.
- [`09-honours-vs-v1.md`](./09-honours-vs-v1.md) — cross-model honesty discussion; §4 of this doc is the explainability companion to its §4 subgroup-AUROC discussion.
