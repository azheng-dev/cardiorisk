# ADR-013: Explainability strategy for v1 (KernelSHAP-headline + native sanity-checks + sum-back LR + cross-model agreement)

- **Status:** Accepted
- **Date:** 2026-05-06
- **Phase:** 2.5
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 / ADR-008 / ADR-009 / ADR-011 / ADR-012.

## Context

Phase 2.5 ships explainability for the four v1 risk models: TabICL (TFM), XGBoost (gradient boosted trees), L1 Logistic Regression with restricted-cubic-spline (RCS) feature expansion, and the Honours-architecture Ensemble (PyTorch 4-net mean-averaged DNN/CNN/LSTM/BiLSTM, ADR-012). The four models live behind a uniform `ModelWrapper` Protocol (`backend/cardiorisk/models/base.py`); any explainer choice has to work against `predict_proba` at minimum, ideally against the underlying model object too where a faster native algorithm exists.

The cross-cutting question is **which SHAP algorithm to run for which model.** The available choices and their trade-offs:

- **KernelSHAP** (`shap.KernelExplainer`) is model-agnostic. It approximates Shapley values by sampling coalitions of features and replacing the missing ones with values drawn from a "background" reference distribution. Works for *any* `f(x) -> probability` callable. Slow (O(n_features × background_size × evals_per_explanation)) — but the values it produces are computed by *the same algorithm* across every model, which is the property a fair cross-model comparison needs.
- **TreeSHAP** (`shap.TreeExplainer`) is exact and fast (polynomial in tree depth) for tree ensembles. Computes conditional expectations along tree paths, which is theoretically a different attribution scheme from KernelSHAP's interventional sampling — Janzing et al. 2020 ("Feature relevance quantification in explainable AI") and Aas et al. 2021 ("Explaining individual predictions when features are dependent") show the two can disagree, sometimes substantially, on correlated-feature problems.
- **Linear-coefficient attribution** for LR is exact: for an additive model `f(x) = β₀ + Σᵢ βᵢ * standardize(xᵢ)`, the per-feature SHAP value of feature `i` for instance `x` is `βᵢ * (standardize(xᵢ) - E[standardize(xᵢ)])`. No SHAP library call needed. The wrinkle: the Phase 2.2 LR pipeline applies a 4-knot RCS expansion to continuous features, so each original continuous feature becomes 3 spline-basis columns. To produce per-original-feature attributions comparable to the other three models, the spline-basis attributions must be summed back to the original feature.

Three further sub-decisions cluster around the SHAP algorithm choice:

1. **Background data sourcing.** KernelSHAP needs a representative reference distribution. Common practice is `shap.kmeans(X_train, k=50)` — k cluster medoids drawn from the per-fold training slice, which is fast and captures the marginal distribution sufficiently well for explanation purposes. Random sampling and larger k are alternatives.

2. **Subgroup-drift scope.** Per-stratum mean |SHAP value| deltas surface whether the model leans on different features for different patient subgroups. The honest question is which strata to audit. Phase 2.3b already established that LongBeachVA-sex (F=6), Switzerland-sex (F=10), and several age-band cells across folds fall below the `min_stratum_size` guard. Running drift analysis on those strata yields numerically defined but practically meaningless deltas.

3. **Cross-model agreement.** Phase 2.4 §8 Q4 explicitly asked for a Spearman rank-concordance matrix of feature importances across the four models. This is the single most interesting cross-model deliverable in Phase 2.5 — it answers "do the four models even agree on which features matter?", which is the prerequisite question to the Phase 3 agentic system trusting any one of them as a risk-driver narrative.

## Decision

The binding choices for Phase 2.5:

### 1. Explainer per model

**KernelSHAP for all four models is the headline cross-model comparison surface.** Computed against `model.predict_proba(...)[:, 1]` (the probability of CVD-positive class), with the same background sampler, the same coalition count, and the same random seed across all four models. This is the apples-to-apples comparison and the table that lands in `MODEL_CARD.md`.

**TreeSHAP additionally for XGBoost** as a sanity check. The XGBoost wrapper exposes the underlying booster via `model.booster_`; TreeSHAP runs against that and produces values in seconds rather than minutes. The deliverable is a side-by-side scatter (`xgboost_<fold>_treeshap_vs_kernelshap.png`) showing how closely the two algorithms agree on this dataset. If they disagree substantially (Spearman rank correlation < ~0.85 by feature), that is itself a finding worth documenting — it would mean the cross-model KernelSHAP comparison is doing real work.

**Linear-coefficient attribution additionally for LR**, with the spline-basis sum-back applied so the LR row of the cross-model comparison aligns on original feature names with TabICL / XGBoost / Ensemble. This is the *only* exact-by-construction attribution in the phase; KernelSHAP on LR should converge to the same values up to background-sampling noise. A second sanity-check figure (`lr_<fold>_summed_vs_spline.png`) shows the per-spline-basis attribution alongside the summed-back per-feature attribution, for reviewers who want to inspect how the RCS expansion uses each continuous feature.

### 2. Background data

`shap.kmeans(X_train_fold, k=50)` per fold. Fitted on the within-fold training slice (the same data the model was trained on), so the background distribution matches the model's training conditional. Standard SHAP recipe; fast enough that KernelSHAP runs against the four models complete in a tolerable wall-clock budget.

### 3. Local-explanation gallery (per model × fold)

Four representative test-set patients per (model × fold), drawn from four archetypes:

- **TP-high** — model predicts high risk, ground-truth positive. Pick the highest-risk correctly-predicted positive.
- **TP-low** — model predicts low risk, ground-truth positive. Pick the lowest-risk correctly-predicted positive (closest call that came out correct).
- **FN** — model predicts low risk, ground-truth positive (missed case). Pick the lowest-confidence missed positive (most over-confident error).
- **FP** — model predicts high risk, ground-truth negative (false alarm). Pick the highest-confidence false positive.

= 4 archetypes × 4 folds × 4 models = 64 waterfall PNGs. The four-archetype framing is the standard "where does the model agree and disagree with reality" surface; it's how clinicians read individual risk scores.

### 4. Subgroup-drift scope

**Auditable strata only.** Sex drift is computed for Cleveland and Hungarian only (LongBeachVA F=6 and Switzerland F=10 are below the `min_stratum_size` guard, same threshold Phase 2.3b uses). Age-band drift is computed wherever the per-stratum count is ≥ 50 (Cleveland 50–69 and <50; Hungarian 50–69 and <50; LongBeachVA 50–69; Switzerland 50–69). Strata that fall below the guard return NA in the JSON and are omitted from the per-(model × fold) drift figures.

This mirrors the Phase 2.3b §3 honesty discipline: we do not impute fairness-gap NA values. The same logic applies to feature-importance deltas — a "feature drift" computed on n=6 women is noise, not signal.

### 5. Cross-model agreement

Per fold + aggregate, a 4×4 Spearman rank correlation matrix of mean |SHAP value| feature rankings across the four models. Heatmap PNG per fold + aggregate-across-folds heatmap. The aggregate heatmap is the headline cross-model finding; the per-fold heatmaps show whether agreement is fold-stable or fold-dependent.

### 6. LR + RCS attribution detail

Per-spline-basis SHAP values are computed and stored; for cross-model comparison they are summed back to the original feature name. The summed-back values are the headline; the per-spline-basis values appear in a per-fold "LR-detail" figure for reviewers who want to inspect the nonlinearity. This is the *both* option — the headline is comparable, the detail is preserved.

### 7. Output surface

```
reports/v1/explainability/
  global_importance.json          # mean |SHAP| per (model × fold × feature)
  subgroup_drift.json             # per (model × fold × stratum × feature) deltas
  cross_model_agreement.json      # per-fold + aggregate Spearman matrices
  local_explanations.json         # 4 archetypes per (model × fold) with raw SHAP
reports/v1/figures/explainability/
  <model>_<fold>_global_bar.png            # 16 (4 models × 4 folds)
  <model>_<fold>_global_beeswarm.png       # 16
  <model>_<fold>_<archetype>_waterfall.png # 64 (4 archetypes × 4 folds × 4 models)
  <fold>_cross_model_agreement_heatmap.png # 4
  <fold>_aggregate_cross_model_agreement_heatmap.png   # 1
  <model>_<fold>_subgroup_drift_<stratum>.png          # ~24 (auditable strata only)
  xgboost_<fold>_treeshap_vs_kernelshap.png            # 4 (XGB sanity)
  lr_<fold>_summed_vs_spline.png                       # 4 (LR detail)
```

Total: ~140 PNGs. All committed for the full run; the smoke variant produces an even smaller subset that is gitignored under `reports/v1/figures/explainability/smoke/` (consistent with ADR-009's figure-commit policy).

### 8. Reproducibility

- Seed pinned to the project seed (`20260505`). KernelSHAP's `shap_values(..., nsamples=N)` uses an internal RNG; we patch via `numpy.random.default_rng` seeding before each call.
- Per (model × fold) outputs are deterministic to ~1e-5 across runs (KernelSHAP background sampling + coalition sampling means we can't expect bit-exact, same as the Ensemble determinism band from ADR-012).
- The orchestrator loads pre-trained models from `models/v1/` (per ADR-010); if absent, calls `train_v1.py` to regenerate. This makes Phase 2.5 a fast post-processing step rather than a re-train.

## Why KernelSHAP-everywhere and not the hybrid (TreeSHAP-for-XGB-native-everything-else)

The hybrid is what most papers do. The papers don't usually mention that they're comparing values produced by different algorithms across models. We are explicitly building a fair cross-model comparison surface; the methodological cost of running KernelSHAP on XGBoost (a few minutes per fold) is far smaller than the methodological cost of pretending TreeSHAP-XGBoost values are commensurable with KernelSHAP-TabICL values.

The hybrid is *additionally* run as a sanity check — TreeSHAP for XGBoost, native coefficients for LR — but those rows do not appear in the cross-model table. They appear only as "extra" deliverables documenting how much the algorithm choice matters for the two model classes where a native fast path exists.

## Why `shap.kmeans(50)` and not random sampling or larger k

Niculescu-Mizil & Caruana 2005 (the same paper that motivates the calibration choice in ADR-009 / ADR-012) shows non-parametric estimators on small slices over-fit; the same logic applies to KernelSHAP background distributions. k=50 cluster medoids gives the SHAP coalition sampler 50 representative reference points — empirically enough for stable Shapley estimates on tabular problems with ~20 features (which is our case after one-hot expansion). Larger k (100, 200) buys marginally smoother estimates at 2–4× the wall-clock; smaller k (random 20) introduces extra variance into the per-instance attributions. k=50 is the standard SHAP recipe and the right default for this dataset size.

## Why subgroup drift is restricted to auditable strata

Phase 2.3b explicitly does not impute the fairness gap when a stratum is below `min_stratum_size`. Reporting a "feature drift" between the 6 LongBeachVA women and the 194 LongBeachVA men would silently violate the same discipline. The explicit-NA-for-low-n behaviour is documented in the JSON output and in the figure titles; the alternative (run-on-everything-with-warnings) sounds more honest but the warnings are easy to miss in a portfolio context.

Where this rule excludes the LongBeachVA ≥70 stratum (n=16), the omission is consistent with the MODEL_CARD.md Phase 2.4 statement that this stratum is structurally out-of-scope for *all four* v1 models. We do not surface a SHAP-based explanation for an out-of-scope subgroup.

## Why Spearman rank concordance and not Pearson on raw values

SHAP values are on different scales for different models (TabICL outputs probability deltas, LR outputs log-odds deltas, etc.). Even after rescaling, *which features matter most* (the rank order) is the cross-model question worth asking, not *exactly how much each one contributes* (the value, which depends on the model's output scale). Spearman rank correlation is the standard answer.

## Consequences

**Positive:**
- A single cross-model surface (KernelSHAP) means the cross-model agreement matrix in `MODEL_CARD.md` is trustworthy as an apples-to-apples comparison.
- TreeSHAP and LR-coef sanity-checks document the attribution-algorithm-disagreement phenomenon explicitly. If they agree closely with KernelSHAP, the headline is reinforced; if they disagree, that is a Phase 2.5 finding worth flagging in the model card.
- The auditable-strata-only rule for subgroup drift extends Phase 2.3b's honesty discipline cleanly into Phase 2.5.
- Phase 3 (agentic system) gains a concrete per-feature attribution surface to drive risk-driver narrative drafting.

**Negative / open risks:**
- KernelSHAP wall-clock is the dominant cost: 4 models × 4 folds × ~150 test rows × 50 background = ~120,000 model evaluations per LODO sweep. TabICL forward passes are the slowest; we expect the full SHAP run to take ~30–60 minutes on the same M4 Pro hardware that ran Phase 2.4 LODO in ~37 minutes. CI runs the smoke variant only.
- KernelSHAP background sampling introduces non-bit-exact determinism; the same ~1e-5 tolerance band Phase 2.4 documents for the PyTorch Ensemble applies here.
- Shap 0.51.0 pulls numba + llvmlite (~38 MB combined) into the lock file. Already accepted in Phase 2.5 dependency change.

## Trigger to revisit

- If KernelSHAP wall-clock blows past ~90 minutes for the full LODO, switch to a sampled-test-rows variant (explain a stratified subset of test rows rather than every test row) and document the change.
- If TreeSHAP and KernelSHAP disagree by Spearman rank correlation < 0.7 on XGBoost, the cross-model KernelSHAP surface needs additional justification (or the analysis needs to surface the disagreement directly in `MODEL_CARD.md` rather than relegating it to a sanity-check figure).
- If a future v1.x adds a deep-learning model that supports `shap.DeepExplainer` (fast for `nn.Module` instances), the Ensemble row could swap KernelSHAP for DeepSHAP and the comparison story would need updating in a follow-up ADR.

## Amendment 2026-05-06: KernelSHAP wall-clock trigger fired

The first full-LODO sweep with the originally-planned settings (`nsamples=256`, no test-row cap, full per-fold test slice up to 303 rows on Cleveland) projected to ~3–4 hours wall-clock — well past the 90-minute trigger above. Two mitigations were applied before re-running:

1. **`nsamples` 256 → 128.** Smoke testing on the LR fold shows the resulting mean |SHAP value| rankings shift by at most one position per feature across the eight bottom-half features and are stable on the top six. Acceptable for a rank-based cross-model surface.
2. **Stratified test-row sampling, capped at `max_test_rows = 80` per (model × fold).** The four archetype rows are always included; the remaining `80 − 4 = 76` rows are stratified random-sampled by `y_test` to preserve per-fold prevalence. Both `n_test_full` and `n_test_explained` are recorded in `explanations_per_cell.json` so the cap is auditable.

Re-running with `nsamples=128`, `max_test_rows=80`, `background_k=50` (unchanged) completed in **~2h 20m wall-clock** on the M4 Pro hardware. Per-cell wall-clock breakdown logged in `/tmp/p25_full_run.log`: TabICL ≈ 13–15 min/cell (dominant cost — transformer forward passes), Ensemble ≈ 2 min/cell, XGBoost ≈ 10 s/cell, LR ≈ 2 s/cell.

The 80-row cap is a methodological cost worth flagging: the global-importance estimate is an average over 80 rows rather than the full per-fold test slice (123–303 rows depending on fold), which inflates the standard error of mean |SHAP| by roughly `√(n_full / 80)`. The Spearman-rank cross-model agreement matrix is not affected at the precision we care about (rank-stable to ~3% feature-position drift in our smoke testing). The local-explanation gallery is not affected at all because archetypes are always included.

If a future iteration needs the full per-fold test slice (e.g., for a Phase 3 risk-driver narrative that quotes per-row SHAP values for arbitrary patients), the orchestrator's `--max-test-rows 0` path explains every row at the same per-row cost — pay 4× the wall-clock, get 4× the rows. This is a configuration change, not a code change.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS. (KernelSHAP / TreeSHAP foundational paper.)
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature MI. (TreeSHAP exact algorithm.)
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causality problem. AISTATS. (Conditional vs interventional SHAP.)
- Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. AI. (Background-distribution sensitivity.)
- ADR-006 — risk-model architecture for v1.
- ADR-008 — preprocessing pipeline (RCS expansion details for the LR sum-back).
- ADR-009 — eval harness (`min_stratum_size` guard reused here).
- ADR-010 — model artefact storage (orchestrator loads from `models/v1/`).
- ADR-011 — TabICL supersedes TabPFN.
- ADR-012 — Honours-baseline reproduction (Ensemble wrapper that KernelSHAP runs against).
