"""Phase 2.5 explainability layer for the v1 risk models.

ADR-013 binds the cross-model surface to **KernelSHAP-everywhere**
(model-agnostic, single algorithm across all four models, comparable
per-feature attributions) with two supplementary native passes for
sanity:

- **TreeSHAP** for XGBoost (fast, exact for tree ensembles).
- **Analytic linear attribution** for the L1 LR with restricted-cubic-
  spline expansion, including the spline-basis-summed-back view that
  aligns LR rows with the other three models on the cross-model
  comparison table.

Module map:

- :mod:`cardiorisk.explainability.encoder` — shared "SHAP feature
  space" encoder. One-hot encodes the HFP categorical columns into a
  fixed numeric matrix that all four models can be queried against
  via a round-trip ``encode`` / ``decode`` pair. Categorical SHAP
  values are summed across their one-hot dummies before being
  reported per raw HFP feature.
- :mod:`cardiorisk.explainability.kernel_shap` — :class:`shap.KernelExplainer`
  wrapper. Uses :func:`shap.kmeans` (k=50) on the per-fold training
  slice as the background distribution per ADR-013 §"Background data".
- :mod:`cardiorisk.explainability.tree_shap` — :class:`shap.TreeExplainer`
  wrapper for XGBoost. Pulls the underlying booster out of the
  calibrated wrapper (``CalibratedClassifierCV(FrozenEstimator(XGBoostModel))``)
  and explains in post-XGBoost-preprocessing feature space.
- :mod:`cardiorisk.explainability.linear_attribution` — analytic per-
  feature SHAP for LR, with the spline-basis sum-back applied so the
  cross-model comparison is on raw HFP feature names.
- :mod:`cardiorisk.explainability.archetypes` — picks the four
  representative test patients per (model, fold): TP-high, TP-low,
  FN, FP. Used for the local-explanation gallery (waterfall plots).
- :mod:`cardiorisk.explainability.subgroup_drift` — per-stratum mean
  |SHAP| deltas, restricted to auditable strata only (Cleveland-sex,
  Hungarian-sex, age bands where n>=50). Mirrors Phase 2.3b's
  :func:`cardiorisk.eval.subgroup.stratified_metrics` discipline.
- :mod:`cardiorisk.explainability.cross_model_agreement` — per-fold
  + aggregate Spearman rank concordance matrices of mean |SHAP|
  feature rankings across the four models.
- :mod:`cardiorisk.explainability.figures` — matplotlib renderers
  (bar, beeswarm, waterfall, heatmap, sanity-scatter).
- :mod:`cardiorisk.explainability.orchestrator` — per (model x fold)
  loop. Loads pre-trained calibrated models from ``models/v1/``,
  runs the explainers, writes JSONs to
  ``reports/v1/explainability/`` and PNGs to
  ``reports/v1/figures/explainability/``.

The package depends on ``shap>=0.51`` (added in Phase 2.5; MIT-licensed,
pulls numba + cloudpickle + slicer transitively). Per ADR-013, all
randomness is pinned to the project seed (20260505) and KernelSHAP
output is deterministic to ~1e-5 across runs (the same band Phase 2.4
documents for the PyTorch Ensemble).
"""
