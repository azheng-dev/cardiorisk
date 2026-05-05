"""Feature engineering and cross-validation infrastructure.

Layer responsibilities:

- :mod:`cardiorisk.features.cv` — Leave-One-Domain-Out cross-validation
  splitter (the headline Phase-2.3 protocol), within-fold 80/10/10
  train/val/calibration sub-split, and a stratified random K-fold
  baseline used only as the "look how badly random K-fold inflates
  numbers" comparison in `04-revised-design.md` §3.5.
- :mod:`cardiorisk.features.spline` — restricted-cubic-spline expansion
  for the L1 logistic regression baseline.
- :mod:`cardiorisk.features.pipeline` — sklearn ``Pipeline`` /
  ``ColumnTransformer`` factories per model: TabPFN (NaN passthrough),
  XGBoost (MissForest impute), LR (mean/mode + spline + standardize),
  WOA-Ensemble (same shape as XGBoost).

The deterministic, no-fit cleaning steps (zero-cholesterol -> NaN,
missingness indicators, categorical-NaN -> "Missing") live one layer up
in :mod:`cardiorisk.data.preprocess` and are called by every pipeline
factory in this package.

Phase 2.3 will plug models into the factories from this package; Phase
2.5 will add ``cardiorisk.explain`` next to it.
"""
