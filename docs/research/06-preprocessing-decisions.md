# Preprocessing pipeline — decisions and rationale (Phase 2.2)

This document explains *why* the Phase 2.2 preprocessing pipeline is shaped the way it is. It cross-references [`04-revised-design.md`](./04-revised-design.md) §3 (the binding design) and [`05-eda-findings.md`](./05-eda-findings.md) (the empirical evidence). The binding decision lives in [ADR-008](../adr/008-preprocessing-pipeline.md); this is the working document that explains the trade-offs.

## Layered structure

```
data/processed/combined.parquet
        │
        ▼
clean_for_modelling()                   <-- pure function, no fit/transform state
   • Cholesterol == 0 -> NaN
   • add 5 was_missing indicator columns
   • categorical NaN -> "Missing"
   • coerce Int64 -> float64
        │
        ▼
LODO splitter (cardiorisk.features.cv)
   • iter_lodo_folds: 4 folds, one per UCI source
   • within_fold_split: 80/10/10 train/val/calib per fold
        │
        ▼
sklearn Pipeline                        <-- stateful, fit per fold
   • make_tabpfn_pipeline()
   • make_xgboost_pipeline()
   • make_lr_pipeline()
   • make_woa_pipeline()
        │
        ▼
   numpy matrix passed to model in Phase 2.3
```

The crucial property is the *boundary* between the pure-function layer (safe to apply to any slice) and the stateful sklearn layer (must be fit on training data only). Sklearn's `fit` / `transform` API enforces this mechanically, which is why the headline pipelines use `Pipeline` / `ColumnTransformer` rather than bespoke pandas chains.

---

## 1. Why `Cholesterol == 0 → NaN` is the first step

EDA found the "0" sentinel in 100% of Switzerland rows and 24.5% of Long Beach VA rows. Treating it as a real measurement (as the prior Honours pipeline did) hands any sufficiently flexible model a perfect source-of-record detector — every Switzerland row has cholesterol exactly 0, no other source does. The model will pick this up and the LODO score will collapse the moment Switzerland is held out.

Converting to NaN restores the variable to its actual semantic ("we don't have a measurement") and lets the imputer in the per-model pipeline handle it the way it handles every other genuinely missing value. The downstream `Cholesterol_was_missing` indicator (added implicitly by `add_missingness_indicators` for the five problem columns) preserves the source-correlated signal *as a feature*, where the model can use it explicitly rather than learning it as a hidden artefact.

This is the single highest-impact decision in the preprocessing pipeline — it directly addresses pitfall #1 in `05-eda-findings.md` §3.

## 2. Why exactly five `was_missing` indicators (and not all eleven)

The EDA threshold is 10% missingness in at least one source. Five columns clear that bar: `RestingBP`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.

Lower thresholds bring diminishing returns:
- `FastingBS` and `RestingECG` have <1% missingness combined. An indicator there would be all zeros 99% of the time and would just add noise to L1 LR's regularisation budget.
- `Cholesterol` is special-cased upstream by the chol-zero cleaning step; its missingness becomes part of the standard NaN flow that all imputers handle uniformly.

This choice was surfaced explicitly to the user during planning and selected against the alternatives (all-with-any-missingness, all features). The five-column set is the most defensible: each indicator corresponds to a documented source-correlated missingness pattern in the EDA.

## 3. Why categorical NaN becomes the literal `"Missing"` category

Two alternatives we rejected:

- **Mode imputation** (replace NaN with the most common value): destroys the source-correlated missingness signal. ST_Slope is missing in 33.6% of rows overall; mode-imputing those would tell the model "every patient has Up-sloping ST" — which is wrong, and erases information.
- **`OneHotEncoder(handle_unknown='ignore')`** (NaN row becomes all-zeros): silently informative. The all-zeros pattern is detectable by any model with cross-feature interactions, so we'd be encoding the same signal as the explicit-`Missing` approach but in a less debuggable form.

The explicit `"Missing"` category gets one-hot-encoded as a dedicated `<col>_Missing` column. A test in `test_features_pipeline.py` verifies the round-trip end-to-end so this can't silently regress.

## 4. Why MissForest via sklearn `IterativeImputer` (not `miceforest`, not `missingpy`)

Three options were considered:

| Implementation | Verdict | Rationale |
|---|---|---|
| `sklearn.experimental.IterativeImputer` + `RandomForestRegressor` / `RandomForestClassifier` | **Chosen.** | Ships with scikit-learn (no new dep). Well-typed, well-tested, actively maintained. Slower than `miceforest` at large `n` but our largest LODO fold is ~720 rows — speed is irrelevant. |
| `miceforest` | Rejected for v1. | ~10x faster than sklearn IterativeImputer, supports multiple imputation natively. Adds a non-sklearn dep with a smaller maintainer pool; the speed advantage doesn't matter at our data scale. Worth revisiting if Phase-3 retrieval expands the dataset to >100k rows. |
| `missingpy` (the implementation the design doc originally cited) | Rejected. | Archived 2018, no Python 3.12 support. Listed only because the design doc mentioned it. |

The choice was surfaced to the user during planning. The `MISSFOREST_N_ESTIMATORS = 50` constant in `pipeline.py` is a deliberate cost trade-off — 50 trees per imputed feature is enough to converge on this data without running for minutes per LODO fold.

## 5. Why per-model pipelines (not a single shared pipeline)

The four headline / baseline models in [ADR-006](../adr/006-risk-model-architecture.md) have genuinely different preprocessing needs:

| Model | Imputation | Scaling | Spline | Why |
|---|---|---|---|---|
| TabPFN (headline) | None — passes NaN through | No | No | TabPFN's published protocol expects raw numerics. Imputing upstream loses the model's native missingness handling. |
| Calibrated XGBoost (white-box baseline) | MissForest (IterativeImputer + RF) | No | No | XGBoost handles NaN natively *too*, but the design doc applies the same MissForest treatment to XGBoost and WOA so the headline comparison isolates model choice from imputation choice. Tree models are scale-invariant, so no scaler. |
| L1 LR (transparency anchor) | Mean (continuous) / mode (binary) | Yes | RCS, 4 knots default | Mean/mode is the published "worst case" for LR (per `04-revised-design.md` §3.3). RCS captures non-linearity without exploding the parameter count under L1 regularisation. StandardScaler so L1 penalises columns on a comparable scale. |
| WOA-Ensemble (Honours baseline) | Same MissForest as XGBoost | Yes | No | The original Honours architecture (CNN+LSTM+ANN) was trained on Z-score-normalised inputs; we faithfully reproduce that for the Phase-2.4 head-to-head comparison. |

The factories share a deterministic prefix (the `clean_for_modelling` step applied once before the pipeline) but diverge in the stateful tail. This is the inverse of the more common "shared transformer + per-model head" pattern, but it matches the actual preprocessing requirements better and makes leakage review per-pipeline trivial.

## 6. Why restricted cubic splines (not sklearn's `SplineTransformer`)

`SplineTransformer(degree=3, n_knots=k)` emits `k + 2` B-spline basis columns per input. With 5 numeric features and 4 knots that's 30 columns — a real concern under L1 regularisation in a 920-row dataset.

RCS emits `k - 1` columns per input (15 columns for the same setup) and has the property that the function is **linear beyond the boundary knots**, which keeps extrapolation well-behaved on out-of-range LODO test rows. This matches the design-doc spec ("captures non-linearity without exploding the parameter count") and is what Harrell (2001) §2.4 recommends for clinical regression with limited n.

Default `n_knots=4` was chosen as the median of the design-doc range (3–5). The constant is exposed as a constructor argument so 2.3 can ablate.

## 7. Why one-hot encoding with `handle_unknown='infrequent_if_exist'`

LODO is the headline protocol, and under LODO it is *possible* (though rare in practice with the four UCI sources) for the test fold to contain a categorical level the training fold has not seen. The default `OneHotEncoder` behaviour would raise on transform; `handle_unknown='infrequent_if_exist'` maps the unknown to a fallback bucket instead. Sklearn 1.6+ ships this behaviour stably.

The fallback bucket is recorded in the OHE's `infrequent_categories_` so we can trace any LODO transform that hit it.

## 8. LODO + within-fold split: 80/10/10 train/val/calibration

`04-revised-design.md` §3.5 binds this. The 10% calibration slice is reserved for post-hoc isotonic calibration in Phase 2.3 (not used in 2.2). The 10% val slice is for hyperparameter tuning in 2.3.

A single deterministic split per fold (rather than nested CV) was chosen because nested CV at our data scale (600–800 train rows per fold) would burn ~5x the compute for a marginal stability gain. The seed is pinned globally (`SEED = 20260505` in both `cv.py` and `pipeline.py`).

## 9. Random K-fold sanity baseline

`iter_random_kfold` exists *only* to publish the gap between random K-fold and LODO discrimination metrics in Phase 2.3 — random K-fold mixes per-source case mix between train and test, so its AUROC is systematically optimistic. We expect the gap to be material, and the sanity test is the comparison that lets us prove it rather than assert it.

## 10. What's deliberately *not* in Phase 2.2

- **Class-imbalance handling.** The four UCI sources have heavy per-source class skew (2% positive prevalence in Cleveland, 92% in Switzerland) but the union is roughly 55% positive. We will revisit oversampling / class-weighting in Phase 2.3 if the headline metrics warrant it; reflexively SMOTE-ing in 2.2 would distort the LODO comparison.
- **Feature selection.** [ADR-006](../adr/006-risk-model-architecture.md) explicitly defers FS to "only if 2.3 metrics motivate it". The pipeline returns all 11 features (plus indicators).
- **Calibration.** Calibrators are part of the *model* layer in 2.3, not the preprocessing layer. The 10% calibration slice from `within_fold_split` is reserved for them.
- **Fairness preprocessing.** No reweighting or counterfactual augmentation in 2.2. Fairness audits live in Phase 2.3+ as evaluation, not preprocessing.

---

## What to verify when reviewing Phase 2.2

1. **Leakage protection.** `test_features_pipeline.py::test_lr_pipeline_imputer_means_differ_when_fit_on_disjoint_slices` and the corresponding XGBoost test directly probe this. If either ever flakes, treat it as an emergency: it means our LODO numbers are inflated.
2. **Indicator preservation under re-cleaning.** `test_preprocess.py::test_add_missingness_indicators_preserves_existing_indicator` is a regression test for a real bug found during implementation: the original code overwrote the indicator after categorical NaN had been replaced with `"Missing"`, silently zeroing out the missingness signal on the second call.
3. **Within-fold split arithmetic.** `test_features_cv.py::test_within_fold_split_default_proportions_are_eighty_ten_ten` asserts the 80/10/10 ratios within 1%.
4. **No NaN leaks past the model-imputing pipelines.** The XGBoost / LR / WOA pipelines must be NaN-free on output (TabPFN's must *not* be — it relies on NaN to know which entries are missing).
5. **Categorical `Missing` round-trip.** A categorical row that was NaN must show up as a `<col>_Missing` one-hot column post-fit. This is what makes the source-correlated missingness signal usable by the model.

## Phase 2.3 questions this preprocessing pipeline opens

- Whether to add class weighting to XGBoost / LR / WOA (TabPFN handles this internally) once we see the per-source confusion matrices.
- Whether the LR baseline benefits from `liblinear` vs `saga` solver under L1 — depends on output dimensionality after RCS expansion (currently 37 features post-pipeline).
- Whether to ablate `n_knots` ∈ {3, 4, 5} for the LR baseline as a robustness check, or accept 4 as the default and move on.
- Whether the random K-fold gap is large enough to warrant a dedicated section in the model card.
