# ADR-008: Preprocessing pipeline for v1

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 2.2
- Supersedes: nothing
- Related: [ADR-006](./006-risk-model-architecture.md) (the model stack this pipeline feeds)

## Decision

The Phase 2.2 preprocessing layer for CardioRisk v1 is structured as a *pure-function cleaning prefix* applied once at data-load time, followed by *per-model sklearn pipelines* fit per LODO fold. The concrete choices:

1. **Cleaning step (deterministic, leakage-free, applied once):** `Cholesterol == 0 → NaN`; add `<col>_was_missing` indicators for the five EDA-flagged columns (`RestingBP`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`); replace categorical NaN with the literal string `"Missing"`; coerce nullable `Int64` columns to `float64` so sklearn passthrough works.
2. **Cross-validation:** Leave-One-Domain-Out via `sklearn.LeaveOneGroupOut` keyed on the `source` column (4 folds, one per UCI subset); within each fold's training rows, an 80/10/10 stratified `train_test_split` into train / val / calibration; a stratified random K-fold (5-fold) is published *only* as the sanity-baseline comparison demanded by [`04-revised-design.md`](../research/04-revised-design.md) §3.5.
3. **Imputation:** `sklearn.experimental.IterativeImputer` with `RandomForestRegressor(n_estimators=50)` for continuous features and `RandomForestClassifier(n_estimators=50)` for binary features (the canonical MissForest variant). Used by the XGBoost and WOA-Ensemble pipelines. The TabPFN pipeline imputes nothing (TabPFN handles NaN natively); the L1 LR pipeline uses mean / mode imputation (the published worst-case path from `04-revised-design.md` §3.3).
4. **Categorical encoding:** `OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')` for all five categorical features. The `"Missing"` category from step 1 gets a dedicated `<col>_Missing` one-hot column.
5. **Spline expansion:** Custom restricted-cubic-spline (Harrell §2.4) expander for the L1 LR pipeline. 4 knots default at quantile positions {5, 35, 65, 95}%, configurable `n_knots ∈ {3, 4, 5}`. RCS over sklearn's `SplineTransformer` because RCS produces `k - 1` columns per input vs `SplineTransformer`'s `k + 2`, which materially affects L1 regularisation in our 920-row regime.
6. **Scaling:** `StandardScaler` for the L1 LR (post-spline) and WOA-Ensemble pipelines; *no* scaling for TabPFN or XGBoost (both scale-invariant).
7. **Pinned RNG seed `SEED = 20260505`** for every stochastic component (IterativeImputer's RF, train_test_split, StratifiedKFold). Same constant as the synthetic-fixture generator and `04-revised-design.md` §3.5.

The pipeline factories live in `cardiorisk/features/pipeline.py` and return unfit `sklearn.pipeline.Pipeline` objects. The CV splitters live in `cardiorisk/features/cv.py`. The cleaning step lives in `cardiorisk/data/preprocess.py` (one layer up, because cleaning depends only on row-local information).

The full design walkthrough is in [`docs/research/06-preprocessing-decisions.md`](../research/06-preprocessing-decisions.md).

## Context

EDA in Phase 2.1 surfaced four pathologies in the combined HFP-schema dataset that any honest preprocessing layer must address:

1. **`Cholesterol == 0` is a missingness sentinel.** 100% of Switzerland rows and 24.5% of Long Beach VA rows. Treating it as a real measurement (as the prior Honours pipeline did) gives any sufficiently flexible model a perfect source-of-record detector and silently inflates random-K-fold metrics while collapsing LODO metrics.
2. **Source-correlated missingness on five other columns.** `RestingBP`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, and `ST_Slope` each have >10% missingness in at least one source. The pattern is informative — "this row has a missing ST_Slope" tells the model something about which hospital recorded the patient.
3. **Per-source class prevalence is wildly different.** Cleveland 2% positive, Switzerland 92% positive. Random K-fold mixes these case mixes between train and test and is mechanically optimistic; LODO is the honest evaluation.
4. **Mixed dtypes from the combine step.** Pandas nullable `Int64` for `Age` and `FastingBS` does not survive sklearn's `ColumnTransformer` passthrough as a numpy array — needed an explicit coercion to `float64` with `np.nan` sentinels.

[ADR-006](./006-risk-model-architecture.md) committed to four models (TabPFN headline, calibrated XGBoost, L1 LR, WOA-Ensemble baseline). Each has materially different preprocessing needs (TabPFN passes NaN through; XGBoost is scale-invariant; LR needs scaling and spline expansion; WOA-Ensemble needs scaling). A single shared pipeline would either underspecify the LR baseline or overspecify the TabPFN headline. Per-model factories with a shared deterministic prefix is the lowest-friction way to satisfy all four without leaking statistics across folds.

The hardest correctness constraint is **leakage protection**: any imputer / scaler / spline expander fit on the union of all sources (or even on a fold's test slice) silently inflates the LODO numbers. Sklearn's `Pipeline` / `fit` / `transform` boundary mechanically enforces this, which is why the stateful preprocessing lives inside sklearn pipelines rather than in a bespoke pandas chain.

## Consequences

### Positive

- **Leakage protection is structural, not procedural.** Two dedicated tests (`test_lr_pipeline_imputer_means_differ_when_fit_on_disjoint_slices` and the corresponding XGBoost test) directly probe the property; if either ever flakes, LODO numbers are compromised and CI breaks loudly.
- **The five `was_missing` indicators give the model a defensible way to learn source-correlated missingness as a feature.** This addresses pitfall #1 in `05-eda-findings.md` §3 head-on.
- **The 80/10/10 within-fold split is computed once per fold deterministically.** Phase 2.3 just slices `train`/`val`/`calib` indices and proceeds; no nested CV bookkeeping.
- **All four pipelines share the same `clean_for_modelling` prefix.** Any improvement to the prefix (e.g., adding a new cleaning step in v1.1) propagates uniformly without touching four factories.
- **The custom RCS implementation gives us a parsimonious basis under L1 regularisation** that sklearn's `SplineTransformer` cannot match without exploding the parameter count.

### Negative

- **A custom `RestrictedCubicSpline` transformer is one more thing to maintain.** Mitigated by 18 unit tests covering knot placement, output shape, linear-extrapolation property, and API contracts.
- **`IterativeImputer` is in `sklearn.experimental`** and the API is not 100% stable across sklearn versions. Mitigated by pinning `scikit-learn>=1.8` in `backend/pyproject.toml` and pinning `SEED = 20260505` for reproducibility.
- **Per-model pipelines mean four nearly-parallel factory functions.** Mitigated by the shared `_one_hot_encoder` / `_missforest_*_imputer` helpers that all four pipelines re-use.
- **`MISSFOREST_N_ESTIMATORS = 50` is a deliberate compute trade-off.** 500 trees per imputed feature would converge cleaner but quadruple the per-fold fit cost; 50 is enough at our `n` and is documented in the `pipeline.py` constant.
- **The `convergence` warning from `IterativeImputer` is suppressed in pytest config.** This is correct behaviour at our small `n` but does mean a real convergence regression in a future sklearn version would not surface as a test failure. We accept this as a small risk.

### Easier now

- Phase 2.3 model training: every model just calls `make_<model>_pipeline()`, then chains its model on the end. Zero preprocessing logic in the training scripts.
- Phase 2.4 WOA-Ensemble re-implementation: same MissForest + Z-score path the original Honours work used, so the comparison isolates architecture from preprocessing.
- Phase 2.5 SHAP: TreeSHAP / KernelSHAP can ingest the pipeline's `transform` output directly, with feature names from `pipe[-1].get_feature_names_out()`.

### Harder now

- Adding a model with substantially different preprocessing needs requires a new factory function, not a parameter on an existing one. Acceptable cost — this happens at most once per phase.
- The `clean_for_modelling` prefix is *outside* the sklearn pipeline, which means anyone applying these factories must remember to call it first. Mitigated by documenting it on every factory's docstring and by `test_features_pipeline.py` exclusively using `cleaned_frame` fixtures (so any drop of the cleaning step in test setup will fail the leakage test).

## Alternatives considered

### A. Single shared sklearn pipeline with a `model_type` parameter

Rejected. The branching logic inside the pipeline (impute-or-not, scale-or-not, spline-or-not) would itself need to be parameter-driven, producing a 200-line factory with multiple `if model_type == ...` arms. Per-model factories are 30 lines each, share the helpers, and let each pipeline stand alone in code review.

### B. `miceforest` instead of `IterativeImputer`

Considered. ~10x faster, supports multiple imputation natively, well-maintained. Rejected for v1 because the speed advantage is irrelevant at our `n` (largest LODO fold is 720 rows), and adding a non-sklearn dep with a smaller maintainer pool is the wrong trade for a public-facing portfolio repo. Worth revisiting if Phase 3+ retrieval expands the dataset.

### C. `missingpy.MissForest` (the implementation `04-revised-design.md` §3.3 originally cited)

Rejected. Archived 2018, no Python 3.12 support. Listed in the design doc only because the maintainer of the original Honours work used it; the modern equivalent is `IterativeImputer` with an RF estimator.

### D. `OneHotEncoder(handle_unknown='ignore')` with NaN-as-all-zeros for categoricals

Rejected. The all-zeros encoding for missing rows is silently informative — any model with cross-feature interactions will detect "all five OHE columns are zero" and learn the missingness signal in a non-debuggable form. The explicit `"Missing"` category emits a labelled column and is what `test_features_pipeline.py::test_categorical_missing_label_appears_as_one_hot_column` asserts on.

### E. Mode imputation for categorical missingness (the prior Honours approach)

Rejected. ST_Slope is missing in 33.6% of overall rows; mode-imputing that says "every patient has Up-sloping ST" and erases the source-correlated signal. Documented as an explicit pitfall in `05-eda-findings.md` §3.

### F. `sklearn.preprocessing.SplineTransformer` instead of custom RCS

Rejected for the LR pipeline. `SplineTransformer(degree=3, n_knots=k)` emits `k + 2` B-spline basis columns per input. With 5 numeric features and 4 knots that's 30 columns — a real concern under L1 regularisation in a 920-row dataset. RCS emits `k - 1` columns per input (15 for the same setup) and is linear beyond the boundary knots, which keeps extrapolation well-behaved. The custom implementation is small (~180 lines) and exhaustively tested.

### G. Add class-weighting / SMOTE in the preprocessing layer

Rejected for 2.2. Per-source class prevalence varies wildly (2% to 92%) but the union is roughly 55% positive. Reflexive oversampling in 2.2 would distort the LODO comparison. Class-imbalance handling, if needed, belongs in the model layer (2.3) where it can be applied per-fold based on actual eval results.

### H. Apply `clean_for_modelling` inside the sklearn pipeline as a `FunctionTransformer`

Considered. Would let the entire preprocessing chain live in a single `Pipeline` object. Rejected because the cleaning steps depend only on each row's own values (no fit/transform distinction needed), and applying them once at data-load time is simpler, faster, and easier to debug. Putting them inside the pipeline would mean every `pipe.transform(X_test)` call re-runs the cleaning, which is wasted work.

## Trigger to revisit

Re-open this ADR (with a superseding ADR) if any of the following becomes true:

- The dataset changes (e.g., HFP → MIMIC-IV, or HFP augmented with a Phase-3-derived AU cohort) — likely necessitates re-deriving the missingness-indicator threshold and possibly different categorical handling.
- A future sklearn version (≥1.10) makes `IterativeImputer` non-experimental and changes its API. We'll need to revalidate that the leakage tests still hold.
- Phase 2.3 reveals that the random-K-fold vs LODO gap is small (<0.02 AUROC). That would change the headline framing in the model card and possibly justify retiring the K-fold sanity baseline.
- A peer-reviewed reproduction publishes evidence that for HFP-class datasets, MissForest is materially worse than `KNNImputer` or a learned generative imputer (e.g., MIWAE). Currently no such evidence exists.
- We add a sixth feature with >10% missingness and need to re-derive the indicator-column list.
- Class imbalance handling becomes necessary in 2.3 — would add a step to the pipeline factories rather than a separate ADR if scoped narrowly.

## References

- [`docs/research/04-revised-design.md`](../research/04-revised-design.md) — the binding design this ADR implements.
- [`docs/research/05-eda-findings.md`](../research/05-eda-findings.md) — the empirical findings that drive the cleaning and indicator choices.
- [`docs/research/06-preprocessing-decisions.md`](../research/06-preprocessing-decisions.md) — the working document with the full rationale.
- [Harrell (2001) — *Regression Modeling Strategies* §2.4](https://link.springer.com/book/10.1007/978-1-4757-3462-1) — RCS basis and recommended knot quantiles.
- [Stekhoven & Bühlmann 2012 — MissForest](https://academic.oup.com/bioinformatics/article/28/1/112/219101) — the imputation method `IterativeImputer` + RF approximates.
- [scikit-learn 1.8 docs — `IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [`LeaveOneGroupOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).
