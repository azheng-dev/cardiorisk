# Evaluation harness design (Phase 2.3a)

This is the *why* behind the evaluation harness shipped in Phase 2.3a. The *what* lives in [ADR-009](../adr/009-eval-harness.md) (binding decisions) and the [`cardiorisk.eval`](../../backend/cardiorisk/eval/) docstrings (per-function contracts).

## Scope

Phase 2.3a delivers the evaluation machinery. **No models are trained in this phase.** Per the Phase 2.3 sub-phasing decision, models, hyperparameter tuning, and the actual training driver land in Phase 2.3b. Splitting it this way means the eval harness can be reasoned about as pure math + pure plotting, divorced from model dependencies (TabPFN brings PyTorch; XGBoost brings its own native lib; both arrive in 2.3b).

The harness exists to satisfy the deliverables in [`04-revised-design.md`](./04-revised-design.md) §5:

- §5.1 — six headline metrics per LODO fold + bootstrapped 95% CIs.
- §5.2 — subgroup performance by sex and age band, with a fairness-gap summary.
- §5.3 — per-source breakdown (no new code; just per-fold reporting using the same harness).
- §5.4 — the cross-model comparison table.
- §5.5 — TRIPOD+AI checklist mapping (deferred to the model card in Phase 2.4).

It also satisfies the design-doc commitments around clinical positioning ([`04-revised-design.md`](./04-revised-design.md) §6): Decision-Curve Analysis at AusCVDRisk thresholds (5%, 10%) is built in, so the eventual results writeup can speak directly to *clinical utility* rather than just AUROC.

## Headline metrics: what's in, what's out

In: AUROC, AUPRC, Brier, calibration slope, calibration intercept, sensitivity at 85% specificity, sensitivity at 90% specificity. Six numbers per fold per model.

Out (deliberately):

- **Accuracy / F1.** Threshold-dependent, base-rate-sensitive, and not what clinicians act on. Net benefit is the threshold-dependent metric we report.
- **ROC plots without numbers.** ROC curves alone are easy to over-read; we report the scalar (AUROC + bootstrapped CI) and the operating-point sensitivities, not a curve.
- **Calibration-in-the-large only.** Mean-predicted vs base-rate is part of calibration intercept; we add slope so we catch the over-/under-confidence pattern Vach et al. show is otherwise missed.

### On Brier vs alternatives

Brier is a strictly proper scoring rule (jointly proper for discrimination + calibration). Log-loss is the alternative. Brier was chosen over log-loss because:

- Brier is bounded in [0, 1] for binary problems; log-loss is unbounded above. Brier intervals from bootstrap are stable.
- Brier penalises overconfident wrong predictions less aggressively than log-loss. With the small per-source samples in HFP (Switzerland has 8 negatives, LongBeachVA has 17 negatives), log-loss is dominated by a handful of confident misclassifications. Brier is more robust.
- Brier decomposes cleanly into reliability + resolution + uncertainty (Murphy 1973), which makes it easier to explain *why* one model beats another in the model card.

### On the calibration slope/intercept implementation

The standard approach (Steyerberg et al., 2010) is to fit a logistic regression of the outcome on the model's logit-predictions and report the coefficient (slope) + intercept. We do exactly that, with a few details that matter:

- We clip predicted probabilities to `[1e-15, 1 - 1e-15]` before taking the logit. Avoids `±inf` blowing up the regression on perfectly-confident predictions.
- We use `LogisticRegression(C=1e10)` to make the regression effectively unregularised. We can't use `penalty=None` (deprecated in sklearn 1.8) or `C=np.inf` (routes to the deprecated path). Finite-but-huge C gives the MLE without the deprecation warning.
- We return `NaN` for slope and intercept when `y_true` is single-class. Not an error; just a degenerate case the caller should handle.

### On sensitivity-at-specificity rather than fixed thresholds

The Australian risk-band thresholds are calibrated probabilities of an event over a horizon (5-year for PREDICT-1°). Our model's raw probabilities aren't directly comparable to that horizon (HFP doesn't have time-to-event data; it's a binary disease-presence label). So *fixed thresholds* would be apples-to-oranges. Sensitivity-at-specificity is threshold-free in the sense that the operating point is selected per model on the calibration slice, then evaluated on the held-out fold.

We report two specificity targets (85% and 90%) so the model card can say something honest about both screening (favour sensitivity) and rule-out (favour specificity) modes.

## Decision-Curve Analysis

Vickers & Elkin (2006) is the canonical reference. The formula is small enough to roll our own (~60 lines in [`dca.py`](../../backend/cardiorisk/eval/dca.py)) rather than pull the [`dcurves`](https://pypi.org/project/dcurves/) package. Trade-off:

- Pros of rolling our own: no new dependency; the formula is on the page (auditable); we test it against the published worked example in [test_eval_dca.py](../../backend/tests/test_eval_dca.py); easier to extend (e.g. censoring later).
- Cons: marginally more code we own. Mitigated by the closed-form unit tests (a perfect predictor's NB equals the prevalence at every threshold; treat-all NB equals `prevalence - (1 - prevalence) * (p_t / (1 - p_t))`; etc.).

We default to a 1%–99% threshold sweep (step 1%) for the curve plot, with explicit per-threshold reporting at the AusCVDRisk decision points: 5% (low / intermediate boundary) and 10% (intermediate / high boundary).

A model is "clinically useful" at threshold p_t iff it dominates **both** treat-all and treat-none at that p_t. The `DCACurve.is_useful_at(t)` helper encodes that decision rule.

## Bootstrap CIs: percentile method, 2,000 resamples, pinned seed

[`04-revised-design.md`](./04-revised-design.md) §5.1 commits to 2,000 resamples. The implementation:

- **Percentile method**, not BCa (bias-corrected and accelerated). Why: BCa requires a jackknife to estimate the acceleration coefficient; for our metrics that's extra computation per CI. Percentile is the "honest 95% interval" interpretation that matches what readers expect, has known small-sample bias that we'll document in the model card if it becomes material, and is what `scipy.stats.bootstrap` defaults to anyway.
- **Case-resampling**: rows are the resampling unit, with replacement. The metric is recomputed on each resample. For ROC-style metrics this is the appropriate unit; for fold-level metrics the natural unit would be folds, but with only 4 LODO folds the fold-level bootstrap has a degenerate sample size.
- **Determinism**: seeded with the repo-wide `SEED = 20260505`. Same `(y_true, y_proba, n_resamples)` always yields the same CI across reruns and machines.
- **Degenerate-resample handling**: a resample that happens to be all-one-class causes some metrics (AUROC, AUPRC) to return NaN. The bootstrap_ci function drops NaN resamples, then errors out if more than half were NaN (signals the input is too small or too imbalanced for percentile bootstrap on that metric).

We considered BCa and stratified bootstrap; both are credible alternatives. The model card will document the choice and note that for the LongBeachVA fold (92% positive) the bootstrap CIs are wider than the fold's degeneracy warrants — a known limitation called out per TRIPOD+AI 5.4.

## Reliability diagrams

Two binning strategies supported:

- **Quantile** (default): equal-population bins. Every bin gets the same number of rows; bin widths vary. Statistically more reliable per bin (each bin has the same `n` so the bin-level CI is comparable across bins).
- **Uniform**: equal-width bins on `[0, 1]`. Easier to read; rare-probability bins can be empty / tiny / unstable.

We default to quantile + 10 deciles, matching modern conventions ([Niculescu-Mizil & Caruana 2005](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html)).

The function returns the matplotlib `Figure` object — we never call `plt.show()`. Callers decide how to render it (notebook inline, save to PNG for the model card, embed in a Streamlit panel later, etc.).

The figure has two stacked axes: top is the calibration curve (mean-predicted vs observed-rate, with a `y=x` reference); bottom is a histogram of the predicted-probability distribution. The histogram matters: a model that's superficially well-calibrated but only ever predicts probabilities in `[0.4, 0.6]` is, in practice, useless for the patient-band decisions the AusCVDRisk score drives — but the calibration curve alone wouldn't show it.

## Subgroup audits and fairness-gap

Per TRIPOD+AI §5.2 and [`04-revised-design.md`](./04-revised-design.md) §5.2, we stratify the headline metrics by:

- **Sex** (Male / Female from the HFP schema).
- **Age band** (<50 / 50–69 / ≥70 — the cut-points the design doc commits to).

The fairness gap is `max - min` of the metric across strata. Strata smaller than `min_stratum_size = 10` are reported with a `NaN` value (so they're visible) but excluded from the gap (so the gap isn't dominated by sampling noise from a single rare stratum).

The design doc commits to a paragraph in the model card for any fairness gap > 5 percentage points on sensitivity. The harness gives us the number; the model card does the explaining (Phase 2.4).

## Calibration wrapper

[`cardiorisk.calibration`](../../backend/cardiorisk/calibration.py) is a thin wrapper around sklearn's `CalibratedClassifierCV`, with two design choices:

1. **`FrozenEstimator` over `cv='prefit'`**. sklearn 1.6 deprecated `cv='prefit'` in favour of wrapping the base estimator in `FrozenEstimator`. Same semantics; the new API is what 2026 sklearn expects.
2. **Method dispatch by model name**. `calibrate_for_model(estimator, X, y, model_name='xgboost')` looks up the right method (isotonic for XGBoost, sigmoid for LR per the Phase 2.3 design); TabPFN passes through unwrapped because it's calibrated by construction. This keeps the per-model decisions in one place where the training driver doesn't have to know about them.

The 80/10/10 within-fold split from [`cardiorisk.features.cv.within_fold_split`](../../backend/cardiorisk/features/cv.py) (Phase 2.2) gives us a 10% calibration slice — about 60 rows per LODO fold. That's borderline for isotonic (which is non-parametric and benefits from more data) and comfortable for sigmoid (one parameter to fit). The choice of isotonic-for-XGBoost is per ADR-006; we accept the small-sample risk and will revisit if the calibration slope on XGBoost looks pathological.

## What this harness does *not* do

- **Score the model artefact.** That's the job of the training driver in Phase 2.3b.
- **Run the actual reproduction.** Same.
- **Net reclassification index (NRI).** [Pencina 2008](https://onlinelibrary.wiley.com/doi/10.1002/sim.2929) introduced it; [Kerr et al. 2014](https://academic.oup.com/aje/article/180/3/318/2739147) and [Pepe et al. 2014](https://academic.oup.com/aje/article/181/4/263/137580) showed it has serious interpretation pitfalls and is dominated by DCA for the same purpose. We use DCA instead.
- **Calibration belt** (CORINE-style). Adds plotting machinery for marginal value over the reliability diagram + calibration slope/intercept already reported. Skipped.
- **Calibration intercept on the original probability scale.** We report it on the logit scale (the `LogisticRegression.intercept_`), which is the convention since Steyerberg 2010. The model card will explain the units.
- **Multiple comparison correction across folds.** With 4 LODO folds and 6 metrics that's 24 CIs per model; Bonferroni would balloon them to ~99% intervals each and lose all interpretability. We report 95% CIs and let the reader judge.

## Verification points for the reviewer

If you're reviewing this PR:

1. **Determinism**: re-run `pytest tests/test_eval_bootstrap.py::test_bootstrap_is_deterministic_under_pinned_seed`; the CI must be byte-identical across runs.
2. **Closed-form correctness**: `tests/test_eval_metrics.py` covers the perfect predictor / random predictor / base-rate predictor cases for every metric.
3. **DCA worked example**: `tests/test_eval_dca.py::test_net_benefit_matches_published_formula` checks the formula against a hand-computable case (4 positives + 4 negatives at threshold 0.5).
4. **No leakage in calibration**: `tests/test_calibration.py::test_calibrate_does_not_refit_base_estimator` asserts `coef_` and `intercept_` of the base estimator are byte-identical before and after calibration.
5. **Reliability diagram saves to PNG**: `tests/test_eval_reliability.py::test_reliability_diagram_can_be_saved_to_png` round-trips a figure to disk and asserts non-trivial size.

The next phase (2.3b) will train the three models, drive the harness across the 4 LODO folds, and produce the cross-model results table from [`04-revised-design.md`](./04-revised-design.md) §5.4.
