# ADR-009 — Evaluation harness for v1 (metrics, DCA, bootstrap, calibration)

## Status

Accepted (Phase 2.3a). Supersedes the placeholder slot previously labelled "embeddings + retrieval architecture" — that decision moves to ADR-011 in the renumbered placeholder list.

## Context

The Phase-2.3 deliverable is a v1 risk model — three architectures (TabPFN headline, XGBoost+isotonic, L1 LR) trained under LODO-CV and reported with the headline metrics + bootstrap CIs + DCA + reliability + subgroup audits committed to in [`04-revised-design.md`](../research/04-revised-design.md) §5.

To keep that scope tractable and reviewable, Phase 2.3 was sub-phased:

- **2.3a (this ADR)**: build the evaluation harness — pure-function metrics, DCA, bootstrap CIs, reliability diagrams, subgroup audits, calibration wrapper. No models trained. The harness is testable in isolation against closed-form correctness checks.
- **2.3b (next ADR)**: train the three models, drive the harness across the LODO folds, produce the cross-model results table and the per-fold artefacts, ship the model artefact storage decision.

This ADR binds the design choices for 2.3a so 2.3b can plug models into a stable interface.

The opinionated walkthrough — what's in, what's out, why each choice — is in [`07-eval-design.md`](../research/07-eval-design.md). This ADR is the binding-decision summary; the walkthrough is the explanation.

## Decision

The Phase 2.3a evaluation harness consists of six modules under [`backend/cardiorisk/eval/`](../../backend/cardiorisk/eval/) plus one top-level module:

1. **`metrics.py`** — six headline scalar metrics from [`04-revised-design.md`](../research/04-revised-design.md) §5.1: AUROC, AUPRC, Brier, calibration slope, calibration intercept, sensitivity-at-specificity (85% and 90% targets). Plus a `headline_metrics()` one-shot returning a `HeadlineMetrics` dataclass.

2. **`dca.py`** — Decision-Curve Analysis per [Vickers & Elkin 2006](https://pubmed.ncbi.nlm.nih.gov/17099194/). Rolled in-house (~60 lines of formula) rather than pulling the [`dcurves`](https://pypi.org/project/dcurves/) package; trade-off documented in [`07-eval-design.md`](../research/07-eval-design.md). Default threshold sweep is 1%-99% (step 1%); explicit reporting at AusCVDRisk thresholds (5% and 10%).

3. **`bootstrap.py`** — non-parametric percentile-method bootstrap CIs. Default 2,000 resamples (per [`04-revised-design.md`](../research/04-revised-design.md) §5.1), pinned `SEED = 20260505`, deterministic across reruns. Drops degenerate (single-class) resamples; errors out if more than half of resamples were degenerate.

4. **`reliability.py`** — reliability diagrams returning `matplotlib.figure.Figure`. Default 10 quantile bins. Two-axis layout: calibration curve on top, predicted-probability histogram below. Caller controls saving / display.

5. **`subgroup.py`** — `stratified_metrics()` for any (metric, grouping) pair, returning per-stratum values + `fairness_gap = max - min`. Strata smaller than `min_stratum_size = 10` are reported with `NaN` value but excluded from the gap. Age-band cut-points are <50 / 50-69 / >=70 per [`04-revised-design.md`](../research/04-revised-design.md) §5.2.

6. **`calibration.py`** (top-level, not under `eval/`) — thin wrapper around `sklearn.calibration.CalibratedClassifierCV` with `sklearn.frozen.FrozenEstimator`. Two methods (`'isotonic'` for XGBoost, `'sigmoid'` for L1 LR per the user's Phase-2.3 decision); `calibrate_for_model(estimator, X, y, model_name=...)` dispatcher routes per [DEFAULT_METHOD_FOR_MODEL](../../backend/cardiorisk/calibration.py); TabPFN passes through unwrapped because it is calibrated by construction.

Bound design choices:

- **Headline metric set is the six listed.** No accuracy, no F1, no log-loss. Threshold-dependent reporting is via DCA at AusCVDRisk thresholds; threshold-independent operating-point reporting is via sensitivity-at-specificity.
- **Calibration regression is unregularised** (`C=1e10`, MLE coefficients). `penalty=None` and `C=np.inf` both route through the sklearn 1.8 deprecation path; finite-but-huge `C` is the supported equivalent.
- **DCA is rolled in-house**, not pulled from `dcurves`. Test-pinned against the published formula and a hand-computable worked example.
- **Bootstrap uses the percentile method**, not BCa. Trade-off documented; revisit if small-sample bias proves material on a per-fold basis.
- **Reliability bins default to quantile** (equal-population), not uniform. Quantile is the modern convention; histogram on the second axis preserves the predicted-probability distribution for diagnostic use.
- **Calibration uses `FrozenEstimator`**, not the deprecated `cv='prefit'`. sklearn 1.6+ API.
- **Per-model calibration dispatch lives in one place** (`DEFAULT_METHOD_FOR_MODEL`), not scattered across the training driver.

Out of scope for 2.3a (in 2.3b unless noted):

- Model wrappers (TabPFN / XGBoost / LR) — 2.3b.
- The training driver script — 2.3b.
- Optuna hyperparameter tuning — 2.3b.
- Model artefact storage decision (Hugging Face vs W&B vs local-with-rebuild) — ADR-010 in 2.3b.
- The model card — Phase 2.4 (after WOA reproduction provides the fourth comparison row).
- Net Reclassification Index (NRI) — never; DCA dominates per [Pepe et al. 2014](https://academic.oup.com/aje/article/181/4/263/137580).
- Calibration belt — superseded by reliability diagram + slope/intercept.

## Consequences

Positive:

- Eval harness is testable in isolation. 84 new tests in this PR (closed-form correctness + determinism + degenerate-input handling), all passing under `filterwarnings=["error"]`.
- 2.3b's model wrappers plug into a frozen interface — no eval changes required to add a fourth model in Phase 2.4 (WOA-Ensemble) or a fifth in any subsequent comparison.
- DCA + bootstrap + reliability all live behind small, model-agnostic functions. Phase 5's UI can call the same reliability_diagram() function to render an inline calibration plot for a clinician on demand.
- No new dependencies added. sklearn 1.8 + numpy + pandas + matplotlib already cover everything; TabPFN, XGBoost, Optuna land in 2.3b.
- The opinionated walkthrough in [`07-eval-design.md`](../research/07-eval-design.md) explains every "why we didn't" alongside every "why we did", which the model card can lean on.

Negative:

- Two PRs (2.3a + 2.3b) for what could have been one. Slightly more review overhead; mitigated by smaller per-PR diffs.
- The harness sits unused in `main` after 2.3a until 2.3b lands. Not load-bearing in production for the Phase 5 UI yet.
- Calibration with isotonic on a 60-row calibration slice is borderline; the harness exposes the choice but cannot mitigate the small-sample risk on its own. 2.3b will check the per-fold calibration slope on XGBoost and revisit if pathological.

## Alternatives considered

- **Single Phase 2.3 PR** (eval + models + driver + report). Rejected: ~2,500 lines is a heavy review lift and intermingles two reviewable concerns (math correctness in the harness, modelling correctness in the wrappers).
- **Three-PR slicing** (eval / LR+XGB / TabPFN+report). Rejected as more overhead than value: TabPFN is the headline and shouldn't ship behind LR/XGBoost; better to land both in 2.3b together.
- **Pull `dcurves`** for DCA. Rejected: the formula is short and well-defined, the package is small but adds a maintainer dependency, and rolling our own gives us auditable code on the page that the model card can reference directly.
- **BCa bootstrap** instead of percentile. Rejected for 2.3a: percentile is interpretable and matches reader expectations; BCa requires per-CI jackknife and adds compute. Will revisit if small-sample bias is material on a per-fold basis (we can swap implementations later without changing the API).
- **Manual logit regression** (numpy linear algebra) for calibration slope/intercept, avoiding sklearn entirely. Rejected: adds maintenance burden to dodge a deprecation that the `C=1e10` workaround already addresses cleanly.
- **`scikit-learn.calibration.calibration_curve`** for reliability bins. Rejected: returns only the (bin_means, bin_observed) arrays; we want the per-bin counts and the bin edges so the histogram axis can use them. Our implementation is small and adds the histogram axis the design doc commits to.

## Triggers to revisit

Revisit this ADR if any of the following becomes true:

- The 2.3b run shows the percentile bootstrap CIs are systematically biased or anti-conservative on the small folds (Switzerland: 123 rows, LongBeachVA: 200 rows). Fix: switch to BCa.
- The 60-row calibration slice produces erratic isotonic calibration on XGBoost (calibration slope wandering far from 1.0). Fix: switch XGBoost to sigmoid; document in 2.3b results.
- A user-facing dashboard in Phase 5 needs a streaming / incremental version of `reliability_bins()`. Fix: add a partial-fit variant; the dataclass shape doesn't need to change.
- The model card review (Phase 2.4) flags a missing metric per TRIPOD+AI 4.7 (e.g. integrated discrimination improvement). Fix: add to `metrics.py`; the harness API doesn't need to change.
- The fairness audit needs additional groupings (e.g. cardiovascular history, smoking status). Fix: pass new strata to `stratified_metrics()`; no module changes needed.

## Bypass log

(None for this ADR. Phase 2.3a was implemented end-to-end on a feature branch under the standard flow.)
