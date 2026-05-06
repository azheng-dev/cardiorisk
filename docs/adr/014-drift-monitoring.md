# ADR-014: Drift / monitoring strategy for v1 (PSI + KS, per-fold combined-pool reference, report-only)

- **Status:** Accepted
- **Date:** 2026-05-06
- **Phase:** 2.6
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 / ADR-008 / ADR-009 / ADR-010 / ADR-011 / ADR-012 / ADR-013.

## Context

Phase 2.6 ships a drift-monitoring layer for the four v1 risk models (TabICL, XGBoost, L1 LR, Honours-baseline Ensemble). The intent is to give the public repo a defensible "what would you put in front of a deployed model?" answer, without overreaching: the repo has no deployment producing live predictions, no labelled new data, and no operational alerting integrations. Anything Phase 2.6 ships has to be useful as a research-artefact deliverable, easy for a reviewer to reproduce, and honest about what it can and cannot detect.

Three cross-cutting questions structured the decision:

1. **What kind of drift do we measure?** The standard taxonomy:
   - **Input-feature drift** (covariate shift): the distribution of `X` changes between training and serving.
   - **Prediction drift**: the distribution of the model's `predict_proba` output changes — useful even when no labels are available, because a sudden shift in the predicted-risk histogram is a strong "something changed" signal.
   - **Concept drift**: the conditional `P(y | x)` changes — requires labelled new data to detect.
2. **Which statistical test for input-feature drift?** PSI, Kolmogorov-Smirnov (KS), Wasserstein (earth-mover) distance, Jensen-Shannon divergence, MMD (maximum mean discrepancy), domain-classifier ROC AUC. Each has different sensitivity / interpretability / cost trade-offs.
3. **What's the reference distribution?** A single combined-training-pool snapshot, four per-LODO-fold snapshots, or per-source snapshots.

Two further sub-decisions clustered around these:

4. **What does the system *do* when drift is detected?** Report-only (just write a JSON + dashboard) vs auto-block-deployment (refuse to serve a model whose prediction-drift PSI exceeds a threshold).
5. **What's the "current" data slice for the headline run?** The Phase-2.6 deliverable needs to produce a non-trivial drift number for the README, and the repo has no production traffic to score against.

## Decision

The binding choices for Phase 2.6:

### 1. Drift scope

**Input-feature drift (PSI on every numeric and categorical column) + prediction drift (PSI on `predict_proba`).** Concept drift is **deferred** to a future phase that has labelled new data — opening it now would either ship a placeholder API or invite a misleading metric.

KS is also computed on numeric features as a *sanity-only* companion: it surfaces a significance-test lens (p-value) which PSI alone does not. The dashboard does not visualise KS — it stays in the JSON for reviewers who want to cross-check. KS is skipped on categoricals (KS is for ordered distributions).

### 2. Statistical test choice

**PSI is the headline metric.** The full alternatives matrix considered:

- **PSI vs Wasserstein.** Wasserstein is scale-free and behaves more gracefully on continuous distributions, but the magnitudes ("a Wasserstein-1 of 0.4 means…") require the reader to internalise an unfamiliar yardstick. PSI carries a 30-year industry convention of severity bands (`< 0.10` stable / `0.10 – 0.25` moderate / `>= 0.25` major) that any senior engineer reviewing this repo will recognise immediately. The README audience optimisation wins.
- **PSI vs JS-divergence.** Same problem as Wasserstein on the readability axis, plus JS is not standard in monitoring stacks (it's more common in IR / topic-modelling).
- **PSI vs MMD / domain-classifier.** Both are *multivariate* drift detectors — they catch joint-distribution shifts that PSI misses by construction (PSI is per-feature). They also require either an RKHS choice (MMD) or a fitted classifier (domain-classifier), which is materially more code surface for a Phase 2.6 budget. Multivariate drift is on the radar (see "What this misses" below) but deferred until there's a deployment producing data that warrants it.

Severity bands are the standard published cut-points (`< 0.10` / `< 0.25` / otherwise). These are *industry convention*, **not** proven from first principles. The research doc surfaces this honestly.

ε-floor = `1e-6` for empty bins. Standard PSI hygiene; without it `log(0)` collapses the metric to NaN/inf the moment a current bin or reference bin happens to be empty (which is common at small n).

Numeric binning: 10 quantile (equal-frequency) bins on the reference. Quantile bins handle skewed features (e.g. `Oldpeak`, `Cholesterol`) far better than equal-width bins; 10 is the convention. **Bin-count sensitivity is documented in the research doc** — more bins = strictly more visible drift on the same shifted-distribution pair, which is a known PSI footgun.

### 3. Reference distribution

**One reference per LODO fold, built from the in-fold *training-pool combined* distribution** (i.e. the same three-source pool the fold's model was fit on).

- **Per-fold vs single combined.** Each LODO model was trained on a different combined-3-source pool (Cleveland's model on Hungarian + LongBeachVA + Switzerland, etc.). The drift baseline a model should be measured against is the data it was trained on — not the cross-fold union, because that conflates "drift between deployments" with "drift between which fold the model came from".
- **Per-fold vs per-source.** Per-source references are valuable in production (they let the system answer "is *this* hospital site's data drifting?") but at training time the model has already merged the sources; per-source reference would imply per-source models, which the v1 stack deliberately does not ship.

References are persisted as joblib artefacts at `models/v1/<source>_reference.joblib`, mirroring the model-artefact storage contract of ADR-010 (gitignored, rebuilt by `backend/scripts/build_reference.py` when needed). The Phase-2.6 orchestrator itself builds references *in-memory* during the LODO sweep — it does not require the on-disk references to exist — so the headline reproduction path is single-command (`compute_drift.py`) rather than two-command. The standalone `build_reference.py` script ships for the production-monitoring use case where new "current" data arrives at deploy-time and a separate process needs to score drift against the reference *the deployed model was trained on* without rerunning the LODO loop.

### 4. Threshold action

**Report-only.** Write a JSON + a dashboard PNG. Do not auto-block any deployment, do not page a human, do not write to any external system.

Rationale: Phase 2.6 is shipping into a repo with no deployment surface. Implementing auto-block now would be either a placeholder API (which is worse than nothing) or a hardcoded threshold whose epistemic status is "industry convention, not validated for this dataset" (which is dishonest). The decision-curve / DCA layer in Phase 2.3a sets the precedent for "ship the metric, document the threshold's status, defer the action": Phase 2.6 follows the same discipline.

When a productionisation phase later opens, it can add the action layer on top of the report-only scaffold without changing the metric definitions.

### 5. "Current" slice for the headline run

**Each fold's held-out LODO source.** The fold's model was trained on the other three sources; the held-out source is, by construction, the most realistic available proxy for "data the model has not seen". Re-using existing data also avoids inventing a synthetic shift fixture (which would be reporting on a shift the orchestrator itself produced).

The README headline becomes "this is the actual covariate shift each fold's model absorbs at deployment" — concrete, honest, and tied to data that already lives in the repo.

### 6. Driver layout

A standalone `cardiorisk/monitoring/` package + `backend/scripts/compute_drift.py` thin CLI wrapper, mirroring `backend/scripts/compute_explanations.py` (Phase 2.5). The CLI sets the macOS OpenMP guards (`OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` + `torch.set_num_threads(1)`) before importing any model wrapper — same xgboost/torch/TabICL OpenMP-deadlock defusing the explainability wrapper does.

`--smoke` and `--full` modes; smoke reuses the smoke-trained artefacts from `backend/scripts/train_v1.py --smoke` and runs in ~10 s on the GitHub Actions ubuntu-latest runner; full runs in ~10 s on the actual data because the maths is closed-form (no SHAP-style perturbation).

### 7. CI hook

`.github/workflows/ci.yml` gains one step in `test-python`, immediately after the `compute_explanations.py --smoke` step:

```yaml
- name: Smoke compute_drift.py --smoke (4 models, 1 LODO fold)
  run: uv run --project backend python backend/scripts/compute_drift.py --smoke
```

Reuses the smoke-trained artefacts from the same job's earlier `train_v1` step. The end-to-end test (`test_compute_drift.py`) asserts the smoke synthetic two-source split flags at least one feature outside `stable` for at least one model, which is the regression-canary the CI smoke test is for: a refactor that silently breaks PSI computation would no longer surface drift on a fixture that's known to contain it.

### 8. Output schema

Two JSON files + 16 dashboard PNGs (one per model × fold):

- `reports/v1/drift/per_fold.json` — list of 16 cells; each cell is `{held_out_source, model, n_current, severity_counts, per_feature: [...], prediction: {...}}`. `per_feature` entries hold `{feature, kind ∈ {numeric, categorical}, psi, severity ∈ {stable, moderate, major}, n_ref, n_cur, n_missing_cur, ks_statistic?, ks_p_value?}`. `prediction` holds `{model_name, psi, severity, n_ref, n_cur, mean_ref, mean_cur}`.
- `reports/v1/drift/aggregate.json` — `{config, n_cells, by_model: {model_name: {n_folds, severity_counts_total, prediction_psi_mean, prediction_psi_max}}}`.
- `reports/v1/figures/drift/<model>_<source>_dashboard.png` — three-pane layout: top PSI bar (severity-coloured), bottom-left ECDF overlay for the top-3 numeric drifted features, bottom-right `predict_proba` histogram overlay.

NaN / inf are coerced to JSON `null` via the same `_to_json_safe` helper the training and explainability drivers use.

## Consequences

**Positive:**

- One-command headline reproduction (`uv run --project backend python backend/scripts/compute_drift.py`); ~10 s wall clock on the real data.
- Cross-fold drift table joins cleanly with the existing reports layout (`reports/v1/{metrics_*, explainability/*, drift/*}`).
- README audience can read both the severity bands and the headline numbers without prior PSI familiarity (industry convention is widely known).
- CI smoke catches regressions that the unit tests on the primitives miss (full pipeline including model loading, joblib I/O, figure rendering).

**Negative / honest weaknesses:**

- **PSI is per-feature; it cannot detect joint-distribution shifts** (e.g. a shift in the *correlation* between `Age` and `MaxHR` while both marginals are unchanged). Multivariate drift detectors (MMD, domain-classifier) are deferred.
- **Bin-count sensitivity.** PSI on the same shifted pair generally grows with bin count. The 10-quantile-bin choice is the industry convention, not a derived optimum. The research doc reports the headline numbers under the chosen binning; a future phase that wants a sensitivity sweep can re-run with `--n-bins` once the orchestrator exposes that flag (currently a constant from `cardiorisk.monitoring.reference.DEFAULT_N_BINS`).
- **No time component.** PSI is a single point-in-time comparison. A real production monitor would also report a rolling-window PSI series. Out of scope for Phase 2.6.
- **KS reconstruction approximation.** The KS sanity-check uses synthetic reference samples reconstructed from the persisted bin midpoints + counts. This is exact for a discretised feature and a faithful approximation for continuous data given the same quantile binning, but it is not the same as running KS on the original raw reference samples. The orchestrator's headline numbers are the PSI values; KS is reported alongside but should be read as an *order-of-magnitude* sanity check rather than a definitive p-value. The alternative — persisting full reference samples — multiplies the reference-artefact size on disk for marginal sanity-check gain.
- **Severity thresholds are not validated for this dataset.** They are industry convention. A different domain (e.g. genomics, where features carry orders-of-magnitude wider variance) could need different cut-points. Out of scope to derive new ones here.

**Trigger to revisit:**

- A real deployment surface lands (Phase 8 deploy + promote, or an earlier productionisation phase). At that point the report-only stance should be revisited and either the severity thresholds get a domain-specific re-derivation or auto-block-deployment gets added (or both).
- New labelled data arrives (e.g. from a partner clinic). At that point a concept-drift module would graduate from deferred to in-scope.
- The cross-model comparison in `MODEL_CARD.md §6` ever finds that one of the four v1 models is materially more sensitive to drift than the others. That would be the trigger to add an MMD-based multivariate detector to triangulate which feature *interactions* are responsible, since per-feature PSI couldn't tell us that.

## Alternatives considered

(See **Decision** for the chosen path.)

- Wasserstein-everywhere (rejected: cost vs interpretability trade-off doesn't pay back the unfamiliar magnitude).
- JS-divergence (rejected: same as above; less standard in monitoring stacks).
- MMD or domain-classifier as the headline (rejected: more code surface; multivariate is genuinely useful but premature at Phase 2.6).
- Single combined reference (rejected: conflates "drift between deployments" with "drift between which-fold-the-model-came-from").
- Per-source reference (rejected: the v1 stack does not ship per-source models).
- Auto-block-deployment threshold action (deferred: no deployment to block).
- Synthetic-shift CI fixture (rejected for the headline; the smoke synthetic two-source dataset already produces non-trivial drift, no need to invent a separate shift fixture).

## References

- Population Stability Index — origins in credit-risk model governance; a clear modern walkthrough is the Evidently AI docs (`https://docs.evidentlyai.com/reference/all-metrics/data-drift/psi`).
- KS two-sample test — `scipy.stats.ks_2samp`.
- ADR-010 (model-artefact storage) — same joblib + local-artefact contract is reused for `*_reference.joblib`.
- ADR-013 (explainability strategy) — Phase 2.5 sets the orchestrator + CLI + smoke-fixture pattern this ADR mirrors.
- `docs/research/11-drift-design.md` — opinionated walkthrough of PSI vs alternatives, the bin-count footgun, and how to read the headline cross-source numbers.
