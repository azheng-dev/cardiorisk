# 11 — Drift / monitoring design (Phase 2.6)

> **TL;DR:** Phase 2.6 ships per-feature input-drift PSI + prediction-drift PSI for the four v1 models, scored per LODO fold against the in-fold training pool, with each fold's held-out source used as the "current" slice. The result is a concrete, honest map of how much the models' predictions move when faced with cross-source covariate shift — and an early warning that two of the four model families (TabICL and the Honours-Ensemble) move much harder than the other two (XGBoost and L1 LR). Severity bands and ε-floor follow ADR-014; thresholds are industry convention, not derived from this dataset, and the doc says so. Multivariate drift, time-series drift, concept drift, and any auto-blocking are explicitly out of scope.

This document is the prose companion to [ADR-014](../adr/014-drift-monitoring.md). The ADR contains the binding decision; this doc walks through the *why* in more detail than the ADR template comfortably hosts, and reports the headline numbers from the Phase-2.6 LODO sweep.

---

## 1. Why PSI (and not Wasserstein, JS, or MMD)

PSI is the headline metric for Phase 2.6. The full alternatives matrix:

| Test | What it measures | Pros | Cons |
|---|---|---|---|
| **PSI** | Per-feature distribution shift, binned | 30 yr industry convention; widely-cited severity bands; trivial to compute closed-form on persisted bin counts; no library dependency beyond numpy | Per-feature only (joint shifts invisible); bin-count sensitive (more bins = strictly more visible drift on the same shifted pair) |
| **Wasserstein-1** | Earth-mover distance | Scale-free; well-behaved on continuous distributions | Magnitudes ("a Wasserstein-1 of 0.4 means…") need the reader to internalise an unfamiliar yardstick |
| **JS-divergence** | Symmetrised KL | Bounded in `[0, log 2]`; no need for ε-floor | Same readability problem; not standard in monitoring stacks (more common in IR / topic modelling) |
| **KS two-sample** | Sup-norm of ECDF difference | Comes with a p-value (significance lens); rank-based, robust to scale | Per-feature; numeric only; p-value is a frequentist construct that production monitoring rarely uses operationally |
| **MMD** | Kernel embedding distance | **Multivariate**: catches joint-distribution shifts | Requires an RKHS / kernel choice; materially more code; harder to communicate |
| **Domain-classifier ROC AUC** | Train classifier to separate ref vs cur | **Multivariate**; produces a single 0.5–1.0 score; intuitive | Requires training a model just to detect drift; harder to assert "no drift" (50% AUC is the ideal but you can't tell the difference between "no drift" and "classifier didn't fit") |

PSI wins on the **README audience optimisation**. A senior engineer skimming the repo will recognise the `< 0.10` / `0.10 – 0.25` / `>= 0.25` severity bands immediately; the same person will not, off the top of their head, know whether a Wasserstein-1 of 0.4 is a lot. KS sits alongside as a sanity-only companion — it lives in the JSON for reviewers who want to cross-check the PSI number against a significance test, but it doesn't appear on the dashboard.

The **multivariate gap is real and acknowledged.** PSI cannot detect a shift in the *correlation* between `Age` and `MaxHR` if both marginals are unchanged. This is the kind of drift MMD or a domain-classifier would catch and PSI would miss by construction. A future phase that has a deployment producing real traffic — and therefore a budget to spend on more sophisticated detectors — should add MMD as a multivariate triangulation. For Phase 2.6, per-feature PSI on a static research artefact is the right depth-vs-cost trade-off.

---

## 2. Why per-fold combined-pool reference (and not single combined / per-source)

Each Phase-2.3b LODO model was trained on a *different* combined-3-source pool (Cleveland's model on Hungarian + LongBeachVA + Switzerland; Hungarian's model on Cleveland + LongBeachVA + Switzerland; etc.). The drift baseline a model should be measured against is the data it was *fit on*. Anything else conflates two different signals:

- A **single combined reference** built from all four sources would mix in the held-out source's distribution — precisely the distribution we're trying to detect drift against. PSI would be artificially deflated for every fold.
- A **per-source reference** would make sense in production (it lets the system answer "is *this* hospital site's data drifting?"), but at training time the v1 models have already merged the sources; building per-source references at training time would imply per-source models, which the v1 stack deliberately does not ship.

Per-fold combined-pool reference is the only choice that keeps "is the input distribution different from what the model saw" honest.

The references themselves are persisted as joblib artefacts at `models/v1/<source>_reference.joblib` (gitignored, mirroring [ADR-010](../adr/010-model-artefact-storage.md)). The Phase-2.6 orchestrator builds equivalent references **in memory** during its own LODO sweep — it doesn't require the on-disk references to exist — so the headline reproduction path stays one command. The standalone [`backend/scripts/build_reference.py`](../../backend/scripts/build_reference.py) ships for the production-monitoring use case where a separate process needs to score new traffic against the reference *the deployed model was trained on* without rerunning the LODO loop.

---

## 3. Why "current" = held-out LODO source

Phase 2.6 needs to produce non-trivial drift numbers for the README. The repo has no production traffic and no synthetic-shift fixture would be more than a demonstration of the orchestrator producing the shift it was told to.

The held-out LODO source already lives in the data layer, was deliberately excluded from each fold's training pool, and represents an honest stand-in for "data the model has not seen". Re-using it for the headline run gives:

- A drift number that's **tied to data already in the repo**, so any reviewer can re-derive it.
- A narrative that's **already part of the v1 story**: every fold's Phase-2.3b headline metrics were computed on this same held-out source, so the drift report is just answering "and *why* does the AUROC drop on Switzerland?" with a concrete distribution-level answer.
- **Zero new fixtures** to maintain.

The smoke fixture (gitignored) uses two synthetic sources from the deterministic generator at [`cardiorisk/data/synthetic.py`](../../backend/cardiorisk/data/synthetic.py) so CI has a reliable trivially-differentiable two-pool dataset to verify the orchestrator can flag drift.

---

## 4. Headline cross-source drift numbers

Computed by `uv run --project backend python backend/scripts/compute_drift.py` on the committed combined parquet (`data/processed/combined.parquet`), full mode (4 LODO folds × 4 models × 10 quantile bins), wall clock ~30 seconds on an M4 Pro. Outputs at [`reports/v1/drift/per_fold.json`](../../reports/v1/drift/per_fold.json), [`reports/v1/drift/aggregate.json`](../../reports/v1/drift/aggregate.json), and 16 dashboard PNGs under [`reports/v1/figures/drift/`](../../reports/v1/figures/drift/).

### 4.1 Per-feature drift is identical across models within a fold

Per-fold severity counts (out of 11 features per fold):

| Held-out source | n_train | n_test | stable | moderate | major | Top-3 drifted features |
|---|---:|---:|---:|---:|---:|---|
| Cleveland | 617 | 303 | 4 | 2 | 5 | `ST_Slope` (PSI=7.06), `RestingECG` (1.84), `ExerciseAngina` (1.13) |
| Hungarian | 626 | 294 | 3 | 2 | 6 | `Age` (2.23), `Oldpeak` (1.42), `ST_Slope` (1.41) |
| LongBeachVA | 720 | 200 | 2 | 1 | 8 | `ExerciseAngina` (1.77), `Age` (0.78), `MaxHR` (0.68) |
| Switzerland | 797 | 123 | 2 | 3 | 6 | `MaxHR` (0.60), `ChestPainType` (0.54), `Oldpeak` (0.42) |

The per-feature severity counts are identical across the four models within each fold, by construction — input-feature drift is a property of the data, not the model. The number that *does* differ across models is the prediction-drift PSI (next section).

The **headline finding**: every LODO fold has between 5 and 8 features in `major` band. This is a *substantial* covariate shift between the UCI subsets, and matches what the Phase-2.1 EDA findings flagged informally (heterogeneous missingness, source-specific encoding conventions, different patient mixes per site). PSI puts numbers on it.

The **standout**: `ST_Slope` PSI = 7.06 on the Cleveland fold. That number is high enough that a production monitor would page someone; here it tells a recoverable story — Cleveland's `ST_Slope` distribution is meaningfully different from the union of the other three sources. The Phase-2.5 explainability run already named `ST_Slope` as a top-3 mean-|SHAP| feature for every model; the Phase-2.6 drift number says *that exact feature is the one most distorted at the cross-source boundary*. Phase 3's risk-driver narrative drafter is going to have to handle this honestly.

### 4.2 Prediction-drift PSI varies dramatically across model families

| Model | mean prediction-PSI (across 4 folds) | max prediction-PSI | mean severity |
|---|---:|---:|---|
| **TabICL** | 1.57 | 2.94 (LongBeachVA) | major |
| **Honours-Ensemble** | 1.24 | 1.72 (Switzerland) | major |
| **XGBoost** | 0.44 | 0.68 (Switzerland) | major (only) |
| **L1 LR** | 0.40 | 0.58 (Switzerland) | major (only just) |

Even though every fold sees the same input drift, the four models translate that drift into wildly different `predict_proba` shifts. **TabICL and the Ensemble move ~3-4× harder than XGBoost and LR** under the same covariate shift.

Two interpretations, both honest:

1. **TabICL and the Ensemble are doing more aggressive feature interaction modelling**, so a marginal shift in `ExerciseAngina` (a known-flagged drifted feature on LongBeachVA) propagates into a larger predicted-probability shift. This is consistent with the Phase-2.4 finding that the Ensemble has the lowest cross-fold consistency on the held-out test sources.
2. **XGBoost's isotonic calibration and LR's Platt calibration both flatten the probability distribution into more bounded bands**, partly absorbing covariate shift inside the calibration mapping. This is a known effect — calibrated trees and linear models often look more drift-stable than they "really are" because the calibration step is doing some of the work.

Both interpretations mean the same thing operationally: **if you were going to deploy any of the four v1 models, you would want to monitor TabICL's and the Ensemble's prediction-drift PSI more aggressively than XGBoost's or LR's**. ADR-014's severity-band thresholds were not designed to be model-family-aware; a future productionisation phase would want to revisit them per model.

### 4.3 Reading the dashboard PNGs

Each of the 16 dashboard PNGs (`reports/v1/figures/drift/<model>_<source>_dashboard.png`) is a single-glance summary for one (model, fold) cell:

- **Top wide panel:** PSI bar across every feature, sorted descending, severity-coloured (green stable / amber moderate / red major). Vertical dashed lines mark the band boundaries (0.10, 0.25). The ECDF and prediction panels below this are coloured to be self-consistent with this top panel.
- **Bottom-left:** ECDF overlay (reference blue, current red) for the top-3 numeric drifted features. Categoricals are skipped here (KS / ECDF aren't defined on unordered levels).
- **Bottom-right:** `predict_proba` histogram overlay (reference blue, current red), with the prediction-drift PSI value and severity in the title. This is the single most decision-relevant chart in the dashboard.

The recommended starting place when reviewing the deliverable is `reports/v1/figures/drift/tabicl_LongBeachVA_dashboard.png` (the worst-case prediction-drift cell, PSI=2.94), followed by `xgboost_LongBeachVA_dashboard.png` (same input drift, ~5× lower prediction shift). Side-by-side they tell the headline story of section 4.2.

---

## 5. What this misses (honest caveats)

Six known weaknesses, each with the trade-off that produced it.

1. **Per-feature only.** PSI cannot detect shifts in the *joint* distribution. If the correlation between `Age` and `MaxHR` flipped while both marginals were stable, this entire phase would report `stable` across the board. Multivariate detectors (MMD, domain-classifier) are deferred until there's a deployment producing data that justifies their cost. Trigger to revisit: a productionisation phase, or a Phase-3 finding that one of the four v1 models is materially more drift-sensitive than the others on a single input feature (which would suggest a feature-interaction effect PSI cannot see).

2. **Bin-count sensitivity.** PSI on the same shifted-distribution pair generally grows with the bin count. The 10-quantile-bin choice is the industry convention (and the orchestrator constant `cardiorisk.monitoring.reference.DEFAULT_N_BINS`), not a derived optimum. A sensitivity sweep is intentionally not in this phase; the orchestrator's reference-build path takes the bin count as a parameter (`build_fold_reference(..., n_bins=N)`) so a future phase can re-run with `--n-bins 5 / 20 / 50` once the CLI exposes the flag. The headline numbers in section 4 are at `n_bins=10`.

3. **No time component.** PSI is a single point-in-time comparison. A production monitor would also report a rolling-window PSI series (e.g. weekly buckets over a 12-month deployment). The Phase-2.6 driver could be wrapped in a cron / Argo schedule that re-runs against new data and writes a JSON-per-day, but the visualisation layer would need extending. Out of scope for the research artefact.

4. **No concept drift.** Concept drift (`P(y | x)` shifts) requires labelled new data. The repo has none. ADR-014 explicitly defers this to a follow-up phase that has it — at which point the existing PSI severity bands would be augmented with a calibration-shift detector (something like reliability-diagram-bin-residual PSI between deployment windows).

5. **KS reconstruction approximation.** The KS sanity-check uses synthetic reference samples reconstructed from the persisted bin midpoints + counts. This is exact for a discretised feature and a faithful approximation for continuous data given the same quantile binning, but it is not the same as running KS on the original raw reference samples. The orchestrator's headline numbers are the PSI values; KS is reported alongside but should be read as an *order-of-magnitude* sanity check rather than a definitive p-value. The alternative — persisting full reference samples — multiplies the reference-artefact size on disk for marginal sanity-check gain. ADR-014 §"Honest weaknesses" documents this.

6. **Severity thresholds not validated for this dataset.** `< 0.10` stable / `0.10 – 0.25` moderate / `>= 0.25` major are industry convention, drawn from credit-risk and post-deployment-monitoring tutorials. Different domains (genomics, sensor data, NLP) use different thresholds. No effort has been made to derive dataset-specific bands from first principles; doing so would require either a long deployment history or a synthetic shift study, neither of which exists for this repo. The numbers in section 4 should be read as "PSI as the convention says it should be read", not "validated for this dataset".

---

## 6. What this enables for Phase 3

The Phase-2.6 outputs are inputs to two Phase-3 deliverables that haven't been built yet but were already on the radar in [`AGENTS.md`](../../AGENTS.md) §2.

- **The risk-driver narrative drafter.** When the LangGraph agentic system explains why a patient is high-risk, it cites the SHAP per-feature attributions from Phase 2.5. Phase 2.6 lets it also caveat that explanation with "note that `ST_Slope` distribution at this clinic differs substantially from the model's training data (PSI=7 on the Cleveland fold)". That kind of conditional caveat is the qualitative difference between a useful clinical co-pilot and a confident-sounding hallucinator.
- **The HITL approval surface.** Each of the four v1 models will be exposed in the UI; the prediction-drift PSI per (model, current-data-batch) is a direct input to which model the UI should prefer for which input. A clinic whose intake distribution looks Switzerland-like should preferentially see XGBoost / LR predictions (lower prediction-drift PSI there, ~0.6), with TabICL / Ensemble flagged as "high-drift, second opinion only".

Both of those land in Phase 3+ and are not implemented here. Phase 2.6 ships the metric surface they will plug into.

---

## 7. Reproduce

```bash
# One-time: rebuild the combined parquet + per-fold model artefacts
uv run --project backend python backend/scripts/build_combined.py
uv run --project backend python backend/scripts/train_v1.py

# The Phase-2.6 sweep (writes reports/v1/drift/{per_fold,aggregate}.json
# and 16 dashboard PNGs to reports/v1/figures/drift/):
uv run --project backend python backend/scripts/compute_drift.py

# Optional: persist per-fold references to disk (production-monitoring
# use case; orchestrator does not need this):
uv run --project backend python backend/scripts/build_reference.py
```

CI smoke (4 models, 1 LODO fold, smoke artefacts; ~10 s on ubuntu-latest):

```bash
uv run --project backend python backend/scripts/compute_drift.py --smoke
```

---

## 8. Cross-references

- [ADR-014](../adr/014-drift-monitoring.md) — binding decision for the Phase-2.6 surface.
- [05 — EDA findings](./05-eda-findings.md) — informal cross-source distribution differences flagged at Phase 2.1; Phase 2.6 puts numbers on them.
- [06 — Preprocessing decisions](./06-preprocessing-decisions.md) — explains why `Cholesterol == 0` is recoded to NaN, which interacts with the drift numbers (the `Cholesterol_was_missing` indicator is part of the cleaned schema and itself drifts).
- [08 — v1 model results](./08-v1-model-results.md) — the cross-source AUROC drop the drift report explains at the distribution level.
- [10 — Explainability](./10-explainability.md) — Phase-2.5 SHAP numbers; the per-feature drift here annotates which of those SHAP features are *also* the most distorted under cross-source shift.
- [`MODEL_CARD.md`](../../MODEL_CARD.md) §6 "Drift monitoring" — the headline drift table for the public face.
