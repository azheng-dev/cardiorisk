# 04 — Revised v1 design for the CardioRisk Co-Pilot risk model

> **Purpose.** The proposed v1 ML system for CardioRisk Co-Pilot, justified line-by-line against the verdicts in [`03-critical-review.md`](./03-critical-review.md) and the modern state of the art summarised in [`02-current-soa.md`](./02-current-soa.md).
>
> **Scope.** This document covers the *risk-model layer only* — what gets trained, on what data, evaluated how, calibrated how, explained how, and positioned how. The agentic layer (LangGraph triage → risk → guideline → letter) and the UI live in Phases 4 and 5 respectively and are out of scope here.
>
> **Status.** Proposal. The binding decision is in [ADR-006](../adr/006-risk-model-architecture.md). Phase 2 implements; Phase 2.4 re-runs the prior WOA-Ensemble as a baseline against this design.

---

## 1. Design principles (in order of priority)

1. **Calibration over accuracy.** A model whose probabilities don't match observed event rates is useless to a clinician, regardless of how high its AUROC is. Brier and reliability come before accuracy in the eval card.
2. **Out-of-distribution generalisation over in-distribution headlines.** LODO-CV is the headline protocol; random K-fold is reported only as a sanity check.
3. **Honesty over impressiveness.** If the rebuilt model lands below the prior 89.7% sensitivity headline under leakage-controlled CV, that is the result we publish. ([AGENTS.md §3](../../AGENTS.md)).
4. **Reproducibility over novelty.** Every choice traceable to a primary citation; deterministic where possible; pinned random seeds; pinned dependencies.
5. **Clinical positioning explicitly subservient to AusCVDRisk.** Not a replacement. UI says so; Model Card says so; README says so.
6. **TRIPOD+AI compliance.** All 27 items addressed by the time the eval card is published.

---

## 2. Model architecture

### 2.1 Headline (primary) model — TabPFN v2.5 / v2.6

**Why.** At n < 10,000 rows on a tabular clinical problem, TabPFN matches or beats hyperparameter-tuned XGBoost zero-shot (no per-dataset training), and ships with calibrated Bayesian-style posterior probabilities by construction. It also natively handles missing values, which addresses the Switzerland-cholesterol issue (see [03 §2](./03-critical-review.md#2-choice--imputation-missforest)). Source: [02 §1.1](./02-current-soa.md#11-tabular-foundation-models--the-new-default-for-small-clinical-data).

**How.**

- Use the official `tabpfn` Python package, latest stable v2.x at Phase-2.3 time.
- Inputs: the 11 HFP features as published. Categorical features one-hot encoded only if TabPFN's tokenizer needs it; otherwise pass-through.
- Output: calibrated probability of `HeartDisease == 1`.
- No metaheuristic optimisation, no per-dataset hyperparameter search.
- Random seed pinned for reproducibility of the in-context demonstration set.

**Why not the other TFMs.**

- **TabICL** — useful when feature count is high; HFP has 11 features, so the column-then-row attention story doesn't pay off.
- **TabuLa-8B** — interesting given we already have an LLM in the loop, but adds substantial inference cost and the headline benefit (LLM world-knowledge over column names) is small for a fixed feature set with documented semantics.

### 2.2 White-box baseline — calibrated XGBoost

**Why.** Defensible, transparent, fast, well-understood by clinical reviewers. It's the baseline a senior reviewer will ask for first. If TabPFN doesn't beat XGBoost+isotonic by a meaningful margin under LODO-CV, we say so plainly.

**How.**

- `xgboost>=2.x`.
- Hyperparameter tuning with [Optuna](https://optuna.org/) under Bayesian optimisation, strict early stopping on a held-out within-fold validation slice.
- Tuning ranges: standard for clinical XGBoost (max_depth 3–8, learning_rate 0.01–0.3, n_estimators capped at 2,000 with early-stopping rounds 50, subsample / colsample_bytree 0.5–1.0, reg_alpha and reg_lambda log-uniform 1e-3 to 10).
- Post-hoc **isotonic regression** calibration on a separate calibration slice within each LODO fold.

### 2.3 Simplest baseline — L1 logistic regression with spline-expanded continuous features

**Why.** Transparency anchor. Coefficients are directly interpretable; the model is publishable as a Cox-table-equivalent if needed. If the headline model can't beat L1-LR by a clinically meaningful margin, the deep learning was theatre.

**How.**

- `scikit-learn` `LogisticRegressionCV` with L1 penalty, `liblinear` or `saga` solver.
- Continuous features expanded with restricted cubic splines (3–5 knots at quantile positions) — captures non-linearity without exploding the parameter count.
- Categorical features one-hot.
- Standard 5-fold CV for the C-tuning, nested inside the LODO loop.
- No post-hoc calibration needed — logistic regression is by construction probabilistic.

### 2.4 Honesty baseline — WOA-Ensemble (prior Honours architecture)

**Why.** Phase 2.4 in [AGENTS.md §7](../../AGENTS.md) requires running the prior architecture as a baseline alongside v1. Done under the *same* leakage-controlled LODO-CV protocol, the comparison is honest. Either:

- WOA-Ensemble matches or beats the modern stack — surprising and worth documenting.
- WOA-Ensemble underperforms — expected given §3 verdict, and worth documenting publicly so the rebuild's verdict is empirically grounded, not just theoretical.

**How.**

- Reimplement WOA-Ensemble (CNN + LSTM + ANN with WOA-tuned hyperparameters) against the same input pipeline.
- Reuse the prior study's hyperparameter ranges where documented; document where they're not.
- Same LODO-CV, same eval card, same calibration (post-hoc isotonic).

This is the *only* purpose for which deep-tabular code lands in this repo. It's a baseline, not the headline.

---

## 3. Data pipeline

### 3.1 Source

- Heart Failure Prediction dataset (Kaggle, fedesoriano).
- Pulled at build time by `backend/scripts/fetch_hfp.py` from the Kaggle URL into `data/raw/`.
- Never committed to the repo (gitignored from Phase 0).
- SHA-256 hash pinned in `data/raw/HFP.sha256` so we detect any silent upstream change.

### 3.2 Cleaning

- **Zero-cholesterol records → NaN** before imputation. The prior pipeline silently treated `Cholesterol == 0` as a real measurement; modern audits flag this as a data-entry artefact for the Switzerland subset.
- **Source-of-record column** (`source ∈ {Cleveland, Hungarian, Switzerland, LongBeachVA, Stalog}`) reconstructed from the published dataset's row ordering and known sample sizes (or — preferred — pulled from the original UCI files separately). This column is *not* a model feature; it's the LODO-CV grouping variable.
- **Missingness mask** appended as additional binary features (`is_<col>_missing`) so the model can learn that "missing X on a record from source Y" is informative.

### 3.3 Imputation

- **Default for TabPFN:** native (no preprocessing imputation; pass NaNs through).
- **For XGBoost / WOA-Ensemble baselines:** MissForest (`missingpy` or `IterativeImputer` with `RandomForestRegressor` / `RandomForestClassifier`), fitted *within each LODO fold's training slice only* — never on the test fold. Imputation leakage is real and was not controlled in the prior study.
- **For LR baseline:** mean / mode imputation as a published worst-case; report sensitivity to this choice in an ablation appendix.

### 3.4 Encoding and scaling

- Categorical: one-hot.
- Continuous: TabPFN passes through; XGBoost passes through; LR uses standard-scaled spline expansions.

### 3.5 Train / val / test split

- **Headline protocol: Leave-One-Domain-Out CV (LODO-CV).** Train on 4 sources, test on the 5th, rotate. Headline metric is the mean ± std across the 5 folds, plus per-source numbers.
- **Within each LODO fold:** further 80/10/10 train/val/calibration split for hyperparameter tuning + post-hoc calibration.
- **Sanity check: stratified 5×5 random K-fold** on the union, reported separately. Only used to demonstrate that random K-fold inflates numbers vs LODO; *not* the headline.
- **Random seeds pinned** (`SEED = 20260505`); reproducible across reruns.

---

## 4. Feature selection

**Decision: no feature selection in v1.** With 11 features and ~700 train rows per LODO fold, the marginal value of FS is small and the failure modes are large (FS done outside the CV is leakage; FS done inside the CV is fine but rarely changes a 11-feature problem).

**Documented in the Model Card.** Use SHAP for *interpretation*, not selection.

If a future v2 wants FS — for example to demonstrate methodology on a higher-dimensional clinical dataset — the recommended methods are **BorutaShap** (statistically grounded, deterministic) or **mRMR** (information-theoretic, compact). The metaheuristic stack (WOA, GWO, HHO, FA, CS, BA) is explicitly *not* on the table; see [03 §3](./03-critical-review.md#3-choice--feature-selection-10-metaheuristic-algorithms-woa--gwo--hho--fa--cs--ba--ga--rf--rfe--rfe-cv).

---

## 5. Evaluation and reporting

### 5.1 Headline metrics (per LODO fold + aggregate)

| Metric | Why |
|---|---|
| AUROC | Discrimination |
| AUPRC | Discrimination at operating point under class imbalance |
| Brier score | Calibration + discrimination, strictly proper scoring rule |
| Calibration slope + intercept | Numeric calibration |
| Reliability diagram | Visual calibration |
| Sensitivity at 85% specificity | Clinical operating-point performance |
| Sensitivity at 90% specificity | Stricter-spec operating point |
| Net benefit (DCA) at p_t = 5% and 10% | Clinical utility at AusCVDRisk thresholds |

All headline metrics reported with **bootstrapped 95% CIs** (2,000 resamples).

### 5.2 Subgroup performance

Per [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378):

- AUROC, Brier, sensitivity-at-85%-spec, calibration slope **stratified by**:
  - Sex (Male / Female)
  - Age band (<50, 50–69, ≥70)
- **Fairness gap** (max - min within each grouping) explicitly stated. Any gap > 5 percentage points on sensitivity gets a paragraph in the Model Card explaining the mechanism and any mitigation tried.

### 5.3 Per-source breakdown

Same metric set computed per LODO fold *individually* (Cleveland-as-test, Hungarian-as-test, Switzerland-as-test, LongBeachVA-as-test, Stalog-as-test). Documents which sources the model generalises to and which it doesn't.

### 5.4 The honesty baseline comparison

Phase 2.4 produces a **single comparison table**:

| Model | LODO mean AUROC ± std | LODO mean Brier ± std | LODO mean Sens@85% spec ± std | LODO mean DCA Net Benefit @ p_t=10% ± std |
|---|---|---|---|---|
| TabPFN v2.6 | _filled in Phase 2.4_ | | | |
| XGBoost + isotonic | | | | |
| L1 LR (spline) | | | | |
| WOA-Ensemble (prior) | | | | |

Plus a paragraph for each where the result diverges from expectation.

### 5.5 Reporting standard

A `MODEL_CARD.md` is published alongside the model, structured per [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993). It includes:

- Intended use / intended user / out-of-scope use
- Training-data composition + per-source summary stats
- Aggregate + subgroup performance tables (the §5.1, §5.2, §5.3 outputs)
- Known failure modes (per [03 §1, §4, §5](./03-critical-review.md))
- Ethical considerations (race not in HFP; AusCVDRisk includes First Nations identity and SEIFA which we cannot)
- TRIPOD+AI checklist mapping (a table linking each of the 27 items to the line in the Model Card / the eval / the README that satisfies it)

---

## 6. Clinical positioning vs. the Australian CVD Risk Calculator

The artefact is positioned as an **educational second-opinion / counterfactual-explorer** subordinate to the official [Aus CVD Risk Calculator](https://www.cvdcheck.org.au) (PREDICT-1° recalibrated to Australian mortality, endorsed by RACGP, embedded in the [Red Book 10th Edition](https://www.racgp.org.au/clinical-resources/clinical-guidelines/key-racgp-guidelines/view-all-racgp-guidelines/preventive-activities-in-general-practice/what-s-new-in-the-10th-ed-red-book) 2025/2026).

**UI commitments** (Phase 5):

- The patient input form prompts for AusCVDRisk inputs first; the AusCVDRisk score is computed (or a link to the official calculator is rendered) before the ML output is shown.
- The ML output is rendered in a panel labelled clearly as "Second-opinion / counterfactual exploration", not "Risk score".
- A non-dismissible disclaimer banner: *"Synthetic data only. Not for clinical use. Defer to the Australian CVD Risk Calculator and RACGP guidelines."*
- The DiCE counterfactual UI is framed as a *motivational-interviewing aid*, not a treatment recommendation.

**Model-Card commitments**:

- Explicit statement that HFP is not Australian-recalibrated and lacks SEIFA / uACR / eGFR / severe-mental-illness / First-Nations-identity inputs.
- Explicit pointer to AusCVDRisk as the clinically authoritative tool.

This positioning is what makes the artefact defensible to a senior reviewer in 2026. Anything stronger ("CVD risk model for Australian primary care") would be a clinical claim the data cannot support.

---

## 7. Explainability

### 7.1 SHAP

- **TreeSHAP** for the XGBoost baseline ([Lundberg et al. 2020](https://www.nature.com/articles/s42256-019-0138-9)). Polynomial-time exact computation; suitable for real-time UI.
- **KernelSHAP** for TabPFN. Higher computational cost; cached per-prediction; documented as an approximation.
- **L1 LR coefficients** shown directly for the LR baseline.

### 7.2 Counterfactuals

- **DiCE** ([Mothilal et al. 2020](https://arxiv.org/abs/1905.07697)) integrated for the headline model. Generates 3 minimal-perturbation lifestyle scenarios per prediction (e.g., "if SBP drops 10 mmHg and the patient stops smoking, the predicted risk band shifts from High to Intermediate").
- Constraints applied so counterfactuals only perturb *modifiable* features (BP, cholesterol, smoking status proxies). Age and sex are pinned.

### 7.3 Natural-language summariser

- A temperature-zero LLM call (Claude Sonnet 4.5 by default; see Phase 6 multi-model evaluation in [AGENTS.md §7](../../AGENTS.md)) takes (raw features, prediction probability, SHAP vector, top-K SHAP-magnitude features, AusCVDRisk band if available) and produces a short narrative (~80–120 words).
- Strict prompt with grounding instructions; the LLM acts as a *translation layer*, not a reasoner.
- Output is verified by NLI (Phase 3.3) against the SHAP vector; unsupported claims are dropped, not "rephrased".
- Versioned prompt template lives in `backend/prompts/risk_explanation.j2`.

---

## 8. Reproducibility

- **Pinned dependencies** via `uv.lock`.
- **Pinned data**: `data/raw/HFP.sha256` checked at every load.
- **Pinned seed**: `SEED = 20260505` everywhere stochasticity is unavoidable.
- **Per-run artefacts** (LODO fold metrics, SHAP arrays, calibration plots) versioned in `eval/runs/<UTC-timestamp>/`, not committed to git but mirrored to a model registry (Hugging Face Hub or W&B Artifacts; decision deferred to Phase 2.3).
- **Notebooks** (`notebooks/01-eda.ipynb`, `notebooks/02-modelling.ipynb`, `notebooks/03-comparison.ipynb`) committed cleared of outputs, with `nbstripout` enforcing this in pre-commit.

---

## 9. Honest weaknesses of v1

If the v1 design is the proposal that ships, here is what I'd weakly defend against:

- **TabPFN is opaque even with KernelSHAP.** It's a transformer doing in-context Bayesian inference; the explainability story is real but not as crisp as TreeSHAP-on-XGBoost. Mitigation: report all three models; let readers compare.
- **No external validation cohort.** The artefact is honest about this, but a properly external Australian cohort (e.g., a synthetic AusCVDRisk-derived test set) would be stronger. Listed in [AGENTS.md §8](../../AGENTS.md) future scope.
- **HFP itself has well-documented pathologies** that no methodology fix can fully overcome. The honest answer is "this is an educational artefact, the methodology is the deliverable, the clinical claim is bounded."
- **No causal inference.** DiCE counterfactuals look causal but are observational — they're "what would the model say if we changed X" not "what would actually happen if we changed X". The Model Card and the UI both label this clearly.
- **The PREDICT-1° equation is not reimplemented as a peer baseline.** Doing so cleanly would require pulling AusCVDRisk's open documentation and validating against a synthetic cohort. Listed for v2.
- **Subgroup audit is restricted to sex × age** because HFP doesn't carry ethnicity. We document this; we do not pretend otherwise.

---

## 10. Out-of-scope decisions (deferred)

- LangGraph orchestration design (Phase 4).
- LLM choice + multi-model evaluation (Phase 6, ADR-009).
- Embedding / retrieval architecture for guideline RAG (Phase 3, ADR-007).
- Citation + NLI verification approach (Phase 3, ADR-008).
- UI brand + visual identity (Phase 5, ADR-010).
- Deployment topology + observability (Phase 7+, ADR-TBD).

These are listed so a reader can see the full v1 boundary.

---

## What this commits us to

- A v1 risk model with TabPFN as headline, XGBoost as white-box baseline, L1 LR as transparency anchor, WOA-Ensemble as honesty baseline.
- LODO-CV as the headline evaluation protocol, with Brier / DCA / calibration slope / subgroup-stratified reporting and bootstrapped 95% CIs.
- A TRIPOD+AI-aligned Model Card published alongside the model.
- Explicit clinical positioning as a second-opinion / counterfactual-explorer subordinate to the Australian CVD Risk Calculator.
- TreeSHAP + DiCE + LLM nat-lang summariser for the explainability layer.
- No metaheuristic feature selection or hyperparameter tuning anywhere in v1.

The binding form of these commitments is in [ADR-006 — Risk-model architecture](../adr/006-risk-model-architecture.md).
