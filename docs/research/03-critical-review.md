# 03 — Critical review of the prior Honours work, against 2025–2026 evidence

> **What this is.** An opinionated, head-to-head review of every non-trivial methodology choice in the prior Honours work ([01-honours-recap.md](./01-honours-recap.md)) against the modern state of the art ([02-current-soa.md](./02-current-soa.md)).
>
> **Honesty contract** ([AGENTS.md §3](../../AGENTS.md)). Where the prior work is defensible in 2026, that's what this document says. Where it's not, that's also what this document says — even if uncomfortable. The senior-engineering signal is the audit, not the headline number.
>
> **Format.** For each Honours design choice: (a) what's still defensible, (b) what's outdated or unsupported, (c) what to upgrade and why, (d) the strongest evidence for the upgrade. Closes with explicit verdicts on the seven AGENTS.md §7 questions and a list of what to preserve verbatim into v1.

---

## 1. Choice — Dataset: Heart Failure Prediction (Kaggle, fedesoriano)

### (a) Defensible
- **Public, freely available.** Sound choice for an open-source portfolio piece. No PHI risk. Aligns with [AGENTS.md §6](../../AGENTS.md) "synthetic / public data only".
- **Manageable size and dimensionality** — small enough to iterate on a laptop, low enough dimensionality that feature engineering is tractable.
- **Clinically interpretable feature set** — the 11 features (age, sex, chest pain type, BP, cholesterol, FBS, RECG, max HR, exercise angina, oldpeak, ST slope) are real, named clinical variables.

### (b) Not defensible in 2026
- **Severe distribution shift across the five constituent sources** is treated as i.i.d. by the prior study, despite published evidence of large demographic and measurement-protocol differences (e.g., Long Beach VA ~all male; Switzerland zero-imputed cholesterol). Source: [02 §4.2](./02-current-soa.md#42-distribution-shift).
- **272-row deduplication suggests near-duplicate leakage.** The mathematical sum of the five sources is 1,190; Kaggle ships 918. The Stalog dataset is widely understood to be a re-curated subset of Cleveland + Hungarian. Random K-fold CV on this concatenation has a non-trivial probability of placing near-duplicates on both sides of the split, inflating reported metrics. Source: [02 §4.3](./02-current-soa.md#43-duplicate-row-leakage-risk).
- **No relevance to the Australian context.** HFP is missing the variables the 2023 Australian Guideline considers essential (SEIFA, uACR, eGFR, severe mental illness, First Nations identity); it cannot underpin a clinically meaningful AU CVD risk tool. Source: [02 §7](./02-current-soa.md#7-the-australian-context-the-bit-the-prior-honours-work-largely-missed).
- **No external validation cohort** — the prior study reports only on a single 80/20 split of HFP itself.

### (c) Upgrade
- **Keep HFP as the primary dataset** for the rebuild. It's the dataset the prior work used; comparability matters; honest re-evaluation under modern protocol is the point. *Don't* swap it out and then claim the new architecture is better — that conflates two changes.
- **Switch to Leave-One-Domain-Out CV (LODO-CV)** in place of randomised K-fold. Train on four sources, test on the fifth, rotate. Measures out-of-distribution generalisation directly and breaks the Stalog ↔ Cleveland/Hungarian leakage path.
- **Report per-source metrics**, not just an aggregate.
- **Reframe the artefact** as an educational second-opinion to AusCVDRisk, not a primary clinical tool. UI must say so. See [04 §6](./04-revised-design.md#6-clinical-positioning-vs-the-australian-cvd-risk-calculator).
- **Document the missingness pattern explicitly** (per [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)). The Switzerland cholesterol-zero issue is the single most important data-quality footnote the prior study did not surface.

### (d) Evidence
- [Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815) on the importance of distribution-shift evaluation.
- [TRIPOD+AI 2024 (BMJ)](https://www.bmj.com/content/385/bmj-2023-078378) on per-cohort and subgroup reporting.
- [2023 AU CVD Guideline](https://pubmed.ncbi.nlm.nih.gov/38623719/) on what variables a credible AU CVD risk model must include.

---

## 2. Choice — Imputation: MissForest

### (a) Defensible
- **MissForest** ([Stekhoven & Bühlmann 2012](https://doi.org/10.1093/bioinformatics/btr597)) remains a reasonable iterative-RF imputer. The prior study's citation ([Waljee et al. 2013](https://doi.org/10.1136/bmjopen-2013-002847), MissForest > mean / kNN / MICE on lab data) is genuine and still cited in 2026 reviews.
- For a small clinical tabular dataset, MissForest produces plausible imputed values without dramatic computational overhead.

### (b) Not defensible
- **Single-imputation only.** MissForest gives one point estimate per missing cell; uncertainty about the imputed value is discarded. For small clinical data this is non-trivial.
- **Treats the Switzerland cholesterol-zero issue as a missing-value problem.** Many of those values are not "missing": they were measured as zero or recorded as 0 by data-entry convention. Imputing a plausible cholesterol where the real semantic is "not measured" or "recorded as zero by convention" can fabricate signal.
- **Doesn't interact with hospital-of-origin.** An imputer that ignores the source can leak that information back in via systematically different imputed values per source.

### (c) Upgrade
- **Keep MissForest as one option**, especially as a baseline.
- **Default to TabPFN's native missing-value handling** for the headline model. TabPFN is pre-trained on synthetic data with missingness as a first-class feature. Source: [02 §1.1](./02-current-soa.md#11-tabular-foundation-models--the_new_default_for_small_clinical_data).
- **Preserve the missingness mask** as an additional explicit feature (`is_<col>_missing`). Lets the model learn that "missing cholesterol on a Swiss record" carries information.
- **Treat zero-cholesterol records as `NaN` upstream of imputation**, not as observed zeros. Document this preprocessing choice in the model card and a notebook cell.
- **Multiple imputation (MICE / mice or `IterativeImputer`) for the GBDT baseline**, with downstream uncertainty propagation, where time permits. ([Still More Shades of Null, 2024 evaluation suite](https://www.researchgate.net/publication/395261243_Still_More_Shades_of_Null_An_Evaluation_Suite_for_Responsible_Missing_Value_Imputation) [unverified DOI but topic is real and well-cited]).

### (d) Evidence
- [Waljee et al. 2013](https://doi.org/10.1136/bmjopen-2013-002847) (the prior study's own citation, still defensible).
- TabPFN's native handling of missing values, [Hollmann et al. 2023](https://arxiv.org/abs/2207.01848).
- [`MICE`](https://stefvanbuuren.name/fimd/) as the standard for multiple imputation (van Buuren).

---

## 3. Choice — Feature selection: 10 metaheuristic algorithms (WOA / GWO / HHO / FA / CS / BA + GA + RF + RFE + RFE-CV)

### (a) Defensible
- **The exhaustive comparison was thorough and pedagogically valuable** for an Honours project. Implementing 10 FS methods × 2 architectures × 4 datasets is real work.
- **GA, RF, and RFE / RFE-CV** are statistically grounded and remain credible methods, particularly when paired with cross-validated stability assessment.

### (b) Not defensible in 2026
- **The "nature-inspired" stack — WOA, GWO, HHO, FA, CS, BA — is widely regarded as scientifically indefensible.** Multiple peer-reviewed audits show:
  - Stripped of zoological terminology, they are functionally equivalent to PSO or basic (μ + λ) Evolution Strategies. Sources: [Aranha et al. 2022](https://www.researchgate.net/publication/362280739) ("Six misleading optimization techniques inspired by bestial metaphors"); [Campelo et al. 2025](https://www.researchgate.net/publication/401939020) ("Beyond metaphors"); [Sörensen 2025 retrospective](https://www.mdpi.com/2227-7390/13/13/2158); [EC Bestiary](https://github.com/fcampelo/EC-bestiary).
  - Many possess a centre-seeking bias on classic benchmark functions (Sphere / Rastrigin), which makes them appear strong on contrived benchmarks while failing on real, asymmetric problems.
  - No theoretical convergence guarantees; high stochasticity; reproducibility hard.
- **Any CV-coupled metaheuristic loop overfits to the validation set.** Iterating WOA across val-set scores creates *methodological leakage* that AUROC alone won't catch.
- **For 11 features, doing FS at all is questionable.** With 11 features and ~700 train rows, the marginal value of FS is small; the failure modes are much larger.

### (c) Upgrade
- **Drop the entire metaheuristic FS stack.** Don't reimplement WOA / GWO / HHO / FA / CS / BA in v1.
- **If FS is wanted at all**, default to **BorutaShap** (statistically grounded, deterministic, robust to multicollinearity) or **mRMR** (information-theoretic, compact). Source: [02 §2.2](./02-current-soa.md#22-what-replaces-them-in-2026).
- **For 11 features, the genuine recommendation is "no FS" — keep all features**. Use SHAP for *interpretation*, not selection. Document this choice and the reasoning.
- **Re-run the WOA-Ensemble headline configuration** alongside v1 in Phase 2.4 (per [AGENTS.md §7 Phase 2.4](../../AGENTS.md)) so we have an honest, leakage-controlled comparison rather than a hand-wave.

### (d) Evidence
- [Aranha, Campelo, et al. 2022 — "Six misleading optimization techniques inspired by bestial metaphors"](https://www.researchgate.net/publication/362280739).
- [Campelo et al. 2025 — "Beyond metaphors"](https://www.researchgate.net/publication/401939020).
- [Sörensen retrospective — "Rethinking Metaheuristics: Unveiling the Myth of 'Novelty'"](https://www.mdpi.com/2227-7390/13/13/2158).
- [BorutaShap comparative study, ResearchGate 2024](https://www.researchgate.net/publication/379152513).
- [Boruta — Kursa & Rudnicki 2010](https://doi.org/10.18637/jss.v036.i11).

---

## 4. Choice — Model architecture: WOCLSA (CNN + LSTM + ANN, WOA-tuned) and Ensemble (DNN + CNN + RNN + BiRNN)

### (a) Defensible
- **The implementation effort is real.** Building a four-net ensemble in TensorFlow / Keras and integrating a metaheuristic optimiser end-to-end is substantial engineering. The Honours team learned a lot.
- **The architectures are faithful to their cited sources** ([Su et al. 2023](https://doi.org/10.1038/s41598-023-39408-8) for WOCLSA; [Midhun et al. 2023](https://doi.org/10.1109/ICECAA58104.2023.10212248) for the four-net ensemble). The choices were defensible *at the time of selection* given the literature being cited.

### (b) Not defensible in 2026
- **Theoretical misalignment with tabular data.**
  - A 1D CNN assumes spatial/translation locality across adjacent feature columns. For 11 named clinical variables, the column order in the CSV is arbitrary. CNN filters that "see" `Age` next to `Sex` next to `Cholesterol` are extracting non-existent spatial structure.
  - LSTM / BiRNN assume a sequence with temporal dependencies between time-steps. For a static patient vector, the columns are not a sequence; the recurrent gating is modelling a non-existent time axis.
  - These observations are not fringe critiques; they're the consensus position from [Borisov et al. 2022](https://arxiv.org/abs/2110.01889) and [Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815) onward.
- **Severe over-parameterisation at n = 918.** The four-net ensemble's parameter count exceeds the training set by orders of magnitude. The most likely explanation for the published HFP numbers (~89.7% sensitivity *and* ~89.6% specificity) is memorisation, not genuine pattern recognition. Source: [02 §4.4](./02-current-soa.md#44-realistic-performance-ceiling-on-hfp-20222026-literature).
- **No reported calibration.** Ensembles of this kind tend to be over-confident; without a reliability diagram or Brier score the probability outputs are unverified for clinical use.

### (c) Upgrade
- **Reject the WOCLSA + DNN+CNN+RNN+BiRNN architectures for v1.** Don't carry them forward as the primary model.
- **Retain WOA-Ensemble as a published baseline** in Phase 2.4, evaluated under the *same* leakage-controlled LODO-CV protocol as v1, so there is an honest empirical comparison and the prior work is not silently dismissed. This is also a signal-of-craftsmanship for the public repo: "I rebuilt my own prior work as a baseline and reported the result honestly."
- **v1 primary architecture: TabPFN v2.5 / v2.6** (zero-shot, in-context, calibrated by construction) — see [04 §2](./04-revised-design.md#2-model-architecture).
- **v1 white-box baseline: XGBoost** with isotonic calibration and Bayesian-tuned hyperparameters (Optuna, strict early stopping). Source: [02 §1.2](./02-current-soa.md#12-gradient-boosted-decision-trees--the-durable-baseline).
- **v1 simplest baseline: Logistic regression with L1 + spline-expanded continuous features**, as a transparency anchor for clinicians and reviewers.

### (d) Evidence
- [Borisov et al. 2022, "Deep Neural Networks and Tabular Data: A Survey"](https://arxiv.org/abs/2110.01889).
- [Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning on typical tabular data?"](https://arxiv.org/abs/2207.08815).
- [TabPFN — Hollmann et al. 2023](https://arxiv.org/abs/2207.01848); TabPFN-2.5 follow-up [Nov 2025 arXiv](https://arxiv.org/abs/2511.08667) [partially verified].
- [Established-ML matches TFMs in clinical predictions, medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.02.02.26345274v1.full-text) [unverified DOI; treated as supporting, not anchoring].
- [Benchmarking transformer-based and conventional ML for CVD prediction, medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.08.03.25332878v1.full-text).

---

## 5. Choice — Train / test split: single 80 / 20 random split

### (a) Defensible
- **Simple, easy to communicate**, common in undergraduate ML pedagogy.
- For very large datasets, a single hold-out can be acceptable.

### (b) Not defensible
- **At n = 918, a single 80 / 20 split produces a 184-row test set.** Sensitivity / specificity reported on 184 patients have wide confidence intervals — the prior study reports neither a CI nor multiple-seed variance.
- **Random split ignores the five-source structure of HFP** and is the *most leakage-prone* choice possible given the duplicate-row issue.
- **No nested CV** to control for hyperparameter / metaheuristic search overfitting.

### (c) Upgrade
- **Primary protocol: Leave-One-Domain-Out CV (LODO-CV).** Train on 4 sources, test on the 5th, rotate. Headline metric is the mean ± std across the 5 folds.
- **Secondary protocol: stratified, repeated 5×5 K-fold within source-aware blocks.** For sanity-checking the within-source numbers and providing CIs.
- **Report bootstrapped 95% CIs** on every headline metric.
- **Nested CV (or a held-out tuning set)** for any hyperparameter search.

### (d) Evidence
- [Steyerberg 2019, *Clinical Prediction Models* (2nd ed.)](https://link.springer.com/book/10.1007/978-3-030-16399-0) — canonical reference for resampling strategies in clinical-prediction-model development.
- [TRIPOD+AI 2024 (BMJ)](https://www.bmj.com/content/385/bmj-2023-078378) — mandates external validation or, where unavailable, transparent CV with CIs.

---

## 6. Choice — Evaluation metrics: loss / accuracy / precision / sensitivity / specificity / F1 / ROC-AUC

### (a) Defensible
- **Sensitivity-first thinking** is appropriate for screening — false negatives in CVD screening are the costlier error class.
- **AUROC reporting** is uncontroversial and necessary.
- **F1** is a reasonable summary on imbalanced data.

### (b) Not defensible
- **No calibration metrics.** No Brier score, no reliability diagram, no calibration slope. The probability outputs are unverified.
- **No decision-curve analysis.** The model's *clinical utility* — does it improve net benefit at the operating threshold relative to "treat all" or "treat none"? — is not measured.
- **No subgroup stratification.** Performance is reported as an aggregate; per-sex and per-age-band performance are not. This is the failure mode TRIPOD+AI was designed to expose.
- **No confidence intervals or significance tests.**

### (c) Upgrade
Add to the v1 eval card:

- **Brier score** + **reliability diagram** + **calibration slope/intercept** (per [02 §3.2](./02-current-soa.md#32-what-is-now-mandatory-in-clinical-ml-reporting)).
- **Isotonic post-hoc calibration** of model outputs before reporting.
- **Decision-curve analysis** — net benefit across threshold probabilities 0–25%, with "treat all" / "treat none" baselines, evaluated at the AusCVDRisk-aligned thresholds (5%, 10%).
- **Subgroup performance** stratified by sex and age band, with explicit fairness-gap reporting in the Model Card.
- **Bootstrapped 95% CIs** on every headline metric.
- **Per-source breakdown** of HFP performance (Cleveland / Hungarian / Switzerland / Long Beach VA / Stalog).

### (d) Evidence
- [TRIPOD+AI 2024 (BMJ)](https://www.bmj.com/content/385/bmj-2023-078378).
- [Vickers & Elkin 2006 (DCA)](https://pubmed.ncbi.nlm.nih.gov/17099194/); [Vickers et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/).
- [Mitchell et al. 2019 — Model Cards](https://arxiv.org/abs/1810.03993).
- AHA 2024 statement on women & CVD; PREVENT 2023–2025 follow-ups (cited in [02 §6](./02-current-soa.md#6-fairness-and-subgroup-audits)).

---

## 7. Choice — Web platform: MERN stack (React + Express + MongoDB + TF.js)

### (a) Defensible
- **Pedagogically sound** for a web-development learning exercise.
- **TensorFlow.js** worked as a pragmatic way to ship inference without a Python backend.

### (b) Not defensible for the rebuild
- **Out of scope for this repo.** The MERN app does not live here and is not being migrated. Phase 5 designs a new UI from scratch ([AGENTS.md §7 Phase 5](../../AGENTS.md)).
- **MongoDB for clinical-shaped data** is the wrong default; relational + Postgres / Supabase is the modern choice.
- **No accessibility, no observability, no auth model worth carrying forward.**

### (c) Upgrade
- Discard the prior platform entirely (already done — never committed to this repo).
- Rebuild in Phase 5 using Next.js 15 + Tailwind v4 + shadcn/ui per AGENTS.md.

### (d) Evidence
- AGENTS.md decision in §4 + §7.
- ADR-003 ([TS tooling](../adr/003-typescript-tooling-pnpm-biome.md)) and ADR-004 ([monorepo layout](../adr/004-monorepo-layout-backend-frontend.md)) already commit to the new stack.

---

## 8. Verdicts on the seven [AGENTS.md §7](../../AGENTS.md) Phase 1 questions

### Q1. Is WOA-Ensemble (CNN + LSTM + ANN with WOA-tuned hyperparameters) still a defensible architecture in 2026 for a small (~918-row) tabular dataset, or should it be replaced?

**Replace.** The architecture is theoretically misaligned with static tabular data (CNN assumes spatial proximity, LSTM/BiRNN assume time-step sequencing — neither is true for a row of 11 named clinical variables) and is overparameterised by orders of magnitude at n = 918. Empirical evidence (Borisov 2022, Grinsztajn 2022, plus 2024–2026 follow-ups) is that tabular DL underperforms GBDT and TFM on small clinical data. The recommended v1 primary is **TabPFN v2.5/v2.6**; baselines are **calibrated XGBoost** and **L1 logistic regression**. WOA-Ensemble is reproduced in Phase 2.4 as an honest baseline under the same protocol, not preserved as the headline.

### Q2. Are the original feature-selection results (10 metaheuristic methods + RF / RFE) reproducible? Should any of them be dropped?

**Drop the entire nature-inspired metaheuristic stack** (WOA, GWO, HHO, FA, CS, BA). Multiple peer-reviewed audits (Aranha et al. 2022, Campelo et al. 2025, Sörensen 2025, EC Bestiary) demonstrate that these algorithms are functionally repackaged PSO/ES, suffer from centre-seeking bias on classic benchmarks, lack convergence guarantees, and are non-reproducible in practice. **Retain GA / RF / RFE / RFE-CV** as defensible classical options if FS is wanted. **Default for v1: BorutaShap or mRMR**, or simply no FS at all (11 features doesn't really need it).

### Q3. Were the original eval metrics (sensitivity, specificity, F1, AUROC) sufficient, or should the new build add calibration, decision-curve analysis, and fairness audits?

**Insufficient.** v1 must add: Brier score, reliability diagram, calibration slope, isotonic post-hoc calibration, Vickers' DCA at AusCVDRisk-aligned thresholds (5% and 10%), and per-sex × per-age-band subgroup performance with explicit fairness-gap reporting in a TRIPOD+AI-aligned Model Card. Bootstrapped 95% CIs on every headline metric.

### Q4. What does current literature say about the *real* upper bound on accuracy for the Heart Failure Prediction dataset? Are the original ~89.7% sensitivity numbers in line with, above, or below the published consensus?

**Above.** Modern, leakage-controlled HFP benchmarks (XGBoost, CatBoost, TabPFN) cluster around **AUROC 0.88–0.92** with **sensitivity 82–85% at ~85% specificity** ([02 §4.4](./02-current-soa.md#44-realistic-performance-ceiling-on-hfp-20222026-literature)). The Honours headline (~89.7% sensitivity *simultaneously* with ~89.6% specificity, single 80/20 split, n = 918) sits *above* the consensus ceiling. The likeliest explanations, in priority order: (1) memorisation by an overparameterised ensemble, (2) random-K-fold leakage through near-duplicate Stalog ↔ Cleveland/Hungarian rows, (3) validation-set tuning leak via the metaheuristic optimiser, (4) test-set lottery on a 184-row test set. Phase 2.4 will test this empirically by running the same WOA-Ensemble under LODO-CV.

### Q5. What are the *known* generalisation failures of models trained on HFP?

**Substantial.** Distribution shift across the 5 sources is severe (sex composition, measurement protocol, missingness pattern). Long Beach VA is almost all male; Switzerland frequently ships zero cholesterol values that are zero-imputed. Models trained on the union with random K-fold tend to learn *hospital-of-origin shortcuts* rather than genuine physiology. Out-of-distribution generalisation to a real Australian primary-care cohort is not measured by the prior study's protocol and is not implied by its reported numbers. Mitigation: LODO-CV + per-source reporting + explicit statement in the Model Card that this artefact is not externally validated.

### Q6. Where is the original Honours work *strongest*? What should be preserved verbatim?

The genuinely strong elements:

1. **Methodological scope of the comparison.** Implementing 10 FS methods × 2 architectures × 4 datasets is real engineering. The structure of the comparison framework can be reused (and will be, in Phase 2 — but with a smaller, more honest comparison).
2. **MissForest as an imputation baseline.** Defensible choice with a real underlying citation. v1 keeps it as a baseline option, with TabPFN's native handling as the default.
3. **Sensitivity-first framing for screening.** Correct intuition. v1 keeps it but reports calibrated probabilities and DCA so the operating-point choice is grounded.
4. **Use of HFP as the CVD dataset.** Sound for a public, reproducible artefact even with the documented pathologies — the alternative (proprietary EHR data) is not available, and the pathologies are now well-documented in the rebuild's docs and Model Card.
5. **Web-platform deployment intent.** Correct end-to-end framing — a research model that no clinician can interact with is half a research artefact. Phase 5 inherits the *intent*, not the *code*.

### Q7. Are there architectures the agent missed that should be considered?

Yes — flagged for the user to weigh in on at the Phase 1 checkpoint:

- **CatBoost** as a third GBDT baseline alongside XGBoost. Native categorical handling, default for many production CVD models in 2025–2026.
- **EBM (Explainable Boosting Machine, [InterpretML](https://github.com/interpretml/interpret))** — glass-box GBM with shape functions, native interpretability, often within 1–2 AUROC points of XGBoost on small clinical data.
- **DeepHit / DeepSurv / Cox PH with neural feature transforms** — only relevant if HFP were re-interpreted as a survival problem, which it can't be (no time-to-event in the dataset). Out of scope.
- **TabuLa-8B / LLM-as-tabular-classifier** — interesting given the rebuild already has an LLM in the loop for citation-grounded generation; could compare against TabPFN. Borderline overkill for Phase 2.

---

## 9. What v1 will preserve verbatim from the prior work

To make the "preserve" call concrete:

- **Dataset choice:** HFP (Kaggle, fedesoriano), as-published; *no* swap to a different dataset.
- **MissForest** as one of the imputation options (alongside TabPFN-native handling).
- **The names and types of the 11 features** — `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, target `HeartDisease` — exactly as in the prior pipeline. Ensures Phase 2.4 comparability.
- **Sensitivity-first operating-point thinking** for clinical framing.
- **Comparison-of-configurations spirit** of the original study, redirected toward TabPFN vs. XGBoost vs. L1-LR vs. WOA-Ensemble.

## 10. What v1 will explicitly drop

- WOA, GWO, HHO, FA, CS, BA — as feature selection methods *and* as hyperparameter optimisers.
- WOCLSA — CNN+LSTM+ANN sequence model on static tabular data.
- The DNN+CNN+RNN+BiRNN four-net ensemble — same reason.
- Single 80/20 random K-fold without source-awareness.
- Reporting only AUROC / sensitivity / specificity / F1 without calibration, DCA, or subgroup audit.
- Any reference to the prior MERN web platform code.
- The "best 4 of 4 UCI benchmark datasets" architecture-selection step. UCI Semeion / DARWIN / Malicious Executable / Arcene are not relevant to a clinical CVD pipeline; the architecture choice should be defended against tabular-clinical evidence directly, not against malware-detection benchmarks.

---

## 11. Summary verdict

The prior Honours work is **methodologically thorough for an undergraduate-thesis context** but is **not defensible as the architecture for a 2026 clinical-grade portfolio artefact**. The chosen model architectures (CNN+LSTM+ANN, DNN+CNN+RNN+BiRNN) are theoretically misaligned with static tabular data, the metaheuristic feature-selection stack is part of a literature now widely regarded as folklore, the evaluation protocol is leakage-prone and missing the calibration/DCA/fairness-audit pieces that any TRIPOD+AI-compliant clinical model must report in 2026, and the headline numbers are above the consensus performance ceiling for HFP under proper protocol — most likely as an artefact of memorisation, leakage, and validation-set tuning.

The honest path forward is to:

1. **Rebuild the model around TabPFN + calibrated XGBoost + L1 logistic regression** ([04 §2](./04-revised-design.md#2-model-architecture)).
2. **Re-run the prior WOA-Ensemble under the same leakage-controlled LODO-CV protocol** as a baseline (Phase 2.4), and report the result honestly — even if it drops materially below the published number.
3. **Adopt TRIPOD+AI + Model Card reporting** for v1 ([04 §5](./04-revised-design.md#5-evaluation-and-reporting)).
4. **Position the artefact as an educational second-opinion to AusCVDRisk**, not a replacement ([04 §6](./04-revised-design.md#6-clinical-positioning-vs-the-australian-cvd-risk-calculator)).
5. **Preserve the genuinely strong elements** of the prior work (HFP dataset, MissForest as an option, sensitivity-first framing, the comparison-of-configurations spirit) and document the choices openly.

Doing this is a stronger CV signal than preserving a 2024-vintage CNN+LSTM ensemble. A senior reviewer reading this repo will respect the honest re-evaluation more than they would respect a replicated 89.7% number.

The proposed v1 design is in [`04-revised-design.md`](./04-revised-design.md). The binding decision is in [ADR-006](../adr/006-risk-model-architecture.md).
