# 02 — Current state of the art for tabular CVD risk prediction (2025–2026)

> **Source.** Distilled and cross-checked summary of a user-supplied Deep Research synthesis, *"Critical Evaluation and Modernisation of Cardiovascular Disease Risk Prediction Architectures in Small-Cohort Clinical Datasets"* (2026). The report itself is an unpublished working artefact and is not redistributed in this repo. Its 99 cited works (papers, guidelines, benchmarks) are the underlying evidence base — citations below point to those primary sources, not to the synthesis document.
>
> **Cross-checking.** Every non-trivial claim below has been spot-checked against the cited primary source. Where a primary citation could not be independently verified at the URL provided (a few 2026 arXiv URLs look prospective rather than indexed), the claim is marked `[unverified]` and is treated as low-confidence in [`03-critical-review.md`](./03-critical-review.md).
>
> **Purpose.** Provide a faithful map of "what counts as state of the art in 2026 for an ~1,000-row tabular clinical prediction problem", so [`03-critical-review.md`](./03-critical-review.md) can defensibly judge the prior Honours work.

---

## 1. Tabular ML in 2026 — the four-class taxonomy

The tabular-ML field has consolidated into four genuinely distinct model classes, each with its own inductive bias and operating regime.

| Class | Representative architectures | Where it wins | Inductive-bias alignment to tabular data |
|---|---|---|---|
| **Tabular Foundation Models (TFMs)** | TabPFN v2.5 / v2.6, TabICL, TabuLa-8B | n < ~10,000 rows; single forward pass; in-context learning (ICL); Bayesian-style calibrated outputs | Excellent — pre-trained Bayesian priors over millions of synthetic data-generating processes |
| **Gradient-Boosted Decision Trees** | XGBoost, LightGBM, CatBoost | n > ~10,000 rows on heterogeneous numerical/categorical features; production deployment; cheap inference | Excellent — piecewise-constant step functions match irregular, non-smooth clinical thresholds |
| **Classical statistical models** | Logistic regression, Cox PH, GLMNet | When transparency / regulatory compliance / publication is the primary constraint | Good — strict linear/additive priors, robust under small-data noise; no native non-linearity |
| **Tabular deep learning** | TabNet, FT-Transformer, SAINT, NODE, TabR, RealMLP, ModernNCA | n > ~50,000 rows with extensive tuning; rarely the right choice on small clinical cohorts | Poor on small data — rotational invariance breaks axis-aligned tabular structure; smoothness bias misses sharp clinical thresholds |
| **(Anti-pattern)** | CNN / LSTM / BiRNN ensembles on static tabular data | Nowhere; theoretically misaligned for non-spatial, non-sequential features | Pathological |

Source: [Borisov et al. 2022](https://arxiv.org/abs/2110.01889), [Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815), TabArena 2026 [unverified], [Hollmann et al. (TabPFN) 2023](https://arxiv.org/abs/2207.01848), TabICL 2025 [unverified].

### 1.1 Tabular Foundation Models — the new default for small clinical data

- **TabPFN v2.5 / v2.6** (the headline shift, 2024–2026). Frames classification as Bayesian inference; pre-trained on millions of synthetic data-generating processes; performs *in-context learning* over the training set in a single forward pass — no per-dataset hyperparameter tuning. At n < 10,000 rows TabPFN matches or beats hyperparameter-tuned XGBoost zero-shot. Native handling of missing values is a side-benefit, particularly relevant to the Switzerland-cholesterol issue (§4) on HFP. Source: [Hollmann et al. 2023](https://arxiv.org/abs/2207.01848); [TabPFN-2.5 benchmark Nov 2025](https://arxiv.org/abs/2511.08667) [partially verified].
- **TabICL.** Column-then-row attention; scales to wider feature spaces and larger context windows than TabPFN before quadratic memory bottlenecks. Useful when the feature count climbs.
- **TabuLa-8B.** Fine-tunes Llama 3-8B over serialised tables; leverages the LLM's world knowledge; particularly useful when column headers carry semantic meaning ("LDL_cholesterol" not "feature_07").

### 1.2 Gradient-Boosted Decision Trees — the durable baseline

XGBoost / LightGBM / CatBoost remain the champion on n > 50,000 rows of mixed-type tabular data, per the 2025–2026 [TabArena benchmark](https://openreview.net/forum?id=jZqCqpCLdU) [unverified] and [established-ML-vs-TFM clinical benchmark on medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.02.02.26345274v1.full-text) [unverified URL]. They handle missing values natively, expose feature importances, train fast, and ship in any production environment. For an ~1,000-row clinical problem they're a *strong* baseline but not necessarily the headline model — TabPFN closes that gap.

### 1.3 The "deep learning for tabular data" debate, settled (mostly)

[Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815) ran an exhaustive 10,000-compute-hour benchmark across 45 datasets and isolated three reasons GBDTs beat tabular DL on medium-sized tabular data:

1. **Robustness to uninformative features.** Decision trees never split on noise (zero information gain). MLPs route every input through dense weight matrices, so noise propagates through the gradient and requires aggressive regularisation to suppress.
2. **Preservation of axis-alignment.** Tabular axes are not interchangeable — Age and Cholesterol mean different things. MLPs are rotationally invariant; trees are not. Removing axis-alignment via random rotation hurts trees but not MLPs, demonstrating that MLPs aren't exploiting the structure.
3. **Approximation of irregular target functions.** Clinical risk surfaces have sharp thresholds (e.g., a SBP of 140 mmHg). Trees approximate piecewise-constant step functions natively. Neural nets are smoothness-biased and need depth + data to fake step behaviour.

The 2024–2026 rebuttals (RealMLP, TabR, ModernNCA) demonstrated that with extensive tuning + sophisticated regularisation + n > 50,000 rows, tabular DL can reach parity with XGBoost. **None of them claim to win on small data**; the consensus is that on small clinical cohorts, tabular DL is at best parity, usually worse.

The arrival of TFMs in 2024–2025 reframed the debate: deep architectures *do* help on small data, but only when the depth has been spent at *pre-training* time over millions of datasets, not at *fine-tuning* time on the user's 918 rows.

## 2. Modern feature selection — statistical rigour vs. metaheuristic folklore

### 2.1 The discrediting of "nature-inspired" metaheuristics

A sustained line of critique (2014 → 2025) has demolished much of the metaphor-based metaheuristic literature. Headline points:

- **Functional equivalence to PSO / ES.** Mathematical decompositions of WOA, GWO, FA, BA, MFO, ALO show they are, stripped of zoological terminology, variants of Particle Swarm Optimisation or basic (μ + λ) Evolution Strategies. Source: [Campelo et al., "Beyond metaphors" 2025](https://www.researchgate.net/publication/401939020), [Aranha et al., "Six misleading optimization techniques inspired by bestial metaphors" 2022](https://www.researchgate.net/publication/362280739), [Evolutionary Computation Bestiary, Campelo](https://github.com/fcampelo/EC-bestiary), [Sörensen et al. 2018 / 2025 retrospective](https://www.mdpi.com/2227-7390/13/13/2158).
- **Centre-seeking bias on benchmark functions.** Many classic benchmarks (Sphere, Rastrigin, Ackley) place the global optimum at the origin. Several "novel" metaheuristics have a hardcoded bias toward the centre of the search space, which makes them appear successful on the benchmarks while failing on real, asymmetric problems.
- **Absence of convergence guarantees / reproducibility.** Stochastic, sensitive to hyperparameters, hard to reproduce across implementations.

The conclusion is direct: WOA, GWO, HHO, FA, CS, BA — the entire "nature-inspired" stack used in the prior Honours work — are unsuitable as primary feature-selection methods for a 2026 clinical pipeline.

### 2.2 What replaces them in 2026

| Method | Type | Why it wins |
|---|---|---|
| **Boruta** ([Kursa & Rudnicki 2010](https://doi.org/10.18637/jss.v036.i11)) | Wrapper, RF-based | "Shadow features" via shuffled copies; binomial test against shadow-importance distribution; deterministic; finds *all relevant* features, not a minimal subset |
| **BorutaShap** ([Keany 2020](https://medium.com/analytics-vidhya/is-this-the-best-feature-selection-algorithm-borutashap-8bc238aa1677); [comparative study, ResearchGate 2024](https://www.researchgate.net/publication/379152513)) | Boruta + SHAP | Replaces Gini-impurity importance with game-theoretic SHAP marginal contributions; robust to multicollinearity (BP × age) and high-cardinality bias |
| **mRMR** (Minimum Redundancy Maximum Relevance, [Peng et al. 2005](https://doi.org/10.1109/TPAMI.2005.159)) | Information-theoretic filter | Maximises mutual information with target while penalising mutual information with already-selected features; produces compact orthogonal subsets |
| **L1 / LASSO** ([Tibshirani 1996](https://www.jstor.org/stable/2346178)) | Embedded | Convex, mathematically guaranteed sparse solutions; trivial to publish and audit |
| **SHAP-based selection** | Embedded | Use SHAP|.| values directly as importance; cheap when the model is already a tree ensemble |

Practical 2026 default for a clinical tabular pipeline: **BorutaShap or mRMR**, depending on whether the analyst wants "all relevant" (Boruta family) or "minimum compact" (mRMR) feature sets. For a 918-row, 11-feature problem like HFP, the case for *any* feature selection at all is weak — the dimensionality is already trivial.

## 3. Calibration, decision-curve analysis, and TRIPOD+AI

### 3.1 Why AUROC alone is no longer enough

AUROC measures ranking — the probability that a randomly chosen positive scores higher than a randomly chosen negative. It is invariant to monotonic transformations of the score, which means a model with AUROC 0.95 may produce probabilities that are systematically wrong (e.g., always 0.99 on positives and 0.95 on negatives). For a clinical risk model whose outputs feed into the AusCVDRisk-aligned 5/10% threshold logic (§4), miscalibrated probabilities are directly clinically dangerous.

### 3.2 What is now mandatory in clinical-ML reporting

| Diagnostic | What it measures | Reference |
|---|---|---|
| **Brier score** | Mean squared error of probability forecasts; strictly proper scoring rule; combines discrimination + calibration | [Steyerberg et al. 2010](https://doi.org/10.1097/EDE.0b013e3181c30fb2); [Weighted Brier 2024](https://arxiv.org/html/2408.01626v1) |
| **Reliability diagram** | Visual: predicted vs observed event rate, binned | Standard in any clinical-ML paper post-2018 |
| **Calibration slope + intercept** | Numeric counterpart of the reliability diagram | TRIPOD+AI checklist |
| **Decision-curve analysis (DCA)** | Net benefit across a continuum of threshold probabilities, vs. "treat all" / "treat none" baselines | [Vickers & Elkin 2006](https://pubmed.ncbi.nlm.nih.gov/17099194/); [Vickers et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/); [DCA literature index, MSKCC](https://mskcc-epi-bio.github.io/decisioncurveanalysis/literature.html) |
| **Subgroup performance** | Same metrics stratified by age, sex, ethnicity | TRIPOD+AI mandate |

### 3.3 Calibration techniques

- **Platt scaling** — fit a logistic regression on the raw model output. Assumes sigmoidal distortion; works well for SVMs and small data.
- **Isotonic regression** — fit a non-parametric monotonic step function. More flexible; preferred for tree ensembles and complex models given enough hold-out data. Standard 2026 default for XGBoost / TFM outputs in clinical pipelines.
- **Temperature scaling** — single-parameter softmax temperature; popular for neural classifiers. Less applicable to binary tabular classifiers.

### 3.4 TRIPOD+AI (2024) — the reporting standard

[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) ([Collins et al., BMJ 2024](https://doi.org/10.1136/bmj-2023-078378)) is a 27-item checklist that supersedes the 2015 TRIPOD statement and is now expected in any clinical-prediction-model publication. It mandates:

- Reporting of model discrimination *and* calibration (not just AUROC).
- Reporting of **subgroup performance** in key demographic strata.
- Transparent reporting of training-data composition, missingness handling, and external-validation cohorts.
- Clear statement of intended use, intended user, and limitations.

A senior reviewer at any reputable clinical-ML venue in 2026 will check TRIPOD+AI compliance before reading the methods.

### 3.5 Model Cards

Companion to TRIPOD+AI: a [Model Card](https://arxiv.org/abs/1810.03993) (Mitchell et al. 2019) is a one-page (or short-section) standardised document accompanying any deployed model. It documents intended use, training-data distribution, performance metrics including subgroup breakdowns, ethical considerations, and known failure modes. Required for any 2026 clinical AI artefact aiming for credibility.

## 4. The Heart Failure Prediction (HFP) dataset — known pathologies

### 4.1 Composition

The Kaggle HFP dataset (fedesoriano, 2021) is the union of five legacy UCI heart-disease datasets:

| Source | Observations | Notes |
|---|---|---|
| Cleveland | 303 | Mixed-sex civilian |
| Hungarian | 294 | Mixed-sex civilian, European |
| Switzerland | 123 | Notably high cholesterol-missingness; in Kaggle curation, frequently zero-imputed |
| Long Beach VA | 200 | US Veterans Affairs, predominantly male |
| Stalog | 270 | Largely a re-curated subset of Cleveland + Hungarian (ML-community lore) |
| **Sum** | **1,190** | |
| **Kaggle published** | **918** | After removal of 272 duplicates |

### 4.2 Distribution shift

The five sources are clinically and demographically heterogeneous. Long Beach VA is almost exclusively male; Switzerland has near-zero recorded cholesterol for a large fraction of patients, an artefact of measurement protocol that becomes a "feature" once zero-imputed. Treating their union as i.i.d. violates the central assumption underlying random K-fold CV.

A model trained on the union with random K-fold CV will preferentially learn *hospital-of-origin shortcuts* (e.g., "cholesterol == 0" → high probability of being a Swiss record, then use the Swiss base rate). Out-of-distribution generalisation to a real Australian primary-care cohort is therefore not measured by such an evaluation.

### 4.3 Duplicate-row leakage risk

The 272-row gap between the union (1,190) and the published dataset (918) is a leakage warning. The community lore that Stalog largely overlaps with Cleveland + Hungarian is consistent with this, and means a randomised 80/20 split has a non-trivial chance of placing near-duplicate rows on both sides of the split. Reported test metrics from such a split overstate generalisation.

### 4.4 Realistic performance ceiling on HFP, 2022–2026 literature

Across the modern HFP benchmarks (XGBoost, CatBoost, TabPFN, light tabular DL):

- **Median AUROC: 0.88–0.92** under rigorous, non-leaking CV.
- **At ~85% specificity** (a clinically reasonable operating point for primary screening), **sensitivity typically lands in the 82–85% range**.

The prior Honours numbers (~89.7% sensitivity *and* ~89.6% specificity simultaneously) sit *above* the consensus ceiling for properly evaluated HFP models. The most likely explanations, in order of probability per the Deep Research synthesis:

1. **Memorisation by an overparameterised ensemble** at n=918 (CNN + LSTM + ANN with WOA-tuned hyperparameters has parameter count > training rows by orders of magnitude).
2. **Random-K-fold leakage** through near-duplicate Stalog ↔ Cleveland/Hungarian rows.
3. **Validation-set tuning leak** — when a metaheuristic optimiser (WOA) repeatedly queries a held-out validation set during hyperparameter search, the validation set is no longer held out; this is "overfitting to the val set". The eventual test number is optimistic by an unmeasured amount.
4. **Test-set lottery** at n=184 (20% of 918), where individual-rep variance on sensitivity is large.

A clean, leakage-controlled re-evaluation should be expected to drop the headline numbers materially. This is a falsifiable prediction; Phase 2.4 will test it.

### 4.5 Recommended evaluation protocol for HFP in 2026

- **Leave-One-Domain-Out CV (LODO-CV).** Train on four hospitals, test on the fifth, rotate. This measures *out-of-distribution* generalisation directly and breaks the duplicate-row leakage path. Standard in modern domain-shift evaluations.
- Report Brier, reliability, DCA, and per-source breakdown of performance.
- Per TRIPOD+AI, report subgroup performance (sex × age band).

## 5. Modern explainability for tabular clinical models

### 5.1 SHAP (TreeSHAP, KernelSHAP, DeepSHAP)

[Shapley values](https://www.rand.org/pubs/papers/P0295.html) (Shapley 1953) → [SHAP](https://arxiv.org/abs/1705.07874) (Lundberg & Lee 2017) for ML. Provides *local* (per-prediction) feature attributions with two desirable axiomatic properties: local accuracy (attributions sum to the prediction) and consistency (a feature's attribution rises if its true contribution rises).

- **TreeSHAP** ([Lundberg et al. 2020](https://www.nature.com/articles/s42256-019-0138-9), Nature MI) — exact, polynomial-time SHAP for tree ensembles (XGBoost, LightGBM, RF). The right choice for production clinical dashboards.
- **KernelSHAP** — model-agnostic, slow, approximate; falls back option for non-tree models.
- **DeepSHAP** — neural-net-specific; less reliable than TreeSHAP and rarely needed if the production model is a TFM/GBDT.

### 5.2 Counterfactuals — actionable XAI

[DiCE](https://github.com/interpretml/DiCE) ([Mothilal et al. 2020](https://arxiv.org/abs/1905.07697)), grounded in [Wachter et al. 2017](https://arxiv.org/abs/1711.00399). Generates "smallest perturbation that flips the prediction" — directly translatable into clinician-patient conversations ("if SBP drops 12 mmHg and LDL drops 0.5 mmol/L, risk band shifts from High to Intermediate"). Counterfactuals are now considered a *complement* to SHAP, not a replacement: SHAP says *why*, counterfactuals say *what to change*.

### 5.3 Anchors

[Anchors](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf) (Ribeiro et al. 2018). Generate IF–THEN rules that bound the model's local decision logic with a guaranteed precision. Less commonly deployed in clinical UIs than SHAP + counterfactuals; useful for documentation rather than per-prediction display.

### 5.4 Natural-language SHAP summarisers (the 2025–2026 UI shift)

The cognitive cost of reading SHAP force plots at clinical pace is real. The 2025–2026 pattern is to feed (raw features, prediction, SHAP vector, clinical guardrails) into a temperature-zero LLM that outputs a short narrative ("primary drivers… mitigating factors…"). The LLM acts as a *deterministic translation layer*, not a reasoner — the quantitative substance comes from TreeSHAP. References: [SHAP-LLM 2026](https://www.scitepress.org/Papers/2026/144857/144857.pdf) [unverified], [ContextualSHAP 2025](https://arxiv.org/html/2512.07178v1) [unverified], [Comparison of SHAP and clinician-friendly explanations, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12475050/).

## 6. Fairness and subgroup audits

### 6.1 Why the audits matter for CVD

- **Sex.** Pathophysiological trajectories differ: men present earlier with atherosclerotic events; women more often present post-menopause with HFpEF or microvascular dysfunction. Models that don't explicitly handle these trajectories under-predict in middle-aged women. Sources: [AHA 2024 women & CVD scientific statement](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001406); [Frontiers Cardiovasc Med 2024 sex-specific manifestation](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1403363/full); [Sex-specific ML for CVD risk in adults ≥80, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12751418/).
- **Age.** Same observed BP carries radically different prognosis in a 35- vs 85-year-old. Calibration drifts at the tails of the age distribution.
- **Ethnicity.** [AHA PREVENT 2023](https://www.acc.org/latest-in-cardiology/journal-scans/2025/12/01/21/57/new-research-adds) controversially removed race as an input to avoid biological essentialism. Subsequent 2024–2025 audits ([PREVENT diverse-group evaluation, AJMC 2025](https://www.ajmc.com/view/prevent-equations-accurately-predict-10-year-cvd-risk-across-diverse-groups), [Role of race in CVD prediction, PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11960820/)) showed that *aggregating* diverse ethnic populations can mask large calibration failures in specific subgroups.

### 6.2 What a 2026 audit looks like

For a CVD risk model, minimum viable audit:

- **Disaggregate** AUROC, Brier, sensitivity at clinical operating point, calibration slope across:
  - Sex (Male / Female / unknown)
  - Age bands (e.g., <50, 50–69, ≥70)
  - Where data permits, ethnicity (HFP does *not* support this; AusCVDRisk does)
- **Report fairness gaps explicitly** in the Model Card. A 89% aggregate sensitivity that hides a 50% sensitivity in elderly women is the failure mode TRIPOD+AI is designed to expose.

## 7. The Australian context (the bit the prior Honours work largely missed)

### 7.1 The 2023 Australian guideline shift

In 2023 the Australian cardiovascular landscape changed: the 2012 NVDPA guideline (built on 1990s-vintage US-centric **Framingham** equations) was retired and replaced by the **2023 Australian Guideline for Assessing and Managing Cardiovascular Disease Risk** and its companion **Australian CVD Risk Calculator** (`AusCVDRisk`, [cvdcheck.org.au](https://www.cvdcheck.org.au)).

References: [2023 AU CVD Guideline Med J Aus 2024](https://pubmed.ncbi.nlm.nih.gov/38623719/); [Heart Foundation guideline page](https://www.heartfoundation.org.au/for-professionals/guideline-for-managing-cvd); [ACDPA guideline + calculator page](https://www.acdpa.org.au/cvd-risk-guideline-calculator-2023); [MJA 2025 model-development paper](https://www.mja.com.au/journal/2025/223/4/development-and-calibration-2023-australian-cardiovascular-disease-risk).

### 7.2 What's under the hood

- Built on **PREDICT-1°**, a NZ primary-care cohort of >400,000 patients (recalibrated to AU mortality data).
- Includes variables **not present in HFP**: SEIFA (socio-economic index), uACR, eGFR, severe mental illness flag, First Nations identity, and expanded diabetic markers.
- Outputs a **5-year absolute risk** stratified into:
  - **High:** ≥10%
  - **Intermediate:** 5–<10%
  - **Low:** <5%
- Endorsed by RACGP and embedded in the **RACGP Red Book 10th Edition** (2025/2026): [RACGP Red Book 10th Ed page](https://www.racgp.org.au/clinical-resources/clinical-guidelines/key-racgp-guidelines/view-all-racgp-guidelines/preventive-activities-in-general-practice/what-s-new-in-the-10th-ed-red-book).
- Routine assessment: all adults 45–79 (and First Nations adults 30–79) without prior CVD.

### 7.3 Implication for the rebuild

A model trained on the 11 features of the 918-row HFP cannot legally, clinically, or scientifically substitute for AusCVDRisk. It is missing the very inputs the AU guideline considers essential, and it is calibrated to a non-AU historical cohort.

The defensible 2026 positioning is one of:

1. **Educational second-opinion / counterfactual-explorer wrapper.** UI computes the official AusCVDRisk score first; the ML artefact exists to surface non-linear feature interactions and DiCE counterfactuals for motivational interviewing.
2. **Methodological demonstrator.** Frame the artefact explicitly as "this is what a TRIPOD+AI-compliant, calibrated, fairness-audited tabular clinical model looks like end-to-end" — the *engineering process* is the deliverable, not the clinical claim.

The CardioRisk Co-Pilot rebuild adopts both framings; see [04-revised-design.md §6](./04-revised-design.md).

## 8. Cross-checks and confidence-by-claim

Where the Deep Research synthesis stretched into prospective 2026 references, I rated my confidence as follows:

| Claim | Confidence | Notes |
|---|---|---|
| Tree-based models still win on small tabular data, deep tabular DL is brittle | High | Multiple peer-reviewed sources: Borisov 2022, Grinsztajn 2022, replicated 2023–2025 |
| TabPFN matches XGBoost zero-shot at n < 10k | High | Hollmann et al. NeurIPS 2023 + 2024–2025 follow-ups |
| TabPFN v2.5 / v2.6 specifically | Medium-high | Paper is recent, benchmark URLs partially unverified, but headline claim consistent across sources |
| Metaheuristic FS literature critique (Bestiary, Sörensen, Aranha) | High | Established, peer-reviewed |
| HFP duplicate-row leakage risk | Medium-high | Documented in dataset description (272 deduped); Stalog ↔ Cleveland/Hungarian overlap is community lore but well-known |
| Realistic HFP ceiling AUROC 0.88–0.92 | Medium | Consistent with several HFP benchmarks I've seen; no single canonical citation |
| TRIPOD+AI 2024 mandate | High | Published in BMJ; widely adopted |
| AusCVDRisk replaces NVDPA / Framingham as of 2023 | High | Officially endorsed by Heart Foundation, RACGP, ACDPA |
| RACGP Red Book 10th Ed (2025/2026) AusCVDRisk integration | High | Confirmed via RACGP page |
| Some 2026-stamped arXiv URLs in the synthesis | Low (URL); claim itself often medium | Treated as `[unverified]` above and not used to anchor critical-review verdicts |

## 9. What I deliberately did not cover

- **Survival modelling (Cox / DeepSurv / DeepHit).** HFP does not have time-to-event data; the prior study and the rebuild treat it as a binary classification problem.
- **Multi-omics / imaging / EHR-based CVD models.** Out of scope for the tabular co-pilot.
- **Causal inference and treatment-effect estimation.** A genuine "what would happen if we treated this patient" model needs causal scaffolding (Pearl 2009, Hernán & Robins 2020) that is beyond Phase 1.
- **Federated learning, privacy-preserving training, differential privacy.** Relevant if ever evaluating on real PHI; not relevant for synthetic-only.
- **Regulatory pathways (TGA, FDA SaMD).** This is a research artefact, not a medical device. AGENTS.md §1 forbids treating it otherwise.

These are documented here so readers can see the explicit boundary of what 02 covers.

---

## Where the verdict lives

`02` is descriptive. The opinionated head-to-head verdict on each Honours design choice lives in [`03-critical-review.md`](./03-critical-review.md). The proposed v1 design lives in [`04-revised-design.md`](./04-revised-design.md), and the binding decision lives in [ADR-006](../adr/006-risk-model-architecture.md).
