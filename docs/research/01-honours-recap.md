# 01 — Honours-work recap

> **Source.** Sanitised summary of a Monash University BE(Hons) Final Year Project, "Analysing Cardiovascular Diseases through a Deep Learning Platform" (S2 2024). The original report and supplementary code are held privately outside this repository — only the methodology and headline numbers needed to inform the rebuild are reproduced here. No author names, student IDs, or supervisor identities appear in the public repo.
>
> **Purpose.** Establish a faithful baseline of *what was done* in the prior study so [`02-current-soa.md`](./02-current-soa.md), [`03-critical-review.md`](./03-critical-review.md), and [`04-revised-design.md`](./04-revised-design.md) can argue *what should change*. This document is descriptive, not evaluative — opinions live in `03-critical-review.md`.

---

## 1. Aim of the prior study

To evaluate whether deep-learning models, combined with metaheuristic-based feature selection, could improve the accuracy of premature CVD detection in urban populations beyond traditional ML baselines, and to deploy the best configuration as a web platform for healthcare professionals.

## 2. Datasets

The prior study evaluated its model architectures on four publicly available UCI datasets (used for benchmarking the architecture x feature-selection grid), and then trained the top configurations on the Heart Failure Prediction dataset (HFP) for the actual CVD task.

### Architecture-evaluation datasets (UCI)

| Dataset | Task | Features | Instances | Notes |
|---|---|---|---|---|
| Semeion Handwritten Digit | Multi-class (10) | 256 | 1,593 | Image-style benchmark |
| DARWIN Handwriting | Binary | 450 | 174 | Alzheimer prediction from handwriting |
| Malicious Executable | Binary | 531 | 373 | Malware detection |
| ARCENE | Binary | 10,000 | 900 | NIPS 2003 feature-selection challenge |

The intent of using these (non-clinical) datasets to choose the model architecture was to demonstrate generalisability across data sizes and dimensionalities before applying to HFP.

### CVD dataset (HFP)

- **Source:** Kaggle, *Heart Failure Prediction Dataset* (fedesoriano, 2021) — a union of five legacy UCI heart-disease datasets (Cleveland, Hungary, Switzerland, Long Beach VA, and Stalog).
- **Composition:** 1,190 instances reduced to **918** after de-duplication, with **11 features** plus a binary `HeartDisease` label (`0` = no disease, `1` = disease).
- **Features:** age, sex, chest pain type (4 categories), resting BP, cholesterol, fasting blood sugar, resting ECG (3 categories), max heart rate, exercise-induced angina, oldpeak (ST depression), ST slope (3 categories).

## 3. Preprocessing pipeline

1. **Missing-value imputation** — MissForest (random-forest-based iterative imputation), chosen on the basis of Waljee et al. (2013), which reported lower imputation error than mean / kNN / MICE on medical lab data.
2. **Numeric normalisation** — min-max scaling to [0, 1].
3. **Categorical encoding** — one-hot.
4. **Train / test split** — 80 / 20, single split (no nested cross-validation).
5. **Feature selection** — applied as a separate preprocessing step before model training (see §5).

## 4. Model architectures

The prior study compared two deep-learning architectures.

### 4.1 WOCLSA (CNN + LSTM + ANN, WOA-tuned)

- **Origin:** Su et al. (2023) — originally proposed for COVID-19 infection prediction (>91% across all metrics on that task).
- **Structure:** sequential combination of a 1D CNN, an LSTM, and a fully-connected ANN head.
- **Hyperparameter tuning:** the **Whale Optimisation Algorithm** (WOA, Mirjalili & Lewis, 2016) tunes:
  - number of neurons per layer
  - dropout rate
  - batch size

### 4.2 Ensemble (DNN + CNN + RNN + BiRNN)

- **Origin:** Midhun et al. (2023).
- **Structure:** four parallel sub-networks (DNN, 1D CNN, RNN, BiRNN), trained independently and combined into a single prediction.
- **No metaheuristic hyperparameter tuning** in the base configuration; metaheuristic methods enter only via the *feature-selection* layer (§5).

## 5. Feature-selection methods compared

The prior study evaluated **11 conditions** (one "no FS" baseline + 10 FS techniques) crossed with the two architectures, giving 22 model configurations per dataset.

| Family | Method |
|---|---|
| Statistical | Recursive Feature Elimination (RFE), RFE with cross-validation (RFE-CV) |
| Tree-based | Random Forest (RF) importance |
| Evolutionary | Genetic Algorithm (GA) |
| Swarm — nature-inspired | Grey Wolf Optimisation (GWO), Whale Optimisation Algorithm (WOA), Harris Hawks Optimisation (HHO), Firefly Algorithm (FA), Cuckoo Search (CS), Bat Algorithm (BA) |

Note that WOA appears at *two* levels: as a hyperparameter optimiser inside WOCLSA, and as a stand-alone feature-selection method that can be paired with either architecture (e.g. `WOA-Ensemble`).

## 6. Evaluation protocol

- **Split:** 80 / 20, single split per dataset.
- **Metrics reported:** loss, accuracy, precision, sensitivity (recall), specificity, F1, ROC-AUC.
- **Selection criterion:** sensitivity and specificity prioritised because of the clinical cost of false negatives in CVD screening.
- **No reported:** calibration metrics (Brier score, reliability diagrams), decision-curve analysis, fairness gaps across age or sex, confidence intervals, statistical-significance tests, nested cross-validation, or per-source (Cleveland / Hungary / etc.) breakdown of HFP performance.

## 7. Headline results on the architecture-evaluation datasets

The full 22-configuration × 4-dataset grid is reproduced verbatim in the prior study's results tables. Average performance across the four UCI datasets, by architecture family:

| Architecture | Mean accuracy | Mean sensitivity | Mean specificity | Mean F1 | Mean ROC-AUC |
|---|---|---|---|---|---|
| Ensemble (best 4 FS methods averaged) | ~93–94% | ~91–93% | ~93–95% | ~91–93% | ~95–97% |
| WOCLSA (best 4 FS methods averaged) | ~89–92% | ~88–92% | ~91–94% | ~88–92% | ~93–96% |

The prior study concluded that the **Ensemble architecture outperformed WOCLSA on average across the UCI benchmarks**, particularly on average sensitivity and specificity. The four feature-selection methods that produced the best Ensemble configurations were: **GA, GWO, WOA, and CS**.

## 8. Headline results on the CVD task (HFP)

The four top-performing Ensemble configurations from §7 were then trained and evaluated on HFP. Reported results (single 80/20 split, no confidence intervals reported in the original study):

| Configuration | Sensitivity | Specificity |
|---|---|---|
| GA-Ensemble | 84.11% | 89.61% |
| GWO-Ensemble | 86.92% | 83.12% |
| **WOA-Ensemble** | **89.72%** | 83.12% |
| CS-Ensemble | 86.92% | 87.01% |

WOA-Ensemble was identified as the headline configuration on the basis of the highest sensitivity. No AUROC, Brier score, calibration plot, or per-source-dataset performance was reported for HFP in the final report.

> **Implementation gap surfaced in Phase 2.4 (2026-05-05).** The supplied Honours archive (`Demos/`, four notebooks) contains the **Ensemble architecture** code in full (`Data_Pre-processing.ipynb` cell 55 — DNN + 1D CNN + RNN(LSTM) + BiRNN(BiLSTM), mean-averaged) but does **not** contain a working implementation of the WOA, GWO, CS, BA, FA, HHO, or RFE-CV feature-selection layers. In `Data_Pre-processing.ipynb` the cells for `WOA` (cell 41), `GWO` (cell 39), `CS`, `BA`, `FA`, `HHO`, and `RFE-CV` are all empty `pass`-equivalent placeholders under markdown section headers. Working FS code in the archive: GA, EAGA, RF, RFE only. The Honours WOCLSA architecture's WOA hyperparameter-tuning layer is similarly absent — `cnnlstma()` (cell 51) ships with fixed hyperparameters. The headline numbers in the table above were therefore generated by code that is no longer in the supplied archive (likely lived on a separate Colab and was not preserved into handover). Phase 2.4 reproduces only the Ensemble *architecture* faithfully under our LODO protocol; the full honesty discussion, the implications for cross-model comparison, and the rationale for not reconstructing WOA from scratch live in [`09-honours-vs-v1.md`](./09-honours-vs-v1.md) and [ADR-012](../adr/012-honours-baseline-reproduction.md).

## 9. Web platform (out of scope for the rebuild)

The prior study also built a MERN-stack web application ("Cardio Vision AI") that wrapped the trained model behind a TensorFlow.js inference call, with MongoDB-backed user accounts and patient records. That platform is **not** part of this rebuild — the CardioRisk Co-Pilot frontend is being designed from scratch in Phase 5 (see [AGENTS.md §7](../../AGENTS.md)).

## 10. Stated limitations from the prior study

The prior study itself acknowledged the following limitations, which we will treat as inputs to the critical review:

1. **Computational resources.** Training was constrained to CPU / single-GPU consumer hardware.
2. **Data availability.** Only one CVD-relevant dataset (HFP, n=918) was used end-to-end; no external validation cohort.
3. **No clinical validation.** No usability testing with healthcare professionals; no clinical trial.
4. **UI/UX maturity.** Acknowledged that the web platform was not optimised for accessibility or cross-device use.
5. **Security posture.** Acknowledged that the platform lacked specialist cybersecurity review.

## 11. Stated future work from the prior study

The prior study proposed three threads of future work:

1. Parameter fine-tuning of the metaheuristic feature-selection methods (population size, iterations, minimum-feature constraint).
2. **Ensemble feature selection** — combining several FS methods via rank aggregation (Janani et al., 2023) for improved robustness.
3. Web-platform improvements: clinical trials, in-app model comparison, real-time model retraining.

---

## What this document does not do

- **Does not evaluate** the methodology choices. The verdict on whether each choice is still defensible in 2026 lives in [`03-critical-review.md`](./03-critical-review.md), grounded in the current state of the art summarised in [`02-current-soa.md`](./02-current-soa.md).
- **Does not recommend** what to keep, change, or discard for the v1 rebuild. That recommendation lives in [`04-revised-design.md`](./04-revised-design.md) and the binding decision in [ADR-006](../adr/006-risk-model-architecture.md) (forthcoming).
- **Does not republish** any private artefacts from the prior study (full report, code, presentation slides, author identities). Only methodology and headline numbers.

## Citations from the prior study referenced above

The following primary sources were cited in the prior study's literature review and are reused here when relevant. The prior study used a mix of APA / Vancouver styles; here we standardise on a minimal author–year format with DOIs where available.

- Su, X., Sun, Y., Liu, H., et al. (2023). *An innovative ensemble model based on deep learning for predicting COVID-19 infection.* Scientific Reports, 13(1). [doi:10.1038/s41598-023-39408-8](https://doi.org/10.1038/s41598-023-39408-8) — origin of WOCLSA.
- Midhun, J., et al. (2023). *Ensemble deep learning models for accurate prediction of cardiovascular disease risk.* ICECAA. [doi:10.1109/ICECAA58104.2023.10212248](https://doi.org/10.1109/ICECAA58104.2023.10212248) — origin of the DNN+CNN+RNN+BiRNN ensemble.
- Mirjalili, S., & Lewis, A. (2016). *The Whale Optimization Algorithm.* Advances in Engineering Software, 95. [doi:10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008) — origin of WOA.
- Heidari, A. A., Mirjalili, S., Faris, H., et al. (2019). *Harris hawks optimization: Algorithm and applications.* Future Generation Computer Systems, 97. [doi:10.1016/j.future.2019.02.028](https://doi.org/10.1016/j.future.2019.02.028) — origin of HHO.
- Premalatha, M., Jayasudha, M., Čep, R., et al. (2024). *A comparative evaluation of nature-inspired algorithms for feature selection problems.* Heliyon, 10(1). [doi:10.1016/j.heliyon.2023.e23571](https://doi.org/10.1016/j.heliyon.2023.e23571) — comparison of GWO, WOA, BA.
- Yang, X.-S. (2020). *Nature-inspired optimization algorithms: Challenges and open problems.* Journal of Computational Science, 46. [doi:10.1016/j.jocs.2020.101104](https://doi.org/10.1016/j.jocs.2020.101104) — Firefly + Cuckoo Search.
- Waljee, A. K., et al. (2013). *Comparison of imputation methods for missing laboratory data in medicine.* BMJ Open, 3(8). [doi:10.1136/bmjopen-2013-002847](https://doi.org/10.1136/bmjopen-2013-002847) — MissForest justification.
- Hicks, S. A., et al. (2022). *On evaluation metrics for medical applications of artificial intelligence.* Scientific Reports, 12(1). [doi:10.1038/s41598-022-09954-8](https://doi.org/10.1038/s41598-022-09954-8) — sensitivity / specificity definitions.
- Janani, K., et al. (2023). *Ensemble feature selection using Bonferroni, OWA and Induced OWA aggregation operators.* Applied Soft Computing, 143. [doi:10.1016/j.asoc.2023.110431](https://doi.org/10.1016/j.asoc.2023.110431) — basis for proposed FS-ensemble future work.
- fedesoriano. (2021). *Heart Failure Prediction Dataset.* Kaggle. [URL](https://www.kaggle.com/fedesoriano/heart-failure-prediction).
