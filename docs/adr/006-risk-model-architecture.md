# ADR-006: Risk-model architecture for v1

- Status: **Proposed** (becomes Accepted on Phase 1 checkpoint approval)
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 1
- Supersedes: the AGENTS.md placeholder "ADR-001 for the risk model"

## Decision

The v1 CardioRisk Co-Pilot risk model is built around the following stack:

- **Headline (primary) model:** TabPFN v2.5 / v2.6 (zero-shot, in-context, calibrated by construction).
- **White-box baseline:** XGBoost with isotonic post-hoc calibration, hyperparameters tuned by Optuna (Bayesian optimisation, strict early stopping).
- **Transparency anchor:** L1 logistic regression with restricted-cubic-spline expansions on continuous features.
- **Honesty baseline:** the prior Honours WOA-Ensemble (CNN + LSTM + ANN with WOA-tuned hyperparameters), reimplemented and run under the same protocol.
- **Evaluation protocol:** Leave-One-Domain-Out cross-validation across the five HFP source datasets (Cleveland / Hungarian / Switzerland / Long Beach VA / Stalog), reported with bootstrapped 95% CIs.
- **Reporting standard:** TRIPOD+AI (2024) — full 27-item checklist mapping in `MODEL_CARD.md`, including subgroup performance stratified by sex and age band.
- **Explainability:** TreeSHAP for XGBoost, KernelSHAP for TabPFN, raw coefficients for L1 LR, plus DiCE counterfactuals and a temperature-zero LLM natural-language summariser whose outputs are NLI-verified against the SHAP vector before display.
- **Clinical positioning:** explicitly an educational second-opinion / counterfactual-explorer subordinate to the Australian CVD Risk Calculator (PREDICT-1°-based, RACGP-endorsed, Red Book 10th Edition 2025/2026). Not a replacement, not a primary clinical tool.

The full design rationale is in [`docs/research/04-revised-design.md`](../research/04-revised-design.md). The verdict against the prior Honours work that motivates each choice is in [`docs/research/03-critical-review.md`](../research/03-critical-review.md). The current state of the art that anchors the choices is summarised in [`docs/research/02-current-soa.md`](../research/02-current-soa.md).

## Context

The prior Honours study (Monash, 2024) shipped a CNN + LSTM + ANN ensemble with hyperparameters tuned by the Whale Optimisation Algorithm and feature selection from a stack of 10 nature-inspired metaheuristics, evaluated on a single 80/20 random split of the 918-row Heart Failure Prediction (HFP) dataset, reporting headline numbers of ~89.7% sensitivity / ~89.6% specificity.

Modern (2025–2026) tabular-ML and clinical-ML literature provides strong evidence that:

1. Deep sequence/spatial models on static tabular data are theoretically misaligned and empirically inferior to GBDTs and Tabular Foundation Models on small clinical cohorts (Borisov et al. 2022, Grinsztajn et al. 2022, multiple replications since).
2. The "nature-inspired metaheuristic" feature-selection literature is now widely regarded as folklore — algorithms repackaging PSO / ES under zoological metaphors, with documented centre-seeking bias on classic benchmark functions and no convergence guarantees (Aranha et al. 2022, Campelo et al. 2025, EC Bestiary, Sörensen 2025).
3. AUROC + sensitivity + specificity reporting is no longer sufficient for a 2026 clinical model; calibration (Brier, reliability, isotonic), Vickers' decision-curve analysis, and TRIPOD+AI subgroup audits are the modern minimum (BMJ 2024, Vickers & Elkin 2006, Mitchell et al. 2019).
4. The HFP dataset has known distribution-shift and duplicate-row pathologies that make random K-fold CV a leakage-prone evaluation; LODO-CV is the modern correction.
5. The realistic published performance ceiling on HFP under leakage-controlled CV is ~AUROC 0.88–0.92 with sensitivity 82–85% at 85% specificity. The Honours headline sits above this consensus, most plausibly explained by memorisation, leakage, and validation-set tuning.
6. The Australian clinical context shifted in 2023: NVDPA / Framingham was retired in favour of AusCVDRisk (PREDICT-1° recalibrated to AU mortality, RACGP-endorsed). A model trained on HFP cannot replace AusCVDRisk; it can at best wrap it.

A direct port of the prior architecture would inherit all of these issues without the engineering signal of having addressed them. A clean rebuild around a modern, calibrated, honestly-evaluated stack is the higher-signal choice.

## Consequences

### Positive

- **The headline numbers in `MODEL_CARD.md` will be defensible to a senior clinical-ML reviewer.** Every choice in the stack maps to a peer-reviewed primary source and a TRIPOD+AI checklist item.
- **The artefact reads as engineering judgement, not architecture novelty-seeking.** A reader sees "TabPFN + calibrated XGBoost + LR + honest WOA baseline" and immediately understands the maintainer knows what 2026 actually looks like.
- **The honesty baseline (Phase 2.4) is itself a recruiter signal.** A maintainer who reproduces their own prior work as a baseline and reports the result honestly demonstrates the eval discipline AGENTS.md §3 is asking for.
- **Calibration-first reporting** means downstream UI (Phase 5) can render absolute-risk percentages confidently, rather than ranking-only outputs that have to be hedged.
- **LODO-CV** measures what we actually care about (out-of-distribution generalisation across hospital sources) and exposes the source-shift / leakage issues openly.

### Negative

- **TabPFN is opaque relative to TreeSHAP-on-XGBoost.** Mitigated by reporting all three models; readers can choose their explainability vs. performance trade-off.
- **No external Australian validation cohort.** Listed as honest weakness in `04-revised-design.md §9` and as future scope in `AGENTS.md §8`.
- **The published headline on HFP under LODO-CV is likely to be lower than the Honours headline.** Documented as expected ([03 §1](../research/03-critical-review.md#1-choice--dataset-heart-failure-prediction-kaggle-fedesoriano)) and treated as a credibility win, not a regression.
- **The clinical positioning is deliberately modest.** Anyone landing on the README expecting "AI for CVD prediction" must be told this is an educational second-opinion. The README leads with disclaimers; the UI leads with disclaimers; the Model Card leads with disclaimers.
- **The Phase-2.4 reimplementation of WOA-Ensemble is engineering work that doesn't go into the v1 production model.** This is a cost we accept in exchange for an empirically grounded verdict.

### Easier now

- Justifying the model choice to a stranger reading the repo cold (a recruiter, a contributor, a reviewer).
- Adding subgroup audits, fairness gaps, and DCA — they're built into the eval card from day one rather than retrofitted.
- Onboarding new contributors who already know XGBoost / SHAP / TRIPOD+AI; no metaheuristic-folklore stack to explain or maintain.

### Harder now

- Claiming any kind of architectural novelty. The artefact's value is the engineering process, not a model contribution. AGENTS.md §3 already accepts this trade.
- Inheriting the prior 89.7% sensitivity headline without empirical re-evaluation. We accept that this is *good* — sliding past that number quietly would be intellectually dishonest.

## Alternatives considered

### A. Direct port of WOA-Ensemble as the v1 headline

Rejected. Theoretically misaligned with static tabular data (CNN assumes spatial proximity across feature columns; LSTM/BiRNN assume time-step sequencing across the 11-feature row); overparameterised by orders of magnitude at n = 918; metaheuristic optimiser introduces validation-set tuning leak. The headline numbers under proper protocol are predicted to be substantially lower than the prior reported numbers, and the architecture is hard to defend to a senior reviewer in 2026. Detailed verdict in [03 §4](../research/03-critical-review.md#4-choice--model-architecture-woclsa-cnn--lstm--ann-woa-tuned-and-ensemble-dnn--cnn--rnn--birnn).

### B. XGBoost-only (skip TabPFN)

Rejected as the *primary* model, accepted as a baseline. XGBoost is the durable workhorse and absolutely belongs in the eval. But the 2024–2026 evidence on TFMs (TabPFN matching/beating tuned XGBoost zero-shot at n < 10k) is strong enough that not running TabPFN would be the conservative-to-a-fault choice, and would miss a credible engineering signal — "I shipped a TFM in production-like conditions in early 2026". The marginal cost of also running TabPFN is low.

### C. FT-Transformer / SAINT / TabNet / NODE / TabR / RealMLP

Rejected. The 2024–2026 consensus on tabular DL is that these architectures need n > 50,000 rows and extensive tuning to reach parity with XGBoost; on n = 918 they reliably underperform. They would also require us to re-introduce GPU-scale training to the pipeline, increasing cost and complexity for no expected benefit. ([Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815), updated through 2026.)

### D. EBM (Explainable Boosting Machine)

Considered as a third white-box baseline alongside L1 LR. Plausible — within 1–2 AUROC points of XGBoost on small clinical data with native interpretability. Deferred to a possible v1.1 to keep the v1 stack focused; will revisit if Phase 2.3 has bandwidth.

### E. Reimplement PREDICT-1° / AusCVDRisk equations as a peer baseline

Rejected for v1. Done cleanly, this would require pulling AusCVDRisk's official documentation, parsing the equations, validating against a synthetic AU-shaped cohort, and explaining the calibration delta. It's substantial work that belongs in v2 and is listed in `AGENTS.md §8` future scope.

### F. Survival modelling (Cox PH / DeepSurv / DeepHit)

Rejected. HFP has no time-to-event field; it's a binary classification dataset. Survival modelling would require a different dataset (e.g., MIMIC-IV with credentialed access, or PREDICT-1°-derived synthetic data), which is outside the v1 scope.

### G. TabuLa-8B / LLM-as-tabular-classifier

Considered. We already have an LLM in the loop for Phase 3+ generation, so adding LLM-as-classifier is architecturally consistent. Rejected for v1 on cost/complexity grounds — the marginal benefit at n = 918 with named clinical features is small, and TabPFN already covers the "modern foundation-model" angle. Listed for possible Phase 6 multi-model evaluation.

### H. Drop the Honours-comparison baseline (Phase 2.4)

Rejected. Skipping the WOA-Ensemble re-run would weaken the rebuild's credibility — the verdict in `03-critical-review.md` would be theoretical only. Running it under the same LODO-CV protocol gives an empirical grounding that a reviewer can verify, and it's the same engineering discipline AGENTS.md §3 calls for.

## Trigger to revisit

Re-open this ADR (with a superseding ADR) if any of the following becomes true:

- TabPFN's licence changes in a way that prevents distribution from this repo.
- A peer-reviewed reproduction publishes evidence that an FT-Transformer-class model meaningfully beats TabPFN + XGBoost + isotonic on HFP under leakage-controlled CV.
- The 2026/27 update of the AusCVDRisk equations makes a wrapper model untenable (e.g., the official tool ships with the same explainability and counterfactual surface we'd add).
- The maintainer decides to swap the dataset (HFP → MIMIC-IV / synthetic-PREDICT-1° / other) — would necessitate a full design refresh.
- A senior reviewer demonstrates a structural critique we missed (always possible — the public PR review process is welcome).

## References

The full citation set lives in [`docs/research/02-current-soa.md`](../research/02-current-soa.md) and [`docs/research/03-critical-review.md`](../research/03-critical-review.md). Key anchors:

- [Borisov et al. 2022 — DNNs and Tabular Data: A Survey](https://arxiv.org/abs/2110.01889)
- [Grinsztajn et al. 2022 — Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815)
- [Hollmann et al. 2023 — TabPFN](https://arxiv.org/abs/2207.01848)
- [Aranha et al. 2022 — Six misleading optimization techniques inspired by bestial metaphors](https://www.researchgate.net/publication/362280739)
- [Lundberg et al. 2020 — TreeSHAP](https://www.nature.com/articles/s42256-019-0138-9)
- [Vickers & Elkin 2006 — Decision curve analysis](https://pubmed.ncbi.nlm.nih.gov/17099194/)
- [Collins et al. 2024 — TRIPOD+AI (BMJ)](https://www.bmj.com/content/385/bmj-2023-078378)
- [Mitchell et al. 2019 — Model Cards](https://arxiv.org/abs/1810.03993)
- [Mothilal et al. 2020 — DiCE counterfactuals](https://arxiv.org/abs/1905.07697)
- [2023 Australian CVD Risk Guideline (MJA)](https://pubmed.ncbi.nlm.nih.gov/38623719/)
