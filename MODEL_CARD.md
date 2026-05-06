# Model Card — CardioRisk Co-Pilot v1 risk models

> **Reading order.** This card is the user-facing summary of [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) (full Phase 2.3b + 2.4 LODO results) and [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md) (Honours-baseline honesty discussion). Read those for the per-fold tables, per-subgroup audit, decision-curve analysis, and bootstrap CIs.
>
> **Status.** Phase 2.4 deliverable. v1 = the four-model risk-prediction stack the rest of the CardioRisk Co-Pilot system is built on. Numbers below are produced verbatim by `backend/scripts/train_v1.py` from `data/processed/combined.parquet` (the Heart Failure Prediction dataset's underlying UCI sources combined under the HFP schema).
>
> **TL;DR.** The cardiovascular-risk module ships **four** binary classifiers — TabICL (TFM), L1 LR with restricted-cubic-spline expansion, XGBoost, and a faithful PyTorch port of the Honours team's 4-net mean-averaged Ensemble — evaluated under Leave-One-Domain-Out CV across the four UCI sources, with post-hoc calibration on a within-fold calibration slice, bootstrap CIs, subgroup audits, and decision-curve analysis at the AusCVDRisk thresholds. **TabICL is the headline model by AUROC, AUPRC, Brier, and calibration slope. L1 LR is the strongest white-box.** XGBoost suffers from isotonic-on-small-slice calibration collapse (slope 0.21). The Honours-Ensemble is reproduced honestly — without the WOA feature-selection layer (because the WOA code is not in the supplied archive); see §3 below and [ADR-012](docs/adr/012-honours-baseline-reproduction.md).

---

## 1. Intended use

This is a **research artefact, not a clinical product.** It exists to demonstrate that the author can build, evaluate, and honestly report on a clinical-domain ML system. It is *not* approved as a medical device, has not been validated in a clinical setting, and must not be used for real patient care.

**Intended downstream consumers:**

- The CardioRisk Co-Pilot agentic system (this repo, Phase 3+), which uses one of the trained models as the risk-score component of a larger explainability + retrieval + drafting pipeline.
- Recruiters / contributors auditing this repo's modelling work.

**Not intended for:**

- Real patient care, EHR integration, or clinical decision support.
- Populations the LODO procedure does not cover (the four UCI sources: Cleveland, Hungarian, LongBeachVA, Switzerland) — particularly any non-European, non-North-American cohort, or a population with a substantially different prevalence profile.
- The LongBeachVA ≥70 stratum, which is structurally under-served by every v1 model under our LODO protocol (see §6 below).

## 2. Models

| Slot | Model | Purpose | Calibration |
|---|---|---|---|
| Headline | **TabICL 2.1** ([Inria Soda](https://github.com/soda-inria/tabicl), BSD-3) | Tabular Foundation Model; zero-shot in-context learning. | Native (passes through unwrapped — calibrated by training objective). |
| Workhorse | **XGBoost 3.x** + Optuna (50 trials / 10-min cap) | White-box gradient boosted trees; the standard tabular baseline. | Isotonic on within-fold calibration slice. |
| Transparency anchor | **L1 LR with RCS** (saga, GridSearchCV C∈{0.001..100}) | Restricted-cubic-spline expansion of continuous features → linear model with nonzero coefficients only. The clinically interpretable model. | Sigmoid (Platt). |
| Honours baseline | **Ensemble** (PyTorch port of Honours `Demos/Data_Pre-processing.ipynb` cell 55) | 4 parallel sub-networks (DNN + 1D CNN + LSTM + BiLSTM), mean-averaged sigmoid output. The Honours team's architecture, faithfully reproduced. | Sigmoid (Platt). |

Wrapper code: [`cardiorisk/models/`](backend/cardiorisk/models/). All four wrappers implement the same `ModelWrapper` Protocol so the training driver and downstream consumers don't branch per model.

## 3. Headline results — LODO across the four UCI sources

**Methodology.** Leave-One-Domain-Out CV: each of the four UCI sources (Cleveland, Hungarian, LongBeachVA, Switzerland) is held out in turn while the model trains on the other three. Within each LODO fold, the training pool is split 80/10/10 into train / calibration / [unused — held for inner-CV in LR + Optuna in XGBoost]. Headline metrics: AUROC, AUPRC, Brier ↓, calibration slope (ideal=1), sensitivity at 85% / 90% specificity. Bootstrap CIs (2,000 percentile resamples) are reported per fold in [`reports/v1/metrics_per_fold.json`](reports/v1/metrics_per_fold.json).

| Model | AUROC | AUPRC | Brier ↓ | Calib. slope | Sens@85% | Sens@90% |
|---|---|---|---|---|---|---|
| TabICL | **0.811 ± 0.085** | **0.891 ± 0.055** | **0.150 ± 0.016** | 0.97 ± 0.30 | 0.567 ± 0.215 | 0.437 ± 0.277 |
| LR (L1+RCS) | 0.804 ± 0.082 | 0.883 ± 0.063 | 0.194 ± 0.037 | 0.75 ± 0.34 | **0.589 ± 0.136** | **0.457 ± 0.226** |
| XGBoost | 0.779 ± 0.081 | 0.826 ± 0.102 | 0.218 ± 0.041 | 0.21 ± 0.30 | 0.186 ± 0.195 | 0.103 ± 0.183 |
| Ensemble *(Honours architecture)* | 0.792 ± 0.076 | 0.860 ± 0.071 | 0.197 ± 0.024 | **1.02 ± 0.48** | 0.585 ± 0.138 | 0.370 ± 0.258 |

Mean ± standard deviation across the four LODO folds, from `reports/v1/metrics_aggregate.json`. Per-fold values + bootstrap CIs are in `reports/v1/metrics_per_fold.json`.

### Per-source breakdown (AUROC)

| Held-out source | TabICL | LR | XGBoost | Ensemble |
|---|---|---|---|---|
| Cleveland (n=303, prev=0.46) | **0.877** | 0.863 | 0.838 | 0.832 |
| Hungarian (n=294, prev=0.36) | **0.893** | 0.886 | 0.859 | 0.877 |
| LongBeachVA (n=200, prev=0.75) | 0.740 | 0.733 | 0.702 | **0.745** |
| Switzerland (n=123, prev=0.94) | **0.736** | 0.733 | 0.717 | 0.714 |

Full per-fold tables (including AUPRC, Brier, calibration, sensitivity) are in [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) §2. The qualitative reading consistent across all four models:

- **Cleveland and Hungarian** are the easiest folds. TabICL leads, all four are within ~5pp.
- **LongBeachVA** — every model loses ~10–15pp AUROC vs. the easier folds. Highest cholesterol-missingness, prevalence inversion vs. the training pool. The Honours-Ensemble *just* edges TabICL on AUROC here (0.745 vs 0.740) and is the only model whose net benefit at the 10% threshold beats treat-all (DCA §6); both within bootstrap noise.
- **Switzerland** — degenerate-by-design (negative class is borderline absent). All metrics are noisy on this fold; we do not drop it from LODO.

## 4. Subgroup audit

Per-(model × stratum) AUROC for sex and age band, with `min_stratum_size` guarding against meaningless small strata. Full table in [`docs/research/08-v1-model-results.md`](docs/research/08-v1-model-results.md) §3. Headlines:

- **Sex (F vs M):** Cleveland and Hungarian are the only auditable folds (LongBeachVA has F=6, Switzerland F=10 — both below the guard). Within auditable folds, Cleveland favours F slightly across every model (gaps 0.04–0.08); Hungarian favours M across every model (gaps 0.04–0.14). The Honours-Ensemble has the **largest Hungarian sex gap (0.142)** of the four — TabICL 0.099, LR 0.054, XGBoost 0.037 — flagged as the strongest single subgroup-audit reason not to prefer the Ensemble for deployment.
- **Age band (<50 / 50–69 / ≥70):** the ≥70 stratum on LongBeachVA (n=16) is the structural weak spot for **every v1 model.** AUROC there: TabICL 0.464, LR 0.536, XGBoost 0.518, Ensemble 0.393. The Ensemble is the worst on this stratum and posts the largest cross-stratum gap on LongBeachVA (0.440). The Honours architecture does not close this gap — it widens it. **The LongBeachVA ≥70 stratum is therefore declared out-of-scope for any deployment use of these four models** (see §7).

## 5. Calibration story (read carefully)

Phase 2.3b's empirical finding ([`08-v1-model-results.md`](docs/research/08-v1-model-results.md) §4): the within-fold calibration slice sits at ~50–100 rows per LODO fold. **Isotonic regression on this slice size collapses XGBoost** (cross-fold mean calibration slope = 0.21, vs ideal=1). The collapse manifests as a near-flat reliability curve and zero/near-zero sensitivity at high specificity. This is *not* a bug in our XGBoost wrapper — it is a known property of isotonic regression on small calibration sets.

Two consequences for the model card:

1. **XGBoost should not be used at clinically meaningful operating points (e.g. sens@85% / sens@90%).** Its uncalibrated discrimination is fine; its calibrated probabilities are not.
2. **Sigmoid (Platt) calibration is robust at this slice size.** Both LR and the Ensemble use it — the Ensemble explicitly per [ADR-012](docs/adr/012-honours-baseline-reproduction.md), to avoid replicating the XGBoost failure mode on the same slice size. Niculescu-Mizil & Caruana (2005) shows Platt scaling outperforms isotonic at n < ~1000 calibration rows.

If the deployment context allows for a larger calibration slice (e.g. an Australian-cohort retraining where ≥1000 calibration rows are available), isotonic for XGBoost becomes viable. The current artefact does not.

## 6. Decision-curve analysis at AusCVDRisk thresholds

DCA at 5% and 10% (the AusCVDRisk treatment thresholds, [`07-eval-design.md`](docs/research/07-eval-design.md) §6). Headline: models add net benefit over treat-all in **moderate-prevalence settings (Cleveland, Hungarian)** but not in **very-high-prevalence settings (Switzerland)**, where treat-all dominates because the negative-class base rate is too low for any model's predicted-low-risk subgroup to deliver clinical value vs. simply treating everyone. **LongBeachVA at 10%** is borderline: the Honours-Ensemble is the only model whose net benefit (+0.7178) exceeds treat-all (+0.7167); LR and XGBoost are *worse* than treat-all on this fold; TabICL ties. The gap is well within bootstrap noise.

This is honest information about the *data*, not a verdict on the *models*. Full per-fold DCA values in [`reports/v1/metrics_per_fold.json`](reports/v1/metrics_per_fold.json) under each `dca` block; visualisations under [`reports/v1/figures/`](reports/v1/figures/).

## 7. Limitations & out-of-scope

- **Not validated in a clinical setting.** No clinician-in-the-loop study, no real-EHR integration, no deployment.
- **Trained on small, biased UCI sources.** ~920 rows total across four heterogeneous sources, none of which is contemporary Australian primary-care data. Generalisability to any cohort outside these four sources is *unverified*.
- **LongBeachVA ≥70 stratum is structurally under-served.** Every v1 model loses meaningful AUROC on that stratum. The model card flags this explicitly; any deployed surface must either exclude this stratum or carry a "low-confidence" UI affordance.
- **No external validation cohort.** The LODO protocol simulates a deployment-like scenario (test on a source the model has never seen) but does not replace external validation on a non-UCI cohort.
- **No fairness audit beyond sex + age band.** Race, ethnicity, socioeconomic status, geographic strata: not in the data, not auditable.
- **Calibration slice size limits XGBoost.** See §5.
- **The Honours-Ensemble row is the architecture only, not the WOA-Ensemble pipeline** (see §3 of [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md)). The WOA feature-selection layer that produced the Honours report's headline number is not in the supplied archive; we did not invent one. Reading the Honours-Ensemble row as if it is "WOA-Ensemble under our protocol" is incorrect.

## 8. Honesty caveats specific to the Honours-Ensemble row

The Honours team's Final Report §7.2 Table 2.2 reports WOA-Ensemble on HFP at sensitivity 89.72%, specificity 83.12% (single 80/20 split, no CV, no per-source breakdown, no calibration). Our table reports a different number under a different protocol (4-fold LODO, calibrated, sensitivity at 85% / 90% specificity not at the default 0.5 threshold, no WOA layer). A direct numerical comparison ("89.72% vs our X%") is misleading. The right reading is qualitative: the Honours architecture's relative position against the v1 trio under a fair LODO protocol. Full discussion in [`docs/research/09-honours-vs-v1.md`](docs/research/09-honours-vs-v1.md).

## 9. Reproducibility

- All code: this repo, MIT-licensed.
- Run: `uv sync --project backend && uv run --project backend python backend/scripts/train_v1.py` (full LODO, ~40 min on a recent CPU). `--smoke` for a 1-fold synthetic-data smoke pass (~70s).
- Model artefacts: not in git (per [ADR-010](docs/adr/010-model-artefact-storage.md)); regenerated locally by the script above.
- Reports: in git under [`reports/v1/`](reports/v1/) — `metrics_per_fold.json`, `metrics_aggregate.json`, and `figures/*.png`.
- Determinism: every wrapper pins `random_state` / `torch.manual_seed` to the project seed (20260505). XGBoost reproduces to ~1e-6 across runs; the PyTorch Ensemble reproduces to ~1e-5 (deterministic-algorithms flag deliberately not enabled — see [ADR-012](docs/adr/012-honours-baseline-reproduction.md)).
- CI: smoke run is enforced on every PR (`train-v1-smoke` step in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 10. References

- [TabICL (Inria Soda)](https://github.com/soda-inria/tabicl) — BSD-3-Clause.
- [XGBoost](https://xgboost.readthedocs.io/) — Apache-2.0.
- [scikit-learn](https://scikit-learn.org/) — BSD-3.
- [PyTorch](https://pytorch.org/) — BSD-style.
- Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. (Cited only as the WOA spec the Honours report claims to follow; not implemented in v1.)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. (Calibration recipe rationale.)
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. (DCA methodology.)
- Honours Final Report (2024S1-1698) — the prior study this v1 work supersedes.

Architecture decisions:

- [ADR-006](docs/adr/006-risk-model-architecture.md) — risk-model architecture.
- [ADR-008](docs/adr/008-preprocessing-pipeline.md) — preprocessing pipeline.
- [ADR-009](docs/adr/009-eval-harness.md) — eval harness.
- [ADR-010](docs/adr/010-model-artefact-storage.md) — model artefact storage.
- [ADR-011](docs/adr/011-tfm-tabicl-supersedes-tabpfn.md) — TabICL supersedes TabPFN.
- [ADR-012](docs/adr/012-honours-baseline-reproduction.md) — Honours-baseline reproduction (Phase 2.4).
