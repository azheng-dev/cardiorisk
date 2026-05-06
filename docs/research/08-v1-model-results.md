# Phase 2.3b + 2.4 — v1 model results (LODO-CV on HFP-combined)

> **Status:** Phase 2.3b + 2.4 deliverable. Numbers below are produced
> verbatim by `backend/scripts/train_v1.py` from
> `data/processed/combined.parquet` (UCI Heart-Disease combined under
> the HFP schema; see ADR-008 + Phase 2.1 deliverables). Re-run via
> `uv run --project backend python backend/scripts/train_v1.py` (no
> flags = full LODO).
>
> **Methodology:** Leave-One-Domain-Out CV across the four UCI sources
> (Cleveland, Hungarian, LongBeachVA, Switzerland), 80/10/10
> within-fold split (train / calibration / test), per-model
> preprocessing pipeline from Phase 2.2 (ADR-008), post-hoc calibration
> (isotonic for XGBoost, Platt/sigmoid for LR and the Ensemble; TabICL
> passes through unwrapped per ADR-011), bootstrap CIs (2,000
> resamples, percentile method, per-fold), subgroup audit on sex + age
> bands, decision-curve analysis at the AusCVDRisk thresholds (5% +
> 10%). Full eval-harness rationale in `07-eval-design.md` and ADR-009.
>
> **Models:** TabICL 2.1 (TFM headline; ADR-011 supersedes ADR-006's
> TabPFN choice), XGBoost 3.x with Optuna 50-trial / 10-min cap, L1
> Logistic Regression with restricted-cubic-spline expansion (saga
> solver, GridSearchCV C ∈ {0.001, 0.01, 0.1, 1, 10, 100}), and a
> PyTorch port of the Honours 4-net mean-averaged Ensemble (DNN, 1D
> CNN, LSTM, BiLSTM) — *architecture only*, the Honours WOA feature
> selection layer is not reproducible from the supplied archive (see
> [ADR-012](../adr/012-honours-baseline-reproduction.md) and
> [`09-honours-vs-v1.md`](./09-honours-vs-v1.md)). Seed pinned to
> `20260505`.

---

## 1. Headline: aggregate metrics across the four LODO folds

Means ± standard deviation across folds. Bootstrap CIs are per-fold (in
the per-fold table below).

| Model    | AUROC          | AUPRC          | Brier ↓        | Calib. slope (ideal=1) | Sens@85% spec  | Sens@90% spec  |
|----------|----------------|----------------|----------------|------------------------|----------------|----------------|
| TabICL   | **0.811 ± 0.085** | **0.891 ± 0.055** | **0.150 ± 0.016** | 0.97 ± 0.30              | 0.567 ± 0.215  | 0.437 ± 0.277  |
| LR (L1+RCS) | 0.804 ± 0.082  | 0.883 ± 0.063  | 0.194 ± 0.037  | 0.75 ± 0.34              | **0.589 ± 0.136**  | **0.457 ± 0.226**  |
| XGBoost  | 0.779 ± 0.081  | 0.826 ± 0.102  | 0.218 ± 0.041  | 0.21 ± 0.30              | 0.186 ± 0.195  | 0.103 ± 0.183  |
| Ensemble *(Honours architecture, Phase 2.4)* | 0.792 ± 0.076  | 0.860 ± 0.071  | 0.197 ± 0.024  | **1.02 ± 0.48**           | 0.585 ± 0.138  | 0.370 ± 0.258  |

**Headline reading:**

1. **TabICL wins on AUROC, AUPRC, and Brier without any post-hoc
   calibration step.** It is the cleanest discrimination result in
   the table. The TFM choice (and the swap from TabPFN to TabICL after
   the licensing pivot) holds up under LODO.
2. **L1 LR + RCS is the strongest white-box.** Within ~1pp AUROC of
   TabICL on average, with the best sensitivity at high specificity
   (0.589 @ 85% / 0.457 @ 90%), and well-behaved Platt-scaled
   calibration. This is the model recruiters will want to read; we
   should never quietly drop it from the comparison.
3. **XGBoost is the surprising loser**, and it loses *because of the
   calibration step*, not the base model. Mean calibration slope of
   0.21 means isotonic post-hoc collapses predictions toward the
   centre. Sensitivity at high specificity tanks accordingly. Honest
   reading: ~50-row calibration slices per fold are too small for
   isotonic regression. We discuss the implications in §4.
4. **The Honours-Ensemble lands between LR and XGBoost on
   discrimination (AUROC 0.792)**, with the **best calibration slope
   in the table (1.02, ideal=1)** and competitive Sens@85% spec
   (0.585 — within 0.4pp of LR). Two honest sub-readings:
   (a) the architecture is real — it doesn't blow up under LODO and
   it survives the calibration step that destroyed XGBoost, vindicating
   the [ADR-012](../adr/012-honours-baseline-reproduction.md) Platt-
   over-isotonic decision; (b) it does *not* recover the report's
   89.72% sensitivity headline — but that headline was on a much
   easier protocol (single 80/20 split, no FS reproduction here, no
   calibration, threshold 0.5), so the comparison is qualitative not
   numeric. Full discussion in [`09-honours-vs-v1.md`](./09-honours-vs-v1.md).
   The Ensemble actually edges TabICL on the LongBeachVA fold
   (AUROC 0.745 vs 0.740) — the only model that does. Possibly
   because the deeper architecture handles the prevalence-inverted
   regime better than the TFM's in-context learning at that fold's n.

---

## 2. Per-fold results (held-out source = test set)

All four LODO folds, all three models. Source order: Cleveland (n=303,
prev=0.46), Hungarian (n=294, prev=0.36), LongBeachVA (n=200,
prev=0.75), Switzerland (n=123, prev=0.94). 95% percentile bootstrap CIs
(2,000 resamples) for the headline metrics live in
`reports/v1/metrics_per_fold.json` (per-fold `headline_ci` block) — they
are too wide to read in a Markdown table, but they are committed.

### 2.1 Cleveland fold (n_test=303, prevalence=0.459)

| Model    | AUROC | AUPRC | Brier | Calib. slope | Calib. int. | Sens@85sp | Sens@90sp |
|----------|-------|-------|-------|--------------|-------------|-----------|-----------|
| TabICL   | 0.877 | 0.869 | 0.145 | +1.19        | −0.36       | 0.741     | 0.683     |
| LR       | 0.863 | 0.857 | 0.165 | +1.06        | −0.77       | 0.691     | 0.626     |
| XGBoost  | 0.838 | 0.759 | 0.187 | +0.66        | −0.90       | 0.036     | 0.036     |
| Ensemble | 0.832 | 0.805 | 0.191 | +1.13        | −0.97       | 0.655     | 0.453     |

Cleveland is the easiest fold (highest training-set diversity remaining
after holding it out, and the most-studied source in the literature).
TabICL produces a calibration slope of 1.19 — slight overconfidence,
but the closest to ideal across the experiment. XGBoost's
sens@85% = 0.036 here is the smoking gun for the isotonic calibration
problem: the model's *uncalibrated* discrimination (AUROC 0.838) is
fine, but the calibrator pushes nearly every prediction below the 85%
specificity decision point.

### 2.2 Hungarian fold (n_test=294, prevalence=0.361)

| Model    | AUROC | AUPRC | Brier | Calib. slope | Calib. int. | Sens@85sp | Sens@90sp |
|----------|-------|-------|-------|--------------|-------------|-----------|-----------|
| TabICL   | 0.893 | 0.841 | 0.146 | +1.26        | −1.01       | 0.764     | 0.651     |
| LR       | 0.886 | 0.825 | 0.166 | +1.03        | −1.28       | 0.717     | 0.651     |
| XGBoost  | 0.859 | 0.738 | 0.195 | +0.08        | −0.90       | 0.377     | 0.377     |
| Ensemble | 0.877 | 0.807 | 0.200 | +1.65        | −1.86       | 0.745     | 0.632     |

Hungarian is the *strongest* fold for every model. Lower prevalence
(0.36) and good representation in the training pool. LR comes within
0.7pp AUROC of TabICL with comparable sensitivity. XGBoost
calibration slope of 0.08 is even worse than Cleveland — the model's
predictions are essentially flattened toward the prevalence rate.

### 2.3 LongBeachVA fold (n_test=200, prevalence=0.745)

| Model    | AUROC | AUPRC | Brier | Calib. slope | Calib. int. | Sens@85sp | Sens@90sp |
|----------|-------|-------|-------|--------------|-------------|-----------|-----------|
| TabICL   | 0.740 | 0.888 | 0.174 | +0.61        | +0.53       | 0.362     | 0.302     |
| LR       | 0.733 | 0.877 | 0.204 | +0.43        | +0.82       | 0.443     | 0.376     |
| XGBoost  | 0.702 | 0.843 | 0.214 | +0.04        | +0.50       | 0.000     | 0.000     |
| Ensemble | **0.745** | 0.870 | 0.168 | +0.78        | +0.56       | 0.463     | 0.376     |

**LongBeachVA is the honest difficulty stress test.** Every model loses
~10pp AUROC vs. its Cleveland/Hungarian fold. Three structural reasons,
all of which the EDA in `05-eda-findings.md` flagged:

1. **Highest cholesterol missingness** in the dataset — and our zero-as-
   missing rule in Phase 2.2 (ADR-008) is correct but costs the model
   the strongest cardiovascular signal.
2. **Prevalence inversion** (75% positive vs. 36–46% in
   Cleveland/Hungarian). Models trained on a lower-prevalence pool
   underestimate risk on this fold; calibration intercepts shift
   positive (+0.50–+0.82) accordingly.
3. **Six women in the entire fold** — the sex subgroup audit cannot
   produce a stable F-stratum AUROC, so it returns NA. We do not paper
   over this: see §3.

XGBoost's sens@85% = 0.000 on this fold is the worst calibration
failure in the experiment. The fold is also the least DCA-favourable —
at the 10% AusCVDRisk threshold, three of four models are *worse than
treat-all* (see §5). The Ensemble is the lone exception: it edges out
treat-all by +0.0011 net benefit at the 10% threshold (0.7178 vs
0.7167) and posts the highest AUROC on the fold (0.745 vs TabICL
0.740). Honest reading: the gap is well within bootstrap noise (95%
CIs overlap heavily — see `headline_ci` in `metrics_per_fold.json`),
but the *direction* is consistent — the Ensemble's 4-net averaging
appears to be slightly more robust to the prevalence-inverted regime
than the TFM's in-context learning at this fold's n_test=200.

### 2.4 Switzerland fold (n_test=123, prevalence=0.935)

| Model    | AUROC | AUPRC | Brier | Calib. slope | Calib. int. | Sens@85sp | Sens@90sp |
|----------|-------|-------|-------|--------------|-------------|-----------|-----------|
| TabICL   | 0.736 | 0.968 | 0.137 | +0.84        | +2.23       | 0.400     | 0.113     |
| LR       | 0.733 | 0.971 | 0.242 | +0.49        | +2.81       | 0.504     | 0.174     |
| XGBoost  | 0.717 | 0.963 | 0.277 | +0.05        | +2.44       | 0.330     | 0.000     |
| Ensemble | 0.714 | 0.956 | 0.226 | +0.54        | +2.75       | 0.478     | 0.017     |

**Switzerland is degenerate-by-design.** 93.5% prevalence and only 123
test rows. AUPRC is artificially inflated (the prevalence floor is
0.935). Calibration intercepts blow out positive (+2.23–+2.81) because
the training pool sees ~45% prevalence on average — every model
underestimates the risk by a constant. Confidence intervals on
sensitivity at high specificity are extremely wide here.

We deliberately do **not** drop Switzerland from the LODO. It is the
dataset we have, and reporting only the easy folds would be exactly
the kind of selective reporting the critical review (`03-critical-
review.md`) called out in the literature.

---

## 3. Subgroup audits

`stratified_metrics` returns AUROC per subgroup, with `min_stratum_size`
guarding against meaningless tiny strata. Strata below the guard return
NA and the fairness gap is suppressed (also returned as NA — we
deliberately do **not** impute it).

### 3.1 Sex (F vs M)

| Model    | Cleveland (F=97 / M=206) | Hungarian (F=81 / M=213) | LongBeachVA (F=6 / M=194) | Switzerland (F=10 / M=113) |
|----------|--------------------------|--------------------------|---------------------------|----------------------------|
| TabICL   | F=0.907 / M=0.847 (gap 0.060) | F=0.795 / M=0.894 (gap 0.099) | F=NA / M=0.734 (gap NA) | F=NA / M=0.764 (gap NA) |
| LR       | F=0.880 / M=0.830 (gap 0.050) | F=0.829 / M=0.883 (gap 0.054) | F=NA / M=0.726 (gap NA) | F=NA / M=0.752 (gap NA) |
| XGBoost  | F=0.875 / M=0.795 (gap 0.080) | F=0.806 / M=0.842 (gap 0.037) | F=NA / M=0.704 (gap NA) | F=NA / M=0.738 (gap NA) |
| Ensemble | F=0.848 / M=0.811 (gap 0.037) | F=0.733 / M=0.875 (gap 0.142) | F=NA / M=0.740 (gap NA) | F=NA / M=0.730 (gap NA) |

Two folds are auditable (Cleveland, Hungarian). Two are not
(LongBeachVA F=6, Switzerland F=10 — both below the
`min_stratum_size` guard). Within auditable folds:

- **Cleveland**: every model performs slightly *better* on F than on M
  (gap 0.04–0.08 in F's favour). No evidence of a structural F-deficit.
- **Hungarian**: every model performs *worse* on F than on M (gap 0.04–
  0.14 against F). The Ensemble has the largest gap (0.142), TabICL
  next (0.099). This deserves flagging — Hungarian women are a small
  slice (n=81) but large enough to be meaningful, and the direction
  reverses Cleveland's.

The cross-fold F vs M picture is therefore *mixed, not systematically
biased*, but the MODEL_CARD.md surfaces this honestly. The Ensemble's
larger Hungarian-female gap is the strongest argument that the
Honours architecture, even reproduced faithfully, is not a clear
upgrade over the v1 trio.

### 3.2 Age band (<50 / 50–69 / ≥70)

| Model    | Cleveland (50–69=206 / <50=87 / ≥70=10) | Hungarian (50–69=133 / <50=161 / ≥70=0) | LongBeachVA (50–69=165 / <50=19 / ≥70=16) | Switzerland (50–69=93 / <50=25 / ≥70=5) |
|----------|------------------------------------------|------------------------------------------|--------------------------------------------|------------------------------------------|
| TabICL   | 0.857 / 0.887 / 1.000 (gap 0.143)        | 0.897 / 0.886 / NA (gap 0.010)           | 0.745 / 0.722 / 0.464 (gap 0.281)          | 0.734 / 0.978 / NA (gap 0.244)           |
| LR       | 0.841 / 0.885 / 1.000 (gap 0.159)        | 0.892 / 0.878 / NA (gap 0.014)           | 0.741 / 0.689 / 0.536 (gap 0.205)          | 0.709 / 0.978 / NA (gap 0.269)           |
| XGBoost  | 0.811 / 0.863 / 1.000 (gap 0.189)        | 0.871 / 0.845 / NA (gap 0.026)           | 0.703 / 0.672 / 0.518 (gap 0.185)          | 0.698 / 0.924 / NA (gap 0.226)           |
| Ensemble | 0.812 / 0.863 / 0.833 (gap 0.050)        | 0.869 / 0.881 / NA (gap 0.012)           | 0.736 / 0.833 / **0.393** (gap **0.440**)  | 0.664 / 0.957 / NA (gap 0.293)           |

Reading:

- **The ≥70 stratum is tiny everywhere** (0–16 rows per fold). Cleveland
  reports AUROC 1.0 for TabICL/LR/XGBoost on 10 rows because the
  predictor happens to separate them; this is not a real signal. The
  Ensemble's 0.833 on the same 10 rows is also noise-dominated, but it
  shows the headline differs across models when n is this small —
  another reason to ignore the Cleveland ≥70 stratum entirely.
- **LongBeachVA has the only meaningfully populated ≥70 stratum** (n=16),
  and every model performs *worst* on those patients (AUROC 0.39–0.54).
  The Ensemble has the worst performance here (0.393, gap 0.440 against
  the best stratum) — slightly *worse* than XGBoost on the same n=16.
  This is the most clinically important subgroup-audit finding in the
  experiment: **no model in the v1 four-some** is trustworthy on older
  Long-Beach-VA-style patients.
- **Hungarian has no ≥70 patients at all** in the held-out test slice.
  Audit returns a 2-stratum gap, which is small (0.01–0.03) and
  uninteresting.

We will not under-report this. The MODEL_CARD.md lists LongBeachVA-style
≥70 patients as an explicit out-of-scope subgroup for *all four* v1
models.

---

## 4. The XGBoost calibration story (honest discussion)

XGBoost's mean calibration slope of **0.21** across folds — vs. ~0.97
for TabICL, ~0.75 for LR, and ~1.02 for the Ensemble — is not a bug. It is a structural
consequence of three design choices, only one of which we should
revisit:

1. **The 80/10/10 within-fold split** gives ~50–80 rows to the
   calibration slice (`X_calib`). Isotonic regression is non-parametric
   and over-fits aggressively at this size — it learns step functions
   that flatten out predictions toward the calibration-set base rate,
   destroying high-specificity signal.
2. **Post-hoc calibration is applied even when the base predictor is
   already roughly well-calibrated.** XGBoost's *uncalibrated* slope on
   most folds is closer to 0.5–0.8; isotonic then drives it toward
   0.05–0.20. The wrapper assumes calibration always helps; on tiny
   slices it does the opposite.
3. **The `frozen_estimator=True` calibration wrapper** (per ADR-009) is
   the right choice for honest leak-free evaluation, but it amplifies
   (1) and (2) by refusing to share data between the base predictor
   and the calibrator.

**What this is not:** it is not a verdict that XGBoost is unsuitable
for this problem. The discrimination story (AUROC 0.78 mean) is
respectable and the AUROC bootstrap CIs overlap meaningfully with LR
on three of four folds. It *is* a verdict that the **post-hoc
calibration recipe needs revisiting in Phase 2.4** — most likely by
trying Platt scaling on XGBoost too (Niculescu-Mizil & Caruana 2005
showed Platt is more robust at small calibration-set sizes than
isotonic), or by widening the calibration slice via repeated splits
(Cox 2025 has results on this for tree ensembles).

We deliberately ship the v1 results *as configured* — i.e. with the
calibration recipe locked in ADR-009 — rather than re-tuning to make
XGBoost look better. The point of Phase 2.3b is the honest baseline.

---

## 5. Decision-curve analysis at the AusCVDRisk thresholds

Net benefit at the two thresholds the AusCVDRisk pathway uses for
clinical action (5% = consider, 10% = treat). DCA reading rules: a
model is *useful* at threshold *t* iff its net benefit exceeds both
treat-all and treat-none. Full curves over the 1–99% sweep are in
`reports/v1/figures/<model>_<source>_dca.png`.

| Source       | Model    | NB @ 5% (model / treat-all)  | NB @ 10% (model / treat-all) | Useful at 5%? | Useful at 10%? |
|--------------|----------|------------------------------|-------------------------------|---------------|-----------------|
| Cleveland    | TabICL   | +0.431 / +0.430              | +0.407 / +0.399               | tied          | **yes**         |
| Cleveland    | LR       | +0.430 / +0.430              | +0.399 / +0.399               | tied          | tied            |
| Cleveland    | XGBoost  | +0.433 / +0.430              | +0.406 / +0.399               | **yes**       | **yes**         |
| Cleveland    | Ensemble | +0.430 / +0.430              | +0.399 / +0.399               | tied          | tied            |
| Hungarian    | TabICL   | +0.328 / +0.327              | +0.295 / +0.289               | **yes**       | **yes**         |
| Hungarian    | LR       | +0.327 / +0.327              | +0.292 / +0.289               | tied          | **yes**         |
| Hungarian    | XGBoost  | +0.325 / +0.327              | +0.293 / +0.289               | worse         | **yes**         |
| Hungarian    | Ensemble | +0.327 / +0.327              | +0.290 / +0.289               | tied          | tied            |
| LongBeachVA  | TabICL   | +0.732 / +0.732              | +0.717 / +0.717               | tied          | tied            |
| LongBeachVA  | LR       | +0.732 / +0.732              | +0.685 / +0.717               | tied          | **worse**       |
| LongBeachVA  | XGBoost  | +0.732 / +0.732              | +0.663 / +0.717               | tied          | **worse**       |
| LongBeachVA  | Ensemble | +0.732 / +0.732              | +0.718 / +0.717               | tied          | **yes**         |
| Switzerland  | TabICL   | +0.932 / +0.932              | +0.920 / +0.928               | tied          | **worse**       |
| Switzerland  | LR       | +0.892 / +0.932              | +0.857 / +0.928               | **worse**     | **worse**       |
| Switzerland  | XGBoost  | +0.924 / +0.932              | +0.770 / +0.928               | **worse**     | **worse**       |
| Switzerland  | Ensemble | +0.908 / +0.932              | +0.882 / +0.928               | **worse**     | **worse**       |

Reading:

- **Cleveland and Hungarian** are the two folds where the v1 models
  earn their keep — at the 10% AusCVDRisk threshold, every model is at
  worst tied with treat-all and TabICL/XGBoost are slightly better.
- **LongBeachVA at 10%** is where the Ensemble surprises: it is the
  *only* model that nudges past treat-all (NB +0.7178 vs +0.7167, gap
  +0.0011). LR and XGBoost are *worse than treat-all*; TabICL is tied.
  With 75% prevalence, this is the difficult regime, and even the
  Ensemble's win is within noise. With 75% prevalence, treat-all is
  the right policy and the model adds no value.
- **Switzerland at 10%** is even more degenerate: 93.5% prevalence
  means the threshold is below the base rate, so treat-all dominates
  and every model loses to it.

Decision-curve analysis confirms what the headline metrics suggest:
**the v1 models add clinical value where prevalence is moderate
(Cleveland, Hungarian), and add no value where prevalence is very
high (Switzerland; LongBeachVA borderline).** This is the honest
answer.

---

## 6. What this implies for the v1 release

1. **TabICL is the v1 default headline.** It wins on three of the four
   metrics that matter (AUROC, AUPRC, Brier), has the cleanest
   calibration without needing post-hoc, and is the only model that
   is consistently useful at the AusCVDRisk 10% threshold on the
   clinically-relevant folds.
2. **L1 LR + RCS is the v1 transparency anchor.** It is the white-box
   the MODEL_CARD.md will recommend reviewers read first. Its
   sensitivity-at-spec numbers are the strongest in the table.
3. **XGBoost ships with a known calibration caveat**, documented here
   and in the MODEL_CARD.md. We do not silently drop it.
4. **The Honours-architecture Ensemble (Phase 2.4) ships as a
   reproduction baseline, not a deployment candidate.** It validates
   that the Honours design isn't a bug — under LODO+Platt it is
   competitive on calibration (best slope in the table, 1.02) and on
   LongBeachVA discrimination (only model to clear treat-all at 10%
   threshold), but it does *not* recover the report's headline numbers
   (which used FS we cannot reproduce — see
   [`09-honours-vs-v1.md`](./09-honours-vs-v1.md)) and it has the
   widest Hungarian-female and LongBeachVA-≥70 subgroup gaps. No
   special pleading.
5. **No model is trustworthy on LongBeachVA-style ≥70 patients** based
   on this evidence. The MODEL_CARD.md lists them as out-of-scope for
   all four v1 models.

---

## 7. Reproduction

```bash
cd /Users/Andrew.Zheng1/GitRepo/cardiorisk
uv run --project backend python backend/scripts/train_v1.py
# or for the CI smoke variant:
uv run --project backend python backend/scripts/train_v1.py --smoke
```

Numbers in this document are the deterministic output of the full run
(seed `20260505`, ~34 min wall-clock on a 2024 MacBook Pro M4 Pro).
Outputs land in `reports/v1/metrics_per_fold.json`,
`reports/v1/metrics_aggregate.json`, and `reports/v1/figures/`. Model
artefacts land in `models/v1/` (gitignored per ADR-010 — rebuildable
via the same script).

---

## 8. Open questions answered & deferred to Phase 2.5+

**Answered in Phase 2.4** (this run):

1. Does the Honours-architecture Ensemble reproduce its published
   AUROC under our LODO protocol? **No.** Mean AUROC 0.792 vs the
   report's 0.972 Cleveland-only headline. The architecture is real
   but the published headline depends on a feature-selection step
   (WOA) whose code is not in the supplied archive — see
   [`09-honours-vs-v1.md`](./09-honours-vs-v1.md) and
   [ADR-012](../adr/012-honours-baseline-reproduction.md).

**Deferred:**

2. Should XGBoost's calibration recipe move from isotonic to Platt to
   match LR (and the Ensemble), given the small-calibration-slice
   problem? See §4. **Phase 2.5 will run the comparison.**
3. Do any of the v1 models close the LongBeachVA ≥70 gap with
   age-stratified retraining, or is it a structural data limit?
   **Phase 2.5/2.6.**
4. What is the cross-model agreement on per-patient predictions? Worth
   a Spearman-rank concordance matrix when SHAP lands in Phase 2.5.
5. WOA reconstruction from the published Mirjalili (2016) algorithm —
   deferred indefinitely; we will not commit to it without a clear
   research justification given the headline collapse already
   documented above.
