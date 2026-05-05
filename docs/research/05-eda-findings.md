# 05 — EDA findings on the UCI Heart Disease subsets (HFP schema)

> **Purpose.** Concrete, opinionated summary of what the [`notebooks/01-eda.ipynb`](../../notebooks/01-eda.ipynb) notebook revealed when run against the four UCI Heart Disease subsets joined into the Heart Failure Prediction (Kaggle, fedesoriano) 11-feature + target schema.
>
> **Scope.** This document is the bridge between Phase 2.1 (data + EDA) and Phase 2.2 (preprocessing pipeline). It does **not** prescribe specific imputation / encoding choices — those are decided in Phase 2.2. It does spell out the data pathologies any v1 pipeline must handle.
>
> **Status.** Findings frozen at 2026-05-05 against the UCI snapshots whose SHA-256 lockfiles are pinned in [`data/checksums/`](../../data/checksums/).

---

## TL;DR

The four UCI subsets concatenate to **920 rows × 11 features + 1 target**. The dataset is much messier than the Kaggle HFP description suggests, and naive use of it would produce a model whose headline metrics are inflated by source membership leaking through missingness patterns.

Three things matter for Phase 2.2:

1. **Switzerland's Cholesterol column is 100% missing** (encoded as `0`). Long Beach VA is 24.5% missing on the same column. Treating `0` as a real measurement, as the prior Honours pipeline did, hands the model a perfect source-membership detector.
2. **Missingness is source-correlated, not random.** `ST_Slope` is missing in 64.6% of Hungarian rows but 0% of Cleveland; `RestingBP`, `MaxHR`, `ExerciseAngina`, and `Oldpeak` are each ≈27% missing in Long Beach VA but ≈0% elsewhere. Imputing under a missing-completely-at-random assumption smuggles source identity into other features.
3. **Heart-disease prevalence ranges from 36.1% (Hungarian) to 93.5% (Switzerland).** Random K-fold CV on the union mixes these case mixes between train and test and inflates discrimination metrics. LODO-CV is the only honest protocol — confirming the [04-revised-design.md §3.5](./04-revised-design.md#35-train--val--test-split) decision empirically.

---

## 1. What's there

### 1.1 Source breakdown

| Source | n rows | % positive | % of total |
|---|---:|---:|---:|
| Cleveland | 303 | 45.9% | 32.9% |
| Hungarian | 294 | 36.1% | 32.0% |
| LongBeachVA | 200 | 74.5% | 21.7% |
| Switzerland | 123 | 93.5% | 13.4% |
| **Total** | **920** | **55.3%** | **100.0%** |

Cleveland is the largest, cleanest, and most-cited subset; Switzerland is the smallest and most pathological in every dimension we measured.

### 1.2 Schema

All 11 HFP features plus the binarised `HeartDisease` target are present after the schema mapping in [`cardiorisk.data.combine`](../../backend/cardiorisk/data/combine.py). The `source` column is added at combine time as the LODO-CV grouping variable; it is **not** a model feature.

### 1.3 Sex distribution (per source)

| Source | F | M | F% |
|---|---:|---:|---:|
| Cleveland | 97 | 206 | 32.0% |
| Hungarian | 81 | 213 | 27.6% |
| LongBeachVA | 6 | 194 | 3.0% |
| Switzerland | 10 | 113 | 8.1% |

The Long Beach VA cohort is essentially male-only. This is consistent with the dataset's origin (a Veterans Affairs hospital) but creates a structural problem for any per-sex fairness audit — the per-sex sample for LongBeachVA-as-test is 6 rows.

### 1.4 Target by sex (overall)

| Sex | n | % positive |
|---|---:|---:|
| F | 194 | 25.8% |
| M | 726 | 63.2% |

The 37-point absolute gap is partly a real epidemiological effect and partly confounded with the fact that the most-pathological cohorts (LongBeachVA, Switzerland) are also the most male-skewed.

### 1.5 Age (per source)

Age distributions overlap heavily across sources (means 47.8 to 59.4, all SDs in the 7.8–9.0 range). Age is not a strong source-distinguishing feature on its own.

---

## 2. What's broken

### 2.1 Cholesterol-as-zero is missingness, not a measurement

| Source | rows | `Cholesterol == 0` | % |
|---|---:|---:|---:|
| Cleveland | 303 | 0 | 0.0% |
| Hungarian | 294 | 0 | 0.0% |
| LongBeachVA | 200 | 49 | 24.5% |
| Switzerland | 123 | 123 | **100.0%** |

The Honours pipeline took these `0` values at face value, which means:

- A naive model can perfectly identify any Switzerland row from `Cholesterol == 0` alone.
- The `Cholesterol` feature's apparent informativeness is partly a source-membership detector dressed up as a clinical signal.
- Anything based on group statistics for `Cholesterol` (mean, SD, correlation with target) computed without the cleaning step is wrong by a wide margin.

**Phase 2.2 must convert `Cholesterol == 0` to `NaN` before any imputation runs.** This is non-negotiable per [04-revised-design.md §3.2](./04-revised-design.md#32-cleaning).

### 2.2 Missingness is highly source-dependent

Per-feature missingness rate (%), by source, after dropping `HeartDisease`:

| Feature | Cleveland | Hungarian | LongBeachVA | Switzerland | Overall |
|---|---:|---:|---:|---:|---:|
| Age | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sex | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ChestPainType | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RestingBP | 0.0 | 0.3 | **28.0** | 1.6 | 6.4 |
| Cholesterol | 0.0 | 7.8 | 3.5 | **0.0**¹ | 3.3 |
| FastingBS | 0.0 | 2.7 | 3.5 | **61.0** | 9.8 |
| RestingECG | 0.0 | 0.3 | 0.0 | 0.8 | 0.2 |
| MaxHR | 0.0 | 0.3 | **26.5** | 0.8 | 6.0 |
| ExerciseAngina | 0.0 | 0.3 | **26.5** | 0.8 | 6.0 |
| Oldpeak | 0.0 | 0.0 | **28.0** | 4.9 | 6.7 |
| ST_Slope | 0.0 | **64.6** | **51.0** | 13.8 | **33.6** |

¹ Switzerland Cholesterol is 100% encoded as `0`; that count appears under §2.1, not in this table — `pd.isna` returns `False` for `0`.

Two patterns dominate:

- **The Long Beach VA exam protocol skipped or under-reported four exercise-ECG-derived measurements** (`RestingBP`, `MaxHR`, `ExerciseAngina`, `Oldpeak`). All four are missing in ≈27% of rows, and the missingness almost certainly correlates with whether the patient could complete the exercise stress test — itself a strong predictor of cardiac status. This is **informative missingness**.
- **`ST_Slope` is mostly missing outside Cleveland.** With 33.6% overall missingness — 64.6% in Hungarian, 51.0% in LongBeachVA — any imputation strategy on this column has to be reported as part of the model, not as a preprocessing detail.

A mean / mode imputer with no source awareness would force these patterns into the feature distributions, producing a model whose generalisation depends on whichever source happens to contribute most to a given fold. **Imputers must be fit within each LODO fold's training slice only**, per [04-revised-design.md §3.3](./04-revised-design.md#33-imputation).

### 2.3 Class prevalence varies by 57 percentage points across sources

| Source | % positive | Implications |
|---|---:|---|
| Cleveland | 45.9% | balanced enough; standard binary classification |
| Hungarian | 36.1% | mildly imbalanced; AUPRC matters |
| LongBeachVA | 74.5% | strongly imbalanced toward "diseased"; default 0.5 threshold over-predicts disease |
| Switzerland | 93.5% | almost everyone is sick; near-degenerate |

When a single random K-fold mixes Switzerland-rows-mostly-positive with Cleveland-rows-near-balanced in the same training fold, the model learns a "this looks like Switzerland → predict 1" shortcut. The reverse — "this looks like Cleveland → predict 0" — is also learnable. Both shortcuts evaporate under LODO-CV, which is the point of LODO-CV.

### 2.4 Sex imbalance, particularly in LongBeachVA

With six female patients in the LongBeachVA subset, any per-sex Phase-2.5 fairness audit on the LongBeachVA-as-test fold will have wide CIs by construction. This is a real limitation we publish in the Model Card; it is not "fixable" without re-collecting data.

---

## 3. Implications for Phase 2.2 and beyond

### 3.1 Phase 2.2 (preprocessing pipeline) must:

1. Convert `Cholesterol == 0 → NaN` as the first cleaning step. (Highest priority finding.)
2. Add a `<col>_was_missing` indicator column for every feature whose per-source missingness exceeds 10% in any source — the missingness *is itself* signal, and a missingness-aware downstream model gets to learn from it.
3. Use **MissForest** or equivalent fitted within each LODO fold's training slice, never on the union or on the test fold. (Same prescription as the design doc.)
4. For categorical features (`ChestPainType`, `RestingECG`, `ST_Slope`), reserve a `Missing` category instead of imputing toward the mode.
5. Document the imputation choice for each feature explicitly so the Model Card can list it.

### 3.2 Phase 2.3 (model training) must:

1. Use LODO-CV as the headline split, with random K-fold reported only as a "look how badly this inflates numbers" comparison.
2. Report per-source AUROC, AUPRC, Brier, and calibration slope/intercept — the per-source breakdown is the point, not an addendum.
3. Calibrate models per LODO fold against a held-out within-fold calibration slice (per the [04-revised-design.md §3.5](./04-revised-design.md#35-train--val--test-split) recipe).

### 3.3 Phase 2.5 (fairness audit) must:

1. Stratify all metrics by sex *and* by source, and report the per-cell sample size next to each metric so the audience can see which CIs are wide.
2. State the LongBeachVA-female n=6 limitation explicitly in the Model Card.
3. Avoid claims about per-sex generalisation in any cohort with fewer than ~30 rows of either sex.

### 3.4 Things explicitly out of scope for Phase 2.2

- **Re-collecting data** to balance sex or source distribution. Out of scope; we use HFP / UCI as published.
- **Synthesising additional rows** for under-represented strata. Adds capability without adding evidence; not on the Phase 2 path.
- **Dropping the Switzerland or Long Beach VA subsets** to make the data look cleaner. The whole point of including them is to stress-test generalisation.

---

## 4. Provenance + reproducibility

- All numbers in this document were produced by running [`notebooks/01-eda.py`](../../notebooks/01-eda.py) (jupytext-paired with `01-eda.ipynb`) against the parquet at `data/processed/combined.parquet`, itself produced from the UCI files whose SHA-256s are pinned in [`data/checksums/`](../../data/checksums/).
- The combine logic lives in [`cardiorisk.data.combine`](../../backend/cardiorisk/data/combine.py); the schema mapping is asserted by tests in [`backend/tests/test_combine.py`](../../backend/tests/test_combine.py).
- Re-running the EDA notebook against any new SHA-256 of the source files will regenerate the figures with current data; if any number in this document drifts, the document is the wrong copy and gets updated, not the data.
