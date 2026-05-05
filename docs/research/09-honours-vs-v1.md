# 09 — Honours vs v1: cross-model honesty comparison

> **Status:** Phase 2.4 deliverable. Companion to [`08-v1-model-results.md`](./08-v1-model-results.md). Numbers below are produced verbatim by `backend/scripts/train_v1.py` from `data/processed/combined.parquet` under the same LODO protocol the v1 trio uses.
>
> **Purpose.** Three things, in order of importance:
>
> 1. Document plainly that the supplied Honours archive does **not** contain a working WOA implementation, so the report's headline number cannot be reproduced byte-for-byte.
> 2. Reproduce the Honours team's *architecture* (the 4-net mean-averaged Ensemble) under our LODO protocol so the comparison against TabICL/XGBoost/LR is apples-to-apples.
> 3. State honestly where the Honours-Ensemble lands relative to v1 and relative to the report's published headline.
>
> **Methodology** is identical to Phase 2.3b ([`08-v1-model-results.md`](./08-v1-model-results.md)) — same LODO splits, same within-fold 80/10/10, same metrics, same bootstrap CIs, same subgroup audit, same DCA. Calibration: sigmoid (Platt), per [ADR-012](../adr/012-honours-baseline-reproduction.md). Implementation in `cardiorisk.models.ensemble`.

---

## 1. The finding: WOA implementation is not in the supplied archive

Pre-implementation reading of every notebook in `_honours-archive/FIT4701-4702-2024S1-1698/Demos/` (4 notebooks, ~150 cells) and the Final Report PDF (`Assessments/2024S1-1698 Final Report.pdf`) revealed that the WOA feature-selection layer that produces the report's headline number — WOA-Ensemble: sensitivity 89.72%, specificity 83.12% on HFP, Final Report §7.2 Table 2.2 — **does not exist as code anywhere in the supplied archive**.

In `Demos/Data_Pre-processing.ipynb`, the cells immediately below the relevant section headers are all empty `pass`-equivalent placeholders:

| Cell | Content | Code? |
|---|---|---|
| 36 (md) | `RFE_CV` | — |
| 37 (code) | _empty placeholder_ | no |
| 38 (md) | `GWO` | — |
| 39 (code) | _empty placeholder_ | no |
| 40 (md) | `WOA` | — |
| 41 (code) | _empty placeholder_ | **no** |
| 42 (md) | `ACO` | — |
| 43 (code) | _empty placeholder_ | no |
| 44 (md) | `Firefly Algorithm` | — |
| 45 (code) | _empty placeholder_ | no |
| 46 (md) | `Snake Optimisation` | — |
| 47 (code) | _empty placeholder_ | no |
| 48 (md) | `Bat Optimisation` | — |
| 49 (code) | _empty placeholder_ | no |

**What is in the archive:** the Ensemble architecture (DNN + 1D CNN + LSTM + BiLSTM, mean-averaged) is fully implemented in `Data_Pre-processing.ipynb` cell 55 and shared across all four Demo notebooks. Working feature-selection code: GA, EAGA, RF, RFE only. The WOCLSA architecture (CNN + LSTM + ANN) appears as a fixed-hyperparameter `cnnlstma()` function in cell 51 — without the WOA hyperparameter-tuning layer the report claims wraps it.

**Inferred history.** The WOA / GWO / CS / BA / FA / HHO / RFE-CV code was likely developed in a separate Colab notebook by a different team member and was not preserved into the handover archive. The headline number in the report therefore corresponds to code that has been lost. None of this implies the original numbers were fabricated — only that we cannot reproduce them from the archive that was supplied to us.

This finding triggered the Phase 2.4 plan reset captured in [ADR-012](../adr/012-honours-baseline-reproduction.md). Three reproduction paths were considered (faithful Ensemble-only port; best-effort WOA reconstruction; both); the chosen path is faithful Ensemble-only (Path A). The honesty-preserving framing of that decision is what this document is about.

## 2. What we *can* reproduce

The full Ensemble architecture from `Data_Pre-processing.ipynb` cell 55:

- **DNN**: `Linear(n→100) → ReLU → Dropout(0.2) → Linear(100→64) → ReLU → Dropout(0.2) → Linear(64→128) → ReLU → Dropout(0.2) → Linear(128→1)`.
- **1D CNN**: `Conv1d(1→64, k=3) → MaxPool1d(2) → Dropout(0.2) → Flatten → Linear(...→128) → Linear(128→64) → Dropout(0.2) → Linear(64→1)`.
- **LSTM**: `LSTM(input_size=1, hidden=128, batch_first=True) → take last step → Dropout(0.2) → Linear(128→1)`.
- **BiLSTM**: same as LSTM but `bidirectional=True` → `Linear(256→1)`.
- **Inference**: mean of the four sigmoid outputs (no learned meta-model).
- **Training**: Adam(lr=1e-3), batch size 32, 100 epochs each, BCEWithLogitsLoss.

We port to PyTorch (the repo already pulls torch in via TabICL); architectural equivalence is preserved with two documented departures (Glorot vs Kaiming initialisation; no `recurrent_dropout` because PyTorch `nn.LSTM` has no equivalent). Full sub-decisions in [ADR-012](../adr/012-honours-baseline-reproduction.md). Wrapper code in [`cardiorisk/models/ensemble.py`](../../backend/cardiorisk/models/ensemble.py).

## 3. Cross-model comparison (LODO)

The headline aggregate table from [`08-v1-model-results.md`](./08-v1-model-results.md) §1, with the Honours-Ensemble row populated from the Phase 2.4 run:

| Model | AUROC | AUPRC | Brier ↓ | Calib. slope (ideal=1) | Sens@85% spec | Sens@90% spec |
|---|---|---|---|---|---|---|
| TabICL | **0.811 ± 0.085** | **0.891 ± 0.055** | **0.150 ± 0.016** | 0.97 ± 0.30 | 0.567 ± 0.215 | 0.437 ± 0.277 |
| LR (L1+RCS) | 0.804 ± 0.082 | 0.883 ± 0.063 | 0.194 ± 0.037 | 0.75 ± 0.34 | **0.589 ± 0.136** | **0.457 ± 0.226** |
| XGBoost | 0.779 ± 0.081 | 0.826 ± 0.102 | 0.218 ± 0.041 | 0.21 ± 0.30 | 0.186 ± 0.195 | 0.103 ± 0.183 |
| Ensemble *(Honours architecture, no FS)* | 0.792 ± 0.076 | 0.860 ± 0.071 | 0.197 ± 0.024 | **1.02 ± 0.48** | 0.585 ± 0.138 | 0.370 ± 0.258 |

Means ± standard deviation across the four LODO folds. Per-fold bootstrap CIs in `reports/v1/metrics_per_fold.json`.

**Reading.** Honours-architecture Ensemble lands fourth on AUROC / AUPRC / Brier (TabICL > LR > Ensemble > XGBoost on AUROC; same order swapped on Brier), but **first on calibration slope** (1.02 vs ideal 1.0), within 0.4pp of LR on Sens@85% spec, and — as we'll see in §4 — first on AUROC on the LongBeachVA fold. The architecture is real; it survives the protocol that destroyed XGBoost via isotonic; but it does not displace TabICL as the headline.

### How to read this against the report's published headline

The report's headline (Final Report §7.2 Table 2.2) — WOA-Ensemble HFP sensitivity 89.72%, specificity 83.12% — was produced under:

- A single 80/20 stratified split (no cross-validation, no bootstrap CIs, no per-source breakdown).
- A WOA feature-selection layer that does not exist in the supplied archive.
- No post-hoc calibration; sensitivity / specificity reported at the default 0.5 threshold; no calibration slope, Brier score, AUROC, or DCA reported on HFP.
- A "validation set" of unspecified size; no calibration slice.

Our table reports under:

- 4-fold LODO-CV (each UCI source held out in turn).
- No FS layer (Honours-Ensemble row); sigmoid (Platt) post-hoc calibration on the within-fold calibration slice.
- Bootstrap CIs at 2,000 resamples per fold (in the per-fold table in [`08-v1-model-results.md`](./08-v1-model-results.md) §2).
- Sensitivity reported at clinically meaningful operating points (85% / 90% specificity), not at the 0.5 threshold.

These are not the same evaluation. The report's protocol is *easier* (random 80/20 splits leak source-level structure into the test set; the unconditional sensitivity at threshold 0.5 of an uncalibrated model is the most generous metric you can quote); ours is *harder* (each LODO fold tests on a source the model has never seen, and we hold the operating point to a clinically interesting specificity). A direct numerical comparison ("89.72% vs our X%") is therefore misleading in either direction. The right reading is a *qualitative* one: the Honours architecture's relative position against TabICL / LR / XGBoost under a fair LODO protocol.

## 4. Per-fold honest reading

| Held-out source | TabICL AUROC | LR AUROC | XGBoost AUROC | Ensemble AUROC |
|---|---|---|---|---|
| Cleveland (n=303, prev 0.46) | **0.877** | 0.863 | 0.838 | 0.832 |
| Hungarian (n=294, prev 0.36) | **0.893** | 0.886 | 0.859 | 0.877 |
| LongBeachVA (n=200, prev 0.75) | 0.740 | 0.733 | 0.702 | **0.745** |
| Switzerland (n=123, prev 0.94) | **0.736** | 0.733 | 0.717 | 0.714 |

(95% bootstrap CIs per fold in `reports/v1/metrics_per_fold.json` — they overlap heavily; the per-fold ranking should not be over-interpreted.)

Three honest sub-readings:

- **Cleveland and Hungarian (low–moderate prevalence)** are the easiest folds for every model. The Honours-Ensemble lands fourth on Cleveland and second on Hungarian — both within ~5pp of TabICL, both with overlapping 95% CIs.
- **LongBeachVA (very high prevalence 0.75)** is the most interesting comparison. The Ensemble *just* edges TabICL on AUROC (0.745 vs 0.740) and posts the lowest Brier on the fold (0.168 vs 0.174). It is also the only model whose net benefit at the 10% AusCVDRisk threshold exceeds treat-all (NB +0.7178 vs +0.7167; see [`08-v1-model-results.md`](./08-v1-model-results.md) §5). The win is within bootstrap noise but the *direction* is consistent — the deeper architecture appears to be slightly more robust to the prevalence-inverted regime than the TFM's in-context learning at n_test=200. We do **not** claim the Ensemble is the right LongBeachVA model on this evidence; we claim only that it is *not worse* than TabICL on this fold, which is itself a non-trivial finding.
- **Switzerland (extreme prevalence 0.94)** is degenerate-by-design and every model lands in the same band (AUROC 0.714–0.736). The Ensemble is fourth here.

**The Honours team's report does not address this per-source structure** — the report's headline is one number, computed on a single random split that mixes all five sources. That is the most important gap between our protocol and theirs, independent of the WOA-code-missing finding above. Under our LODO protocol, *no* model holds its Cleveland/Hungarian-level performance on LongBeachVA/Switzerland; the report's 89.72% sensitivity is plausibly a Cleveland/Hungarian-dominated number that was never tested on the harder source distributions.

## 5. Subgroup audit (sex + age band)

Full per-(model × stratum) AUROC tables are in [`08-v1-model-results.md`](./08-v1-model-results.md) §3. Honours-Ensemble-specific findings:

**Sex (LongBeachVA F=6 / Switzerland F=10 are below the `min_stratum_size` guard and therefore unaudited for any model):**

- *Cleveland* — F=0.848 / M=0.811 (gap 0.037). Smallest sex gap of the four models on this fold; the Ensemble does not introduce a structural F-deficit.
- *Hungarian* — F=0.733 / M=0.875 (**gap 0.142**). The largest sex gap any of the four v1 models posts on any auditable fold. TabICL's gap on the same slice is 0.099, LR's is 0.054, XGBoost's is 0.037. This is the strongest single argument that the Honours architecture, even reproduced faithfully, is **not** a clear upgrade over the v1 trio for clinical deployment.

**Age band (≥70 stratum is n=10 Cleveland / n=0 Hungarian / n=16 LongBeachVA / n=5 Switzerland):**

- *Cleveland* — Ensemble's ≥70 AUROC is 0.833 (n=10). The other three models all post 1.000 on the same 10 rows; the Ensemble's lower number is a useful reminder that any ≥70 number on n=10 is noise. We do not read into either.
- *LongBeachVA* — the only meaningfully populated ≥70 stratum (n=16). Ensemble posts the **worst** AUROC of the four (0.393, vs TabICL 0.464 / LR 0.536 / XGBoost 0.518) and the **largest gap against its best stratum** (0.440). This does not close the LongBeachVA-≥70 gap — it widens it.

**Reading.** The Honours architecture does not break the v1 trio's structural under-service of older LongBeachVA-style patients; it makes it worse on this evidence. The MODEL_CARD.md flags LongBeachVA-style ≥70 patients as out-of-scope for *all four* v1 models, including the Ensemble.

## 6. Decision-curve analysis at AusCVDRisk thresholds

Full per-fold DCA table is in [`08-v1-model-results.md`](./08-v1-model-results.md) §5. Honours-Ensemble-specific findings:

- **Cleveland @ 5% / 10%:** tied with treat-all (NB +0.430 / +0.399). Indistinguishable from LR and TabICL.
- **Hungarian @ 5% / 10%:** tied / tied. Slightly behind TabICL on the 10% threshold.
- **LongBeachVA @ 10%:** **the Ensemble is the only one of the four models that beats treat-all** (NB +0.7178 vs +0.7167, gap +0.0011). LR and XGBoost are *worse than treat-all* on this fold; TabICL ties. The gap is well within bootstrap noise and the prevalence dominates the signal, but the direction is real and suggests the Ensemble's averaging may help in the prevalence-inverted regime.
- **Switzerland @ 5% / 10%:** worse than treat-all. Same pattern as the v1 trio — at 0.94 prevalence the threshold is below the base rate and treat-all dominates structurally.

**Reading.** The Honours architecture's DCA story under our protocol is *qualitatively the same* as the v1 trio's: useful where prevalence is moderate, useless where prevalence is extreme. The LongBeachVA @ 10% finding is the only place the Ensemble offers anything the v1 trio doesn't, and it is well within noise.

## 7. Why we did not implement WOA from scratch

Three options were surfaced at planning ([ADR-012](../adr/012-honours-baseline-reproduction.md) §"Why Path A and not Path B"):

1. **Path A (chosen)**: port only what the archive contains.
2. Path B: implement WOA from scratch (Mirjalili & Lewis 2016) as a feature-selection wrapper, choose defensible hyperparameters (population, max iterations, fitness function), pair with the same Ensemble.
3. Path C: both A + B.

Reasons for choosing A:

- **Honesty.** Any WOA implementation we ship is *our* implementation and *our* hyperparameter choices. Calling it a "reproduction" of WOA-Ensemble would be misleading; calling it a "reconstruction" is honest but invites the very-fair criticism "you used different hyperparameters from us, so the comparison isn't valid." Better to not invite that criticism by not making the claim.
- **The narrative is already complete.** Phase 2.3b's result (TabICL beats LR beats XGBoost on calibrated LODO) is strong enough on its own. Adding the Ensemble row answers the only Phase 2.4 question that needs answering — "does the Honours architecture hold up under our LODO protocol?" — without needing a WOA reconstruction.
- **Time budget.** Path A is ~1 day of implementation; Path B is 2-3 days. The marginal evidential value of Path B given the honesty caveat does not justify the time.

The user agreed to Path A at planning. If a future maintainer (or the original Honours team) ships a recovered WOA implementation, the Ensemble wrapper plugs into a `WOAEnsembleModel` subclass without disrupting the rest of the v1 stack — Phase 2.4b is a one-PR sub-phase if needed.

## 8. Other deferred reconstructions

The Honours archive is also missing implementations for: GWO, ACO, Firefly Algorithm, Snake Optimisation, Bat Optimisation, HHO, Cuckoo Search, RFE-CV. The same logic applies to each: any reconstruction we ship is our implementation, not the Honours team's. None of these are reconstructed in Phase 2.4.

The WOCLSA architecture's WOA hyperparameter-tuning layer is similarly absent (`cnnlstma()` in `Data_Pre-processing.ipynb` cell 51 ships with fixed hyperparameters). The fixed-hyperparameter `cnnlstma()` *could* be reproduced under the same LODO protocol, but the report frames WOCLSA's value as the WOA tuning layer, not the underlying CNN+LSTM+ANN stack. Reproducing the unwrapped stack and labelling it "WOCLSA" would misrepresent what the report claims; we do not do that.

## 9. What this means for the public-repo audience

The signal a portfolio reader should take from Phase 2.4 is **not** "the Honours team's results don't replicate" — that's an unfair characterisation given the archive gap is most likely a handover preservation issue, not a methodology critique. The signal is:

1. The Honours team's *architecture* still works under a more careful evaluation protocol — the Ensemble row in §3 is honest evidence of that.
2. The Honours team's headline *number* is from a more permissive evaluation than ours; quoting it next to our LODO numbers without context would be misleading.
3. Engineering due diligence on a research project includes verifying that supplied artefacts can actually reproduce supplied claims. When they can't, the right move is to document the gap openly and run the closest fair comparison you can — which is what this phase does.

## 10. Pointers

- [`08-v1-model-results.md`](./08-v1-model-results.md) — full v1 results table the Ensemble row appends to.
- [ADR-012](../adr/012-honours-baseline-reproduction.md) — binding decision: Path A + PyTorch port + sigmoid calibration + identical eval harness.
- [`01-honours-recap.md`](./01-honours-recap.md) §8 — patched in this PR with the implementation-gap disclaimer immediately under the report's results table.
- [`cardiorisk/models/ensemble.py`](../../backend/cardiorisk/models/ensemble.py) — the wrapper.
- [`tests/test_models_ensemble.py`](../../backend/tests/test_models_ensemble.py) — wrapper smoke tests including the "predict_proba is the mean of four sub-model outputs" audit.
- `reports/v1/metrics_per_fold.json` and `reports/v1/metrics_aggregate.json` — refreshed by this PR with the Ensemble row.
