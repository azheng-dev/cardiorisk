# ADR-012 — Honours-baseline reproduction strategy (Phase 2.4)

- Status: **Accepted** (Phase 2.4)
- Date: 2026-05-05
- Deciders: maintainer (with user approval surfaced inline during Phase 2.4 kickoff)
- Phase: 2.4
- Relates to: [ADR-006 §"WOA-Ensemble (honesty baseline)"](./006-risk-model-architecture.md), [ADR-009](./009-evaluation-harness-and-acceptance.md), [01-honours-recap.md §8](../research/01-honours-recap.md), [09-honours-vs-v1.md](../research/09-honours-vs-v1.md)

## Decision

The Phase 2.4 Honours-baseline reproduction is **Path A** of the three options surfaced at planning: a faithful PyTorch port of the 4-net mean-averaged Ensemble (DNN + 1D CNN + LSTM + BiLSTM) that lives in the Honours archive, run under the v1 LODO protocol with the same eval harness as TabICL/XGBoost/LR. The Honours WOA feature-selection layer is **not** reconstructed here; the implementation is missing from the supplied archive (see "Critical finding" below) and any reconstruction we shipped would be a new implementation we made up, not the Honours team's. Calibration: **sigmoid (Platt)** on the same 10% within-fold calibration slice the v1 trio uses.

The four binding sub-decisions are:

1. **Architecture**: 4 parallel sub-networks — DNN, 1D CNN, LSTM, BiLSTM — each trained independently on `BCEWithLogitsLoss` (numerically stable equivalent of Keras `binary_crossentropy` on a sigmoid output) with Adam (lr=1e-3), batch size 32, 100 epochs, dropout 0.2 throughout. Inference takes the **mean** of the four sigmoid outputs (no learned meta-model). Hyperparameters are pinned to the Honours notebook (`Demos/Data_Pre-processing.ipynb` cell 55).
2. **DL framework**: **PyTorch**, not the Honours notebook's TensorFlow/Keras. The repo already pulls torch in transitively (TabICL); adding TensorFlow as a parallel backend would roughly double the install footprint and the type-checker config surface for ~140 LoC of model code. Departures from Keras semantics where there is no PyTorch equivalent are documented inline (`recurrent_dropout`, default initialiser).
3. **Calibration**: **sigmoid (Platt)** via `cardiorisk.calibration.calibrate_for_model` on the within-fold calibration slice, matching the L1 LR recipe and diverging deliberately from the XGBoost isotonic recipe. Phase 2.3b found isotonic-on-~50-rows collapsed XGBoost's calibration slope to 0.21 ([`08-v1-model-results.md`](../research/08-v1-model-results.md) §4); the Ensemble's mean-averaged sigmoid output has the same tail-saturation profile, and Niculescu-Mizil & Caruana (2005) demonstrate Platt scaling is more robust at small calibration-set sizes. We want the Ensemble to lose (or win) on its own merits, not on the same calibration mishap we already know how to defuse.
4. **Eval harness**: **identical** to Phase 2.3b — same LODO splits, same within-fold 80/10/10 train/val/calib, same headline metrics + bootstrap CIs + subgroup audit + DCA at AusCVDRisk thresholds. No special pleading; the Ensemble appears as one more row in the same per-fold and aggregate JSONs the v1 trio produces.

## Critical finding: the Honours WOA implementation is not in the archive

Pre-implementation reading of every notebook in `_honours-archive/FIT4701-4702-2024S1-1698/Demos/` (4 notebooks, ~150 cells total) and the Final Report PDF revealed that the WOA feature-selection layer that produces the report's headline number (WOA-Ensemble: sensitivity 89.72%, specificity 83.12% on HFP — Final Report §7.2 Table 2.2) **does not exist as code anywhere in the supplied archive**. Specifically, in `Demos/Data_Pre-processing.ipynb`:

- Cell 38 (markdown) `GWO`, cell 39 (code) — empty single-line `pass`-equivalent.
- Cell 40 (markdown) `WOA`, cell 41 (code) — empty single-line `pass`-equivalent.
- Cell 42 (markdown) `ACO`, cell 43 (code) — empty single-line `pass`-equivalent.
- Cell 44 (markdown) `Firefly Algorithm`, cell 45 (code) — empty single-line `pass`-equivalent.
- Cell 46 (markdown) `Snake Optimisation`, cell 47 (code) — empty single-line `pass`-equivalent.
- Cell 48 (markdown) `Bat Optimisation`, cell 49 (code) — empty single-line `pass`-equivalent.

Working feature-selection code that *is* present in the archive: GA, EAGA (a custom variant), RF, RFE. The four nature-inspired metaheuristics that the report claims were used to produce its headline (WOA, GWO, CS, BA) all have section headers but no code under them. The 4-net Ensemble architecture itself is fully implemented in `Data_Pre-processing.ipynb` cell 55 and the three other Demo notebooks share variants of the same stack — this part is faithfully reproducible.

The Honours WOCLSA architecture (CNN + LSTM + ANN with WOA-tuned hyperparameters) is in the same situation: `Data_Pre-processing.ipynb` cell 51 contains a `cnnlstma()` function with **fixed** hyperparameters; the WOA hyperparameter-tuning layer the report describes is not implemented anywhere.

Inferred history: the WOA / GWO / CS / BA / etc. code was likely developed in a separate Colab notebook by a different team member and was not preserved into the handover archive. The headline numbers in the report were generated by code that has been lost.

## Why Path A and not Path B (WOA reconstruction)

Three options were surfaced to the user at planning:

- **Path A** (chosen): port only what the archive contains — the 4-net Ensemble, no WOA layer.
- Path B: implement WOA from scratch (Mirjalili & Lewis 2016; well-specified; ~150 LoC) as a feature-selection wrapper, choose defensible hyperparameters (population, max iterations, fitness function), pair with the same Ensemble. Frame as "Honours-WOA-Ensemble reconstruction" — explicitly *not* a reproduction.
- Path C: both A + B as separate result rows.

Path A was chosen for three reasons:

1. **Honesty**. Any WOA implementation we ship is *our* implementation and *our* hyperparameter choices, not the Honours team's. Calling that a "reproduction" of WOA-Ensemble would be misleading; calling it a "reconstruction" is honest but invites the very-fair criticism "you used different hyperparameters from us, so the comparison isn't valid." Better to not invite that criticism by not making the claim.
2. **The narrative is already complete.** The Phase 2.3b result ([`08-v1-model-results.md`](../research/08-v1-model-results.md)) already demonstrates that TabICL beats LR beats XGBoost on calibrated LODO. Adding the Honours-Ensemble row answers the only Phase 2.4 question that needs answering — "does the Honours team's *architecture* hold up under our LODO protocol?" — without needing the WOA layer for the headline.
3. **Time budget.** Path A is ~1 day; Path B is 2–3 days; Path C is 3–4. The marginal evidential value of Path B over Path A, given the honesty caveat above, does not justify the time it would take.

The user agreed to Path A at planning (`p24_path` = `path_a_recommended`). The honest framing of the WOA gap lives in [`09-honours-vs-v1.md`](../research/09-honours-vs-v1.md), and the recap doc has been patched to flag the implementation gap directly under the original results table.

## Architecture decisions (Path A)

### Sub-network sizes are pinned to the Honours notebook

From `Data_Pre-processing.ipynb` cell 55 (`ensemble()` function):

- DNN: `Linear(n→100) → ReLU → Dropout(0.2) → Linear(100→64) → ReLU → Dropout(0.2) → Linear(64→128) → ReLU → Dropout(0.2) → Linear(128→1)`.
- 1D CNN: `Conv1d(1→64, k=3) → MaxPool1d(2) → Dropout(0.2) → Flatten → Linear(...→128) → Linear(128→64) → Dropout(0.2) → Linear(64→1)`.
- LSTM: `LSTM(input_size=1, hidden=128, batch_first=True) → take last step → Dropout(0.2) → Linear(128→1)`.
- BiLSTM: `LSTM(..., bidirectional=True) → Linear(256→1)` (bidirectional doubles hidden).

All four sub-networks are trained independently for 100 epochs (smoke: 1) with Adam(lr=1e-3) and `BCEWithLogitsLoss`. Inference: `mean(sigmoid(dnn(X)), sigmoid(cnn(X)), sigmoid(lstm(X)), sigmoid(bilstm(X)))`. No learned meta-model.

### Inputs are reshaped per sub-network

The Honours notebook treats each scalar feature as a single-channel 1-step "sequence" for the convolutional and recurrent paths. We replicate that:

- DNN: `(N, n_features)`.
- CNN: `(N, 1, n_features)` — channel-first per PyTorch convention.
- LSTM / BiLSTM: `(N, n_features, 1)` — `batch_first=True` with `seq_len=n_features`, `input_size=1`.

This is architecturally faithful to the Honours notebook even though "treating each feature as a time step" is not how anyone would naturally use a recurrent network on tabular data. The Honours team made this choice; we reproduce it.

### Preprocessing reuses Phase 2.2's `make_woa_pipeline()`

The Phase 2.2 preprocessing module already provisioned `cardiorisk.features.pipeline.make_woa_pipeline()` for this exact future use: MissForest imputation + StandardScaler on continuous + binary features, OHE on categoricals, indicator passthrough. No new preprocessing code was needed.

### Departures from Keras semantics (documented for honesty)

PyTorch `nn.LSTM` has no equivalent of Keras `LSTM(recurrent_dropout=0.2)`. We omit recurrent dropout on both LSTM sub-networks and apply only output dropout. The architectural class (LSTM-with-dropout) is preserved; the specific stochastic regularisation between time steps is not. Documented inline in `cardiorisk.models.ensemble`.

PyTorch `nn.Linear` defaults to Kaiming-uniform initialisation; Keras `Dense` defaults to Glorot-uniform. We accept the PyTorch defaults — different initialiser, same architecture family. With dropout 0.2 throughout and 100 epochs of training, initialiser sensitivity is low; we do not consider this a meaningful departure but disclose it.

## Calibration decision (sigmoid / Platt)

The user delegated the calibration choice at planning. Rationale for picking sigmoid (Platt) over isotonic or no calibration:

- **Phase 2.3b's empirical evidence**: isotonic on the ~50-row calibration slice collapsed XGBoost's calibration slope to 0.21 ([`08-v1-model-results.md`](../research/08-v1-model-results.md) §4). The Ensemble's mean-averaged sigmoid output has the same tail-saturation behaviour that drove that collapse — multiple sub-models all confidently agreeing inflates the predicted probability mass at the extremes, exactly where isotonic regression has the least data to fit a monotonic spline.
- **Niculescu-Mizil & Caruana (2005)**: Platt scaling is provably more robust than isotonic regression on small calibration sets (n < ~1000). Our calibration slice sits at ~50–100 rows depending on the LODO fold; well inside the Platt-favoured regime.
- **Architectural symmetry**: the Honours-Ensemble's prediction *is* a sigmoid output (mean of four sigmoids is itself in [0, 1]); fitting a one-parameter logistic on top of that is the natural recipe for the family. Isotonic would be over-flexible at our slice size.
- **Honest framing**: if the Ensemble loses to the v1 trio, we want it to lose because the architecture is genuinely outperformed, not because we picked the wrong calibration recipe. Picking the recipe that works at small calibration-set sizes removes one degree of freedom from the result interpretation.

The dispatcher entry is a one-line addition to `cardiorisk.calibration.DEFAULT_METHOD_FOR_MODEL` (`"ensemble": "sigmoid"`).

## Determinism

- `torch.manual_seed(seed)` and `np.random.seed(seed)` are pinned at the start of every `EnsembleModel.fit()`.
- `torch.use_deterministic_algorithms(True)` is **not** enabled — it disables several PyTorch CPU kernels (notably some scatter ops in LSTM backward) we depend on, and the speedup from the non-deterministic kernels is significant on the LSTM/BiLSTM paths.
- Empirically (`tests/test_models_ensemble.py::test_determinism_under_seed`), two fits at the same seed produce predictions identical to ~1e-5. That tolerance is tight enough for the LODO comparison; it does not affect any LODO-CV decision boundary.

## Consequences

### Positive

- **The Honours architecture appears as a fourth row in the v1 results table.** Anyone reading [`08-v1-model-results.md`](../research/08-v1-model-results.md) sees the cross-model comparison directly without needing to mentally bridge two separate documents.
- **Reproducibility commitment from ADR-010 is preserved.** No new dependencies (torch is already pulled by TabICL); no model weights to download (the network trains from scratch each LODO fold); CI smoke runs the Ensemble row end-to-end in ~60s.
- **The "WOA code missing" finding is documented openly.** The Honours team's architecture is honoured (faithfully reproduced), but the report's headline claim is contextualised against the supplied archive's actual contents in [`09-honours-vs-v1.md`](../research/09-honours-vs-v1.md). This is exactly the kind of honest engineering signal the public-repo audience is being shown.
- **The calibration story is consistent.** Phase 2.3b established "sigmoid for small calibration slices, isotonic only when there's enough data"; Phase 2.4 follows that rule.

### Negative

- **The result is not a 1-for-1 reproduction of the report's headline.** The Ensemble we ship is the architecture from the archive code, not the WOA-Ensemble pipeline that produced 89.72% sens / 83.12% spec on HFP. A reviewer expecting that exact number will see a smaller AUROC (under our LODO protocol, no FS layer, calibrated). The honesty-doc `09-honours-vs-v1.md` explains this prominently.
- **PyTorch port introduces small architectural differences from the Keras original** (Glorot vs Kaiming init; no recurrent dropout). Disclosed in the wrapper docstring + this ADR.
- **The Ensemble training is the slowest run of the v1 stack** (~30–60s per LODO fold on CPU, vs <5s for TabICL/LR and ~10s for XGBoost). The full LODO run grew from ~34min to ~40min. Acceptable.

### Easier now

- Cross-model comparison narrative — one table, four rows.
- MODEL_CARD.md — populated from one set of `reports/v1/` outputs.
- Future SHAP / explainability work (Phase 2.5) — the Ensemble exposes the same `predict_proba` surface, so KernelSHAP works against it without per-model branching.

### Harder now

- Re-running the full LODO is slightly more expensive (~6 min more wall time).
- Adding a *real* WOA reconstruction later (deferred to a possible Phase 2.4b) would require re-running the entire Ensemble training under the WOA-selected feature subsets per LODO fold; non-trivial extra wall time.

## Alternatives considered

### Path B — best-effort WOA reconstruction

Rejected for the reasons in the "Why Path A" section above. The honesty cost (calling our reconstruction a "reproduction") and the time cost (~2 days) outweigh the marginal evidential value.

### Path C — Path A + Path B

Rejected. Path A already answers the cleanest version of the Phase 2.4 question; Path B's incremental value is low and its time cost is high. If the user later wants the WOA reconstruction, it can land as a Phase 2.4b sub-phase against the same harness with minimal disruption.

### Keep the Honours notebook in TensorFlow / Keras

Rejected. Adding TensorFlow as a parallel DL backend roughly doubles the install footprint, the type-checker config surface, and the CI dependency-resolution time. For ~140 LoC of model code, the framework consolidation is worth the small loss of byte-faithfulness to the original Keras code.

### Skip post-hoc calibration entirely

Considered. The mean of four sigmoids is in [0, 1] by construction, so the raw output is at least nominally a probability. But Phase 2.3b's reliability diagrams demonstrated that "in [0, 1]" is not the same as "calibrated"; the same tail-saturation that hurt isotonic-XGBoost is present in the Ensemble's averaged sigmoid. Sigmoid (Platt) calibration is cheap (~one extra fit per fold on 50 rows) and the design-doc precedent says "calibrate everything that can be calibrated." Rejected.

### Apply isotonic calibration (matching XGBoost)

Considered for apples-to-apples symmetry with XGBoost. Rejected because Phase 2.3b's empirical evidence is exactly that isotonic-on-this-slice-size collapses; we have no reason to expect the Ensemble would behave differently.

## Trigger to revisit

Re-open this ADR if any of the following becomes true:

- The user later requests a WOA reconstruction (Path B): a Phase 2.4b sub-phase opens and this ADR is amended (or a new ADR-013 is opened) to cover the WOA hyperparameter choices.
- The Honours team's lost WOA code is recovered and shipped to the maintainer: a faithful reproduction (not a reconstruction) becomes possible and this ADR's Path A choice is revisited against the now-available comparison.
- A subsequent phase (e.g. SHAP / explainability) discovers the Ensemble's calibration slope under sigmoid is also poor: the calibration choice may need to be reconsidered.

## Related decisions

- **ADR-006 §"WOA-Ensemble (honesty baseline)"** is *partially superseded* by this ADR. ADR-006 envisaged a faithful WOA reproduction; the supplied archive does not support that, and this ADR documents the substitute (Honours-Ensemble without the WOA layer) and the reasoning.
- **ADR-009** (eval harness) is unaffected. The harness is model-agnostic; the Ensemble plugs in via the same `ModelWrapper` Protocol.
- **ADR-010** (model artefact storage) governs how the trained Ensemble is persisted: same `joblib` recipe, same `models/v1/` location (gitignored), same reproduce-script reproducibility commitment.
- **ADR-011** (TabICL supersedes TabPFN) shares the "preserve the engineering intent when literal reproduction is blocked" pattern.

## References

- Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. *Advances in Engineering Software*, 95, 51–67. (Cited only as the WOA spec the Honours report claims to follow; not implemented in this phase.)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*. (Calibration recipe rationale.)
- Honours Final Report (2024S1-1698) §7.2 Table 2.2 — the WOA-Ensemble HFP headline (sensitivity 89.72%, specificity 83.12%) we are *not* reproducing 1:1.
- Honours `Demos/Data_Pre-processing.ipynb` cell 55 — the Ensemble architecture we *are* reproducing.
