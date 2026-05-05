# ADR-011 — TabICL replaces TabPFN as the v1 TFM headline

- Status: **Accepted** (Phase 2.3b)
- Date: 2026-05-05
- Deciders: maintainer (with user approval surfaced inline during Phase 2.3b kickoff)
- Phase: 2.3b
- Supersedes: [ADR-006](./006-risk-model-architecture.md) §"Headline (lead-in) model" (TabPFN). All other ADR-006 decisions remain in force.

## Decision

The v1 risk model's Tabular Foundation Model headline is **TabICL** ([Inria Soda team, 2025](https://github.com/soda-inria/tabicl), BSD-3-Clause), not TabPFN as originally specified in ADR-006. The pinned version is `tabicl>=2.1,<2.2` (currently 2.1.1), pulling weights from the public Hugging Face repo `jingang/TabICL` on first use with no authentication required.

Everything else in ADR-006 stands: XGBoost+isotonic remains the white-box workhorse; L1 LR with restricted-cubic-spline expansions remains the transparency anchor; WOA-Ensemble (Phase 2.4) remains the honesty baseline; LODO-CV remains the evaluation protocol; TRIPOD+AI remains the reporting standard.

## Context

ADR-006 was written in May 2026 and specified TabPFN v2.5/v2.6 as the headline model. Two empirically discovered constraints made that choice unworkable at Phase 2.3b implementation time:

1. **TabPFN's licensing model changed.** The current TabPFN release line (7.x as of May 2026) requires the user to accept a [Prior Labs License](https://ux.priorlabs.ai) (Apache-2.0 with an additional Llama-3-style attribution clause) and to obtain a `TABPFN_TOKEN` from Prior Labs before model weights can be downloaded. The token check happens at `.fit()` time and raises `TabPFNLicenseError` in non-interactive environments unless the env var is set. This was added between TabPFN 2.x (which the ADR-006 design assumed) and the current 7.x line.

2. **Older TabPFN (≤6.x) is incompatible with our scikit-learn 1.8.** TabPFN 2.2.1 caps at `scikit-learn<1.7`; TabPFN 6.0.0 caps at `scikit-learn<1.8`. Downgrading sklearn to 1.7 would unwind the Phase 2.2 preprocessing pipeline (which depends on sklearn-1.8-only behaviour around `IterativeImputer`, `FrozenEstimator`, and the `penalty=None` deprecation path that ADR-009 already accommodated).

The combination — current TabPFN gates reproduction behind a third-party account, older TabPFN breaks the existing infrastructure — meant either (a) accepting a CI secret + a manual setup step for outside reproducers, (b) dropping the TFM headline entirely, or (c) substituting an alternative TFM. This ADR documents (c).

The substitution preserves what ADR-006 §"Headline (lead-in) model" actually cares about — running a *Tabular Foundation Model* in production-like conditions in early 2026 as the engineering signal — while removing the licensing friction that conflicted with ADR-010's "anyone can `train_v1.py --reproduce`" commitment.

## TabICL: what it is, why it qualifies

TabICL is a tabular in-context learning foundation model from Inria's Soda team, published in 2025. It is in the same architectural family as TabPFN (transformer pretrained on synthetic tabular tasks; zero-shot inference via in-context demonstrations of (X_train, y_train) followed by query rows). The relevant properties for our v1:

- **License**: BSD 3-Clause. No additional attribution clause beyond the standard BSD one. Compatible with our MIT-licensed repo. Compatible with public-repo distribution.
- **Weights**: hosted on Hugging Face Hub at `jingang/TabICL`, downloaded automatically on first use via `huggingface_hub`. No token required.
- **API**: scikit-learn-compatible. `TabICLClassifier(device='cpu', random_state=42).fit(X, y).predict_proba(X_test)`. Same shape as TabPFN's classifier.
- **NaN handling**: NaN in input passes through cleanly (verified by smoke test in `cardiorisk/models/tabicl.py`'s test). The ADR-008 TabPFN pipeline (NaN-passthrough column transformer) works unmodified.
- **Calibration**: TabICL is calibrated by construction in the same way TabPFN is — the in-context-learning training objective is direct probabilistic prediction, not a margin-based score. We treat it the same way in `cardiorisk.calibration.calibrate_for_model` (passes through unwrapped; no post-hoc calibrator).
- **CPU-only inference cost**: comparable to TabPFN — sub-second per (fit, predict_proba) call on a 700-row LODO fold, no GPU required.
- **scikit-learn 1.8 compatibility**: yes. No version conflicts.

The papers + benchmarks that motivate using a TFM at all (Hollmann et al. 2023, the 2024–2026 follow-up evidence) apply to TabICL too — both are pretrained on synthetic tabular tasks; both demonstrate matching/beating tuned XGBoost zero-shot on small clinical cohorts; the published TabICL benchmarks in particular show competitive performance on the OpenML-CC18 suite and on small-n binary clinical tasks.

## Consequences

### Positive

- **Reproducibility commitment from ADR-010 is preserved.** Anyone with a clone of the repo can run `train_v1.py --reproduce` and get the headline TFM row populated, with no PriorLabs account, no `TABPFN_TOKEN` setup, no CI secret. Outside contributors and recruiters reading the repo cold can run the full pipeline themselves.
- **CI doesn't need a TABPFN_TOKEN secret.** The `train-v1-smoke` step runs end-to-end including the TFM row.
- **License compatibility is unambiguous.** BSD-3 has no additional attribution requirement beyond the boilerplate; the model card section on third-party licences will read as a standard short paragraph.
- **The "ran a TFM in production-like conditions in early 2026" engineering signal from ADR-006 is preserved.** TabICL is an in-context-learning TFM in the same family; it is the *kind* of thing ADR-006 was reaching for, even though the *brand* changed.
- **No `tabpfn` install footprint.** TabICL has a smaller dependency closure than TabPFN 7.x (fewer transitive deps, no `pydantic`-based settings layer, no `posthog` telemetry).

### Negative

- **TabICL is less widely cited than TabPFN as of May 2026.** A reviewer recognising "TabPFN" instantly may need to read the ADR to understand what TabICL is. Mitigation: this ADR + the model card both link to the TabICL paper and benchmark suite; the engineering signal of *picking the appropriate TFM given the licensing constraint* is itself a defensible signal.
- **Published HFP benchmarks for TabICL specifically are thinner than for TabPFN.** We will report our own LODO numbers honestly per `04-revised-design.md` §5; this risk is the same one ADR-006 already accepts ("the published headline on HFP under LODO-CV is likely to be lower than the Honours headline").
- **Some downstream tooling (e.g. SHAP integrations, Phase 2.5) is documented for TabPFN, not TabICL.** KernelSHAP works on any `predict_proba`-shaped model so the integration is straightforward; the documentation will need a sentence noting we're using TabICL.
- **If TabICL's maintenance pace slows or PriorLabs relaxes the TabPFN licence later, we may want to revisit.** Captured in the trigger-to-revisit section below.

### Easier now

- Onboarding outside contributors. They run `uv sync && uv run python backend/scripts/train_v1.py --reproduce` and get the same TFM row we report.
- Public-repo distribution. No "you must register at a third party first" caveat in the README.
- CI configuration. No `secrets.TABPFN_TOKEN` in `.github/workflows/ci.yml`.

### Harder now

- Explaining the TFM choice to someone who came expecting TabPFN. The first paragraph of `08-v1-model-results.md` and the model card both call out the substitution explicitly.

## Alternatives considered

### A. Keep TabPFN 7.x, add `TABPFN_TOKEN` as a CI secret

Rejected. Honours the letter of ADR-006; erodes its spirit (and ADR-010's). Outside reproducers — the recruiters and contributors the public repo is meant for — would need to register at PriorLabs, accept their license, and configure their own token before `train_v1.py --reproduce` succeeds. That's a meaningful adoption barrier for a portfolio piece. Also adds an irreversible CI secret to manage.

### B. Drop the TFM entirely; XGBoost+isotonic as the v1 headline

Rejected. The "I shipped a TFM in production-like conditions in early 2026" signal from ADR-006 is non-trivial — it demonstrates awareness of the 2024–2026 tabular ML literature in a way XGBoost-only doesn't. With TabICL as a free, capable, license-compatible substitute, we can keep that signal at no reproducibility cost.

### C. TabPFN 7.x but local-only; CI runs LR+XGBoost only

Rejected. Splits the TFM row's reproducibility from the others, weakens CI's regression coverage, and still requires the token for anyone trying to reproduce the headline. The mixed model is also harder to explain than a clean substitution.

### D. Downgrade sklearn to 1.7 to unlock TabPFN 6.x

Rejected. Unwinds Phase 2.2's `IterativeImputer` work and Phase 2.3a's `FrozenEstimator` calibration wrapper, both of which depend on sklearn 1.8 behaviour. The cost of the downgrade is much larger than the value of preserving the specific TabPFN brand.

### E. Other TFM candidates (`mothernet`, custom)

`mothernet` was checked but is less mature and has its own dependency awkwardness. Custom in-house TFM training is wildly out of scope for v1. TabICL was the cleanest hit.

## Trigger to revisit

Re-open this ADR if any of the following becomes true:

- TabICL's maintenance pace slows materially (no releases in >12 months, unmerged critical issues against current sklearn / torch).
- PriorLabs relaxes the TabPFN license to a no-token-required model and publishes weights on a public CDN. We would re-evaluate against TabICL on benchmark performance.
- A new TFM with stronger published HFP-comparable results (e.g. from Phase 6's multi-model evaluation work) emerges with a permissive license.
- A senior reviewer demonstrates a structural critique of TabICL we missed (always possible — public-PR review process welcome).

## Related decisions

- **ADR-006 §"Headline (lead-in) model"** is *partially superseded* by this ADR. ADR-006 remains the binding source on every other Phase-1 design choice (XGBoost+isotonic, L1 LR with RCS, WOA-Ensemble baseline, LODO-CV, TRIPOD+AI).
- **ADR-010** (model artefact storage — local + reproduce-script) directly motivates this ADR. Without ADR-010's no-auth reproducibility commitment, option A (TabPFN+token) would have been viable.
- **ADR-009** (eval harness) is unaffected. The harness is model-agnostic; it works against any `predict_proba`-shaped output.

## References

- [TabICL repository (Inria Soda team)](https://github.com/soda-inria/tabicl)
- [TabICL on PyPI](https://pypi.org/project/tabicl/) — BSD 3-Clause
- [TabICL weights on Hugging Face Hub (`jingang/TabICL`)](https://huggingface.co/jingang/TabICL)
- [Hollmann et al. 2023 — TabPFN](https://arxiv.org/abs/2207.01848) (the original TFM paper this whole class of models traces to)
- [PriorLabs License (TabPFN 6.x+)](https://ux.priorlabs.ai) — the licensing change that triggered this ADR
