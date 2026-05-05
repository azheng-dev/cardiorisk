"""Evaluation harness for the v1 risk model (Phase 2.3a).

Layer responsibilities:

- :mod:`cardiorisk.eval.metrics` — discrimination + calibration scalar
  metrics (AUROC, AUPRC, Brier, calibration slope/intercept,
  sensitivity-at-specificity), plus a one-shot ``headline_metrics``
  bundle used by the per-fold reporter.
- :mod:`cardiorisk.eval.dca` — Decision-Curve Analysis (Vickers & Elkin
  2006) computed from raw labels + probabilities. Rolled in-house in
  ~60 lines so the formula is auditable on the page.
- :mod:`cardiorisk.eval.bootstrap` — non-parametric bootstrap CIs
  (2,000 resamples per ``04-revised-design.md`` §5.1), pinned seed,
  deterministic across reruns.
- :mod:`cardiorisk.eval.reliability` — reliability diagrams returned as
  matplotlib ``Figure`` handles (caller saves them where it wants).
- :mod:`cardiorisk.eval.subgroup` — stratified evaluation by sex / age
  band per TRIPOD+AI §5.2, plus the ``fairness_gap`` summary metric.

These functions are deliberately model-agnostic: each takes ``y_true``
and ``y_proba`` numpy arrays and returns scalars / dataclasses /
matplotlib figures. The Phase-2.3b training driver is the only thing
that knows about the LODO splitter, the ``cardiorisk.features``
pipelines, or the model wrappers — the eval layer is reusable.

Phase 2.5 (SHAP) and Phase 6 (multi-model evaluation) will reach into
this package without modification.
"""
