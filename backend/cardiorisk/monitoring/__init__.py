"""Phase 2.6 drift / monitoring layer (binding decision: ADR-014).

This package implements a *report-only* drift-monitoring surface for the
v1 risk models. It computes two kinds of drift between a "current" data
slice and a per-fold reference snapshot:

- **Input-feature drift** (per HFP feature): Population Stability Index
  (PSI) for every numeric and categorical feature, plus a Kolmogorov-
  Smirnov two-sample test on numerics as a sanity check.
- **Prediction drift**: PSI on the calibrated ``predict_proba`` of the
  pre-trained per-fold model.

Concept drift (label distribution shift) is deliberately **out of
scope** — it requires labelled new data, which we do not yet have a
deployment producing. ADR-014 documents the deferral.

Module map:

- :mod:`.psi` — pure-function PSI primitives + severity bands.
- :mod:`.ks` — thin :func:`scipy.stats.ks_2samp` wrapper for numeric
  features only.
- :mod:`.reference` — :class:`FoldReference` dataclass, builders, and
  joblib save/load helpers.
- :mod:`.drift` — :func:`compute_drift` that combines the above into a
  per-(model, fold) :class:`DriftReport`.
- :mod:`.figures` — single-PNG-per-cell dashboard renderer.
- :mod:`.orchestrator` — end-to-end driver and ``--smoke``/``--full``
  CLI surface, mirroring :mod:`cardiorisk.explainability.orchestrator`.

Reference choice (per ADR-014): one :class:`FoldReference` per LODO
fold, built from the in-fold *training-pool combined* distribution
(i.e. the same three-source pool the fold's model was fit on). A
single-combined-pool reference would conflate "drift between
deployments" with "drift between which fold the model came from".

Severity bands (per ADR-014, drawn from the industry PSI convention):

- ``stable`` if PSI < 0.10
- ``moderate`` if 0.10 <= PSI < 0.25
- ``major`` if PSI >= 0.25

These thresholds are widely cited but not derived from first principles;
the research doc and ADR-014 document this honestly.
"""
