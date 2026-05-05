"""v1 risk-model wrappers (Phase 2.3b).

Each module in this package wraps one trained estimator the design doc
([`04-revised-design.md`](../../../docs/research/04-revised-design.md)
§2 + ADR-006 / ADR-011) commits to:

- :mod:`cardiorisk.models.tabicl` — TabICL (TFM headline; ADR-011
  supersedes ADR-006's TabPFN choice).
- :mod:`cardiorisk.models.xgboost_model` — XGBoost + Optuna hyperparam
  tuning. Calibration is applied externally by the training driver
  via :func:`cardiorisk.calibration.calibrate_for_model`.
- :mod:`cardiorisk.models.lr` — L1 logistic regression with restricted
  cubic spline expansions (transparency anchor).

The shared interface lives in :mod:`cardiorisk.models.base` as a
:class:`~cardiorisk.models.base.ModelWrapper` Protocol. Every wrapper
exposes the same surface (``fit(X, y) -> Self`` and ``predict_proba(X)
-> np.ndarray``) so the Phase-2.3b training driver can iterate over
them without per-model branching.

The Phase-2.4 WOA-Ensemble reproduction will land as a fourth module
in this package without changing the shared interface.
"""

from cardiorisk.models.base import ModelWrapper

__all__ = ["ModelWrapper"]
