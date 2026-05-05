"""Shared interface for v1 risk-model wrappers.

Every wrapper in :mod:`cardiorisk.models` implements the
:class:`ModelWrapper` Protocol so the Phase-2.3b training driver can
treat them uniformly:

>>> from cardiorisk.models import ModelWrapper
>>> def train(model: ModelWrapper, X_train, y_train, X_test):
...     model.fit(X_train, y_train)
...     return model.predict_proba(X_test)[:, 1]

The Protocol is structural — wrappers don't need to inherit from any
base class, they just need to expose the right attributes. Per-wrapper
modules also provide a thin factory (``build_<model>()``) so the
driver can instantiate them without importing each class directly.

Calibration is *not* part of the wrapper contract. The Phase-2.3a
:func:`cardiorisk.calibration.calibrate_for_model` dispatcher applies
the right post-hoc calibrator (isotonic / sigmoid / passthrough) per
model name — the wrapper just needs to be a fitted ``predict_proba``-
shaped object that ``CalibratedClassifierCV`` can wrap.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Pinned RNG seed (matches the rest of the project).
SEED: Final[int] = 20260505

#: Canonical model names used by ``calibrate_for_model`` and the per-fold JSON
#: schema. The training driver iterates over these in this order so the
#: cross-model results table reads consistently across runs.
MODEL_NAMES: Final[tuple[str, ...]] = ("tabicl", "xgboost", "lr")


@runtime_checkable
class ModelWrapper(Protocol):
    """Structural contract every v1 model wrapper satisfies.

    Concrete wrappers expose:

    - ``fit(X, y) -> Self``: train on the inner training slice (and,
      where applicable, run hyperparameter selection on the inner val
      slice provided via the constructor).
    - ``predict_proba(X) -> np.ndarray``: predict positive-class and
      negative-class probabilities; shape ``(n_rows, 2)`` with columns
      ``[P(y=0), P(y=1)]`` per sklearn convention.
    - ``predict(X) -> np.ndarray``: hard-class prediction. Required by
      ``sklearn.calibration.CalibratedClassifierCV`` which we wrap
      around the fitted estimator for post-hoc calibration.
    - ``model_name`` (class or instance attribute): one of
      :data:`MODEL_NAMES`. Used by the calibration dispatcher.
    """

    model_name: str

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> ModelWrapper: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
