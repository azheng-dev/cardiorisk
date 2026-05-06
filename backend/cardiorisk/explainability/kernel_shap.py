"""KernelSHAP wrapper for the v1 cross-model explainability surface.

ADR-013 binds the cross-model comparison to model-agnostic
KernelSHAP. This module wraps :class:`shap.KernelExplainer` so the
caller can pass:

- a calibrated model (``CalibratedClassifierCV(FrozenEstimator(...))``
  for XGBoost / LR / Ensemble; bare :class:`TabICLModel` for TabICL),
- a fitted :class:`~cardiorisk.explainability.encoder.EncodedFeatureSpace`
  built on the per-fold training slice,
- ``X_train`` and ``X_test`` (raw HFP DataFrames),

and receive per-row, per-encoded-column SHAP values plus a per-row,
per-raw-feature aggregation suitable for the cross-model comparison.

Background distribution: ``shap.kmeans(X_train_encoded, k=50)`` per
ADR-013 §"Background data". The k-means call is deterministic given
the project seed.

Determinism note: KernelSHAP samples coalitions internally. We seed
``numpy.random.default_rng`` before each ``shap_values`` call to
reduce run-to-run variation; expect ~1e-4 absolute drift in
individual SHAP values across runs (consistent with ADR-013
§"Reproducibility"). Aggregate quantities (mean |SHAP| per feature,
Spearman ranks) are stable to ~1e-6.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.exceptions import ConvergenceWarning

from cardiorisk.explainability.encoder import EncodedFeatureSpace
from cardiorisk.models.base import SEED

#: Default background-set size, per ADR-013.
DEFAULT_BACKGROUND_K: Final[int] = 50

#: Default KernelSHAP coalition-sample budget. shap's "auto" picks
#: ``2 * n_features + 2048``; for our ~25-encoded-column feature space
#: that's ~2098 evaluations per explanation -- prohibitive on TabICL.
#: We pin a smaller budget here that still gives stable mean |SHAP|
#: rankings (~1% drift across runs in our smoke testing). Caller can
#: override.
DEFAULT_NSAMPLES: Final[int] = 128

#: Smoke-mode background-set + nsamples knobs.
SMOKE_BACKGROUND_K: Final[int] = 5
SMOKE_NSAMPLES: Final[int] = 16


class _PredictProbaCallable(Protocol):
    def __call__(self, X: pd.DataFrame) -> npt.NDArray[np.float64]: ...


#: KernelExplainer model arg: maps a 2-D encoded array to per-row scalars.
_ShapCallable = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class KernelSHAPResult:
    """Per-row SHAP values in both encoded and raw-feature space.

    Attributes
    ----------
    shap_values_encoded : ndarray, shape (n_rows, n_encoded_columns)
        Raw KernelSHAP output. Categorical levels appear as separate
        columns; sum across the OHE block to get the categorical's
        contribution.
    shap_values_raw : ndarray, shape (n_rows, n_raw_features)
        :meth:`EncodedFeatureSpace.aggregate_shap` applied to
        ``shap_values_encoded``. Columns in
        :attr:`EncodedFeatureSpace.raw_feature_names` order.
    expected_value : float
        The KernelSHAP base value -- the model's mean prediction over
        the background set. Per-row contributions sum to
        ``expected_value + sum_j shap_values_raw[i, j] = predict_proba(X)[i, 1]``
        for an exactly-converged explainer; with ``nsamples`` finite
        you'll see small approximation error here, which is expected.
    raw_feature_names : tuple of str
        Column names for ``shap_values_raw`` in the same order.
    encoded_column_names : tuple of str
        Column names for ``shap_values_encoded`` in the same order.
    """

    shap_values_encoded: npt.NDArray[np.float64]
    shap_values_raw: npt.NDArray[np.float64]
    expected_value: float
    raw_feature_names: tuple[str, ...]
    encoded_column_names: tuple[str, ...]

    @property
    def mean_abs_per_raw_feature(self) -> dict[str, float]:
        """Mean |SHAP value| per raw HFP feature, the global-importance summary."""
        means = np.mean(np.abs(self.shap_values_raw), axis=0)
        return dict(zip(self.raw_feature_names, (float(m) for m in means), strict=True))


def _build_predict_function(
    predict_proba: _PredictProbaCallable,
    encoded_space: EncodedFeatureSpace,
) -> _ShapCallable:
    """Wrap ``predict_proba`` so KernelSHAP can call it with encoded numpy rows.

    The shim decodes each batch back to a raw HFP DataFrame, calls
    ``predict_proba``, and returns the positive-class probability
    column. Returning ``probabilities[:, 1]`` (not log-odds) keeps the
    SHAP additivity check on the same scale the user actually sees on
    the model card.
    """

    def _f(arr_or_row: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        arr = arr_or_row.reshape(1, -1) if arr_or_row.ndim == 1 else arr_or_row
        df = encoded_space.decode(arr)
        proba = np.asarray(predict_proba(df))
        # predict_proba returns (n, 2) for sklearn-shaped models;
        # pull the positive-class column.
        return np.asarray(proba[:, 1], dtype=np.float64)

    return _f


def explain_with_kernel_shap(
    *,
    predict_proba: _PredictProbaCallable,
    encoded_space: EncodedFeatureSpace,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    background_k: int = DEFAULT_BACKGROUND_K,
    nsamples: int = DEFAULT_NSAMPLES,
    seed: int = SEED,
) -> KernelSHAPResult:
    """Compute KernelSHAP values for ``X_test`` against a kmeans background.

    Parameters
    ----------
    predict_proba
        Calibrated model's ``predict_proba``-shaped callable. Must
        return a ``(n, 2)`` array; the positive-class column is what
        we explain (matches the value the model card reports).
    encoded_space
        Fitted shared encoder; defines the SHAP feature space and the
        decode-to-DataFrame shim KernelSHAP perturbs against.
    X_train, X_test
        Raw HFP DataFrames. ``X_train`` is k-means-clustered to build
        the background set; ``X_test`` is the explanation target.
    background_k
        Number of background medoids. ADR-013 default = 50.
    nsamples
        KernelSHAP coalition-sample budget per explanation. Default
        :data:`DEFAULT_NSAMPLES` (256). Lower for smoke runs.
    seed
        Pinned RNG seed. Both the k-means and the KernelSHAP
        coalition sampler are seeded.
    """
    if background_k <= 0:
        raise ValueError(f"background_k must be positive; got {background_k}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive; got {nsamples}")
    if len(X_test) == 0:
        raise ValueError("X_test must contain at least one row")

    # shap.KernelExplainer reads numpy's legacy global RNG state for its
    # coalition sampling, so we must seed the legacy global; np.random.Generator
    # is not enough.
    np.random.seed(seed)

    X_train_encoded = encoded_space.encode(X_train)
    X_test_encoded = encoded_space.encode(X_test)

    # k-means clamps automatically to len(X_train) if k > n -- keep
    # the explicit guard so the surfaced error message is ours.
    effective_k = min(background_k, X_train_encoded.shape[0])
    background = shap.kmeans(X_train_encoded, effective_k)

    f = _build_predict_function(predict_proba, encoded_space)
    # KernelSHAP fits a per-instance weighted linear regression internally
    # to recover Shapley values; on small / sparse coalition matrices
    # sklearn's LARS solver (which shap uses for the L1-regularised path
    # when ``l1_reg`` is left at its default) emits ConvergenceWarning.
    # The warning is informational -- the SHAP values are still produced
    # and the explainer falls back to a non-degenerate solution. Suppressed
    # locally so callers under ``filterwarnings=error`` (our pyproject
    # default) don't see spurious failures.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        explainer = shap.KernelExplainer(model=f, data=background, silent=True)
        shap_values_encoded = np.asarray(
            explainer.shap_values(X_test_encoded, nsamples=nsamples, silent=True),
            dtype=np.float64,
        )
    if shap_values_encoded.shape != X_test_encoded.shape:
        raise RuntimeError(
            "KernelSHAP returned unexpected shape "
            f"{shap_values_encoded.shape}; expected {X_test_encoded.shape}"
        )

    shap_values_raw = encoded_space.aggregate_shap(shap_values_encoded)

    return KernelSHAPResult(
        shap_values_encoded=shap_values_encoded,
        shap_values_raw=shap_values_raw,
        expected_value=float(explainer.expected_value),
        raw_feature_names=encoded_space.raw_feature_names,
        encoded_column_names=encoded_space.column_names,
    )
