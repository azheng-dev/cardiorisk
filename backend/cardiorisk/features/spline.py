"""Restricted-cubic-spline (RCS) basis expansion for the L1 LR baseline.

Implements the parsimonious RCS basis from Harrell (2001) "Regression
Modeling Strategies" §2.4: for ``k`` knots at quantile positions, each
continuous input feature becomes ``k - 1`` columns (the linear term plus
``k - 2`` cubic-piece terms). The spline is **linear beyond the boundary
knots** by construction, which keeps extrapolation well-behaved.

Why RCS rather than sklearn's :class:`~sklearn.preprocessing.SplineTransformer`:

- ``SplineTransformer(degree=3, n_knots=k)`` emits ``k + 2`` B-spline
  basis columns per input. With 5 numeric features and 4 knots that's
  30 added columns — a real concern under L1 regularisation in a 920-row
  dataset.
- RCS emits ``k - 1`` columns per input (15 columns for the same setup),
  preserving the "captures non-linearity without exploding the parameter
  count" property `04-revised-design.md` §2.3 explicitly asks for.

Knot quantile positions follow Harrell's recommended schedule:

- ``k = 3``: 10%, 50%, 90%
- ``k = 4``: 5%, 35%, 65%, 95%
- ``k = 5``: 5%, 27.5%, 50%, 72.5%, 95%

Usage:

>>> import pandas as pd
>>> rcs = RestrictedCubicSpline(n_knots=4)
>>> X = pd.DataFrame({"Age": [30, 45, 60, 75]})
>>> rcs.fit_transform(X).shape  # 1 input feature * (k-1) = 3 columns
(4, 3)
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#: Harrell's recommended quantile positions for each supported knot count.
HARRELL_QUANTILES: Final[dict[int, tuple[float, ...]]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}

DEFAULT_N_KNOTS: Final[int] = 4


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return the RCS basis for one feature, given pre-computed knots.

    Output shape: ``(n_rows, len(knots) - 1)``. Column 0 is the linear
    term; columns 1..k-2 are the cubic-piece terms.
    """
    k = len(knots)
    if k < 3:
        raise ValueError(f"RCS requires at least 3 knots; got {k}")

    t = knots
    # Numerical-scale denominator from Harrell §2.4.
    scale = (t[-1] - t[0]) ** 2

    n = x.shape[0]
    out = np.empty((n, k - 1), dtype=np.float64)
    out[:, 0] = x

    pos_cubed_kminus1 = np.maximum(x - t[-2], 0.0) ** 3
    pos_cubed_k = np.maximum(x - t[-1], 0.0) ** 3
    span_last = t[-1] - t[-2]

    for j in range(1, k - 1):
        pos_cubed_jminus1 = np.maximum(x - t[j - 1], 0.0) ** 3
        weight = (t[-1] - t[j - 1]) / span_last
        out[:, j] = (
            pos_cubed_jminus1 - pos_cubed_kminus1 * weight + pos_cubed_k * (weight - 1.0)
        ) / scale

    return out


class RestrictedCubicSpline(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Sklearn-compatible RCS basis expander, fitting knots per feature.

    Parameters
    ----------
    n_knots : int, default=4
        Number of knots per input feature. Must be one of {3, 4, 5}; knot
        quantile positions follow :data:`HARRELL_QUANTILES`.

    Attributes
    ----------
    knots_ : list[np.ndarray]
        Per-feature knot positions, learned from the training quantiles
        on ``fit``. Length matches ``n_features_in_``.
    feature_names_in_ : np.ndarray
        Input feature names (or generated names if input was a numpy array).
    """

    def __init__(self, n_knots: int = DEFAULT_N_KNOTS) -> None:
        self.n_knots = n_knots

    def _validate_n_knots(self) -> None:
        if self.n_knots not in HARRELL_QUANTILES:
            raise ValueError(
                f"n_knots must be one of {sorted(HARRELL_QUANTILES)}; got {self.n_knots}"
            )

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> RestrictedCubicSpline:
        """Learn per-feature knots from the training-data quantiles."""
        self._validate_n_knots()
        X_arr, names = _as_numpy_with_names(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X_arr.shape}")

        quantiles = HARRELL_QUANTILES[self.n_knots]
        knots: list[np.ndarray] = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            col = col[~np.isnan(col)]
            if col.size < self.n_knots:
                raise ValueError(
                    f"feature {names[j]!r} has only {col.size} non-NaN rows; "
                    f"need at least n_knots={self.n_knots}"
                )
            ks = np.quantile(col, quantiles)
            unique = np.unique(ks)
            if unique.size < 3:
                raise ValueError(
                    f"feature {names[j]!r} has degenerate knot positions {ks.tolist()}; "
                    "RCS requires at least 3 distinct knot values (try fewer knots or "
                    "drop this feature)"
                )
            knots.append(unique)

        self.knots_: list[np.ndarray] = knots
        self.feature_names_in_: np.ndarray = np.asarray(names, dtype=object)
        self.n_features_in_: int = X_arr.shape[1]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply the learned RCS basis to ``X``.

        Output shape: ``(n_rows, sum(len(knots_j) - 1 for j))``.
        """
        if not hasattr(self, "knots_"):
            raise RuntimeError("RestrictedCubicSpline must be fit before transform()")

        X_arr, _ = _as_numpy_with_names(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features; fitted on {self.n_features_in_}")

        out_blocks: list[np.ndarray] = []
        for j, knots in enumerate(self.knots_):
            out_blocks.append(_rcs_basis(X_arr[:, j].astype(np.float64), knots))
        return np.concatenate(out_blocks, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Sklearn-compatible feature-name generator for the expanded basis."""
        if not hasattr(self, "knots_"):
            raise RuntimeError("RestrictedCubicSpline must be fit before get_feature_names_out()")
        names: list[str] = []
        for j, knots in enumerate(self.knots_):
            base = str(self.feature_names_in_[j])
            names.append(base)
            for s in range(1, len(knots) - 1):
                names.append(f"{base}_rcs{s}")
        return np.asarray(names, dtype=object)


def _as_numpy_with_names(X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return ``(X_array, feature_names)`` from a DataFrame or numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64, na_value=np.nan), list(X.columns)
    arr = np.asarray(X, dtype=np.float64)
    names = [f"x{j}" for j in range(arr.shape[1] if arr.ndim == 2 else 1)]
    return arr, names
