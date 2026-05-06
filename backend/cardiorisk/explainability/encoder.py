"""Shared SHAP-feature-space encoder for raw HFP DataFrames.

KernelSHAP perturbs numeric arrays. Our raw HFP DataFrame mixes
continuous numerics (``Age``, ``Cholesterol``, ...), binary numerics
(``FastingBS``), missingness indicators (``RestingBP_was_missing``, ...),
and string categoricals (``ChestPainType``, ``RestingECG``, ``Sex``,
``ExerciseAngina``, ``ST_Slope``). To explain all four v1 models in a
single, model-agnostic numeric space we one-hot-encode the categoricals
*once* into a fixed wide matrix, then provide a :func:`decode` that
turns rows of that matrix back into the raw DataFrame each model's
internal preprocessing pipeline expects.

This is deliberately a *shared* encoder, not per-model:

- Each model wrapper (LR, XGBoost, TabICL, Ensemble) still applies its
  own internal preprocessing (RCS for LR, MissForest for XGBoost +
  Ensemble, OHE-then-passthrough for TabICL) inside ``predict_proba``.
- The encoder's only job is to give KernelSHAP a uniform numeric
  surface to perturb against, and to provide an inverse so the model
  wrappers can be called with the DataFrame they want.

ADR-013 §"Background data" pins the design.

Why we don't try to use each model's *own* post-preprocessing matrix
as the SHAP feature space:

- The models have different preprocessing pipelines (RCS expansion
  expands ``Age`` to 3 columns for LR but keeps it as 1 for XGBoost).
- Cross-model SHAP comparison would need to reconcile ``Age`` (XGB)
  vs ``Age, Age_rcs1, Age_rcs2`` (LR) -- the very sum-back step we
  do once at the encoder level here, but having it folded into the
  KernelSHAP loop would obscure it and complicate the per-spline-basis
  LR-detail figure.
- A single shared OHE encoding makes the cross-model agreement matrix
  trivially apples-to-apples: every model's KernelSHAP returns values
  per OHE column, and we sum across the same OHE blocks regardless of
  which model the explainer was attached to.

The LR-detail "per-spline-basis" view is computed separately and
analytically (no SHAP library call) by
:mod:`cardiorisk.explainability.linear_attribution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    MISSING_CATEGORY_LABEL,
    MISSINGNESS_INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
)

#: Indicator columns appended by
#: :func:`cardiorisk.data.preprocess.add_missingness_indicators`. These
#: are already numeric (0/1), so the encoder passes them through.
INDICATOR_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{c}_was_missing" for c in MISSINGNESS_INDICATOR_COLUMNS
)


@dataclass(frozen=True)
class FeatureGroup:
    """A raw HFP feature and the encoded-matrix column indices that map to it.

    For numerics and indicators ``column_indices`` has length 1 (the
    encoded space is one-to-one with the raw feature). For categoricals
    ``column_indices`` has one entry per OHE level (including the
    explicit ``"Missing"`` level emitted by
    :func:`cardiorisk.data.preprocess.replace_categorical_missing`).
    """

    raw_name: str
    column_indices: tuple[int, ...]
    is_categorical: bool


@dataclass(frozen=True)
class EncodedFeatureSpace:
    """The fixed numeric SHAP feature space + the inverse to raw DataFrames.

    Construct via :func:`fit_encoder` on the per-fold training slice.
    ``encode`` and ``decode`` are then deterministic and stateless.

    Attributes
    ----------
    column_names : tuple of str
        Encoded-matrix column names, one per encoded column. Numeric
        and indicator columns keep their raw HFP name; categoricals
        become ``<raw>__<level>`` (double-underscore so the split-back
        in :meth:`group_for` is unambiguous).
    feature_groups : tuple of FeatureGroup
        One entry per raw HFP feature, in the order the encoder was
        fit. Used by both KernelSHAP (sum SHAP values across each
        group) and the figure layer (label bars by raw feature name).
    """

    column_names: tuple[str, ...]
    feature_groups: tuple[FeatureGroup, ...]

    # private fitted state -- not in the public dataclass surface
    _ohe: OneHotEncoder
    _categorical_columns: tuple[str, ...]
    _numeric_columns: tuple[str, ...]
    _binary_columns: tuple[str, ...]
    _indicator_columns: tuple[str, ...]

    @property
    def n_columns(self) -> int:
        return len(self.column_names)

    @property
    def n_groups(self) -> int:
        return len(self.feature_groups)

    @property
    def raw_feature_names(self) -> tuple[str, ...]:
        """Raw HFP feature names, in the encoder's fit order."""
        return tuple(g.raw_name for g in self.feature_groups)

    def encode(self, df: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Transform a raw HFP DataFrame into the SHAP feature space.

        Categoricals are one-hot expanded; numerics, binaries, and
        indicators pass through unchanged. NaN in numeric columns
        survives the encoding (KernelSHAP treats NaN as just another
        value to perturb against the background).
        """
        ohe_block = self._ohe.transform(df[list(self._categorical_columns)])
        numeric_block = df[list(self._numeric_columns)].to_numpy(dtype=np.float64, na_value=np.nan)
        binary_block = df[list(self._binary_columns)].to_numpy(dtype=np.float64, na_value=np.nan)
        indicator_block = df[list(self._indicator_columns)].to_numpy(dtype=np.float64, na_value=0.0)
        return np.concatenate(
            [ohe_block, numeric_block, binary_block, indicator_block],
            axis=1,
        )

    def decode(self, arr: npt.NDArray[np.float64]) -> pd.DataFrame:
        """Round-trip an encoded matrix back to a raw HFP DataFrame.

        For one-hot blocks the inverse picks the argmax over the
        block (which is also what KernelSHAP-perturbed intermediate
        states want -- the "two categories on at once" combinatorial
        states it can produce are decoded to the dominant level).

        The returned frame has the same columns the model wrappers'
        internal pipelines expect: categoricals as object/string,
        numerics + binaries as float64, indicators as float64.
        """
        if arr.ndim != 2:
            raise ValueError(f"arr must be 2-D; got shape {arr.shape}")
        if arr.shape[1] != self.n_columns:
            raise ValueError(f"arr has {arr.shape[1]} columns; encoder fit on {self.n_columns}")

        out: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.object_]] = {}

        for cat_col in self._categorical_columns:
            group = self._group_for(cat_col)
            block = arr[:, list(group.column_indices)]
            argmax_idx = np.argmax(block, axis=1)
            categories = self._ohe.categories_[self._categorical_columns.index(cat_col)]
            out[cat_col] = np.asarray(categories)[argmax_idx]

        for num_col in self._numeric_columns:
            group = self._group_for(num_col)
            (idx,) = group.column_indices
            out[num_col] = arr[:, idx]

        for bin_col in self._binary_columns:
            group = self._group_for(bin_col)
            (idx,) = group.column_indices
            out[bin_col] = arr[:, idx]

        for ind_col in self._indicator_columns:
            group = self._group_for(ind_col)
            (idx,) = group.column_indices
            out[ind_col] = arr[:, idx]

        return pd.DataFrame(out)

    def _group_for(self, raw_name: str) -> FeatureGroup:
        for g in self.feature_groups:
            if g.raw_name == raw_name:
                return g
        raise KeyError(f"unknown raw feature {raw_name!r}")

    def aggregate_shap(self, shap_per_column: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sum per-encoded-column SHAP values back to per-raw-feature.

        Input shape: ``(n_rows, n_columns)``. Output shape:
        ``(n_rows, n_groups)`` with columns in :attr:`raw_feature_names`
        order. For non-categorical features the operation is a copy;
        for categoricals it is a sum across the OHE block.
        """
        if shap_per_column.ndim != 2 or shap_per_column.shape[1] != self.n_columns:
            raise ValueError(
                f"expected shape (n_rows, {self.n_columns}); got {shap_per_column.shape}"
            )
        out = np.empty((shap_per_column.shape[0], self.n_groups), dtype=np.float64)
        for j, group in enumerate(self.feature_groups):
            cols = list(group.column_indices)
            out[:, j] = shap_per_column[:, cols].sum(axis=1)
        return out


def fit_encoder(df_train: pd.DataFrame) -> EncodedFeatureSpace:
    """Fit the shared SHAP-feature-space encoder on a per-fold training frame.

    The frame must already be cleaned (run through
    :func:`cardiorisk.data.preprocess.clean_for_modelling` first), so
    categoricals carry the explicit ``"Missing"`` level and the
    missingness indicator columns are present.
    """
    _validate_frame(df_train)

    categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS
    binary_columns: tuple[str, ...] = BINARY_NUMERIC_COLUMNS
    indicator_columns: tuple[str, ...] = INDICATOR_COLUMNS

    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype="float64",
    )
    ohe.fit(df_train[list(categorical_columns)])

    column_names: list[str] = []
    feature_groups: list[FeatureGroup] = []

    for cat_col, levels in zip(categorical_columns, ohe.categories_, strict=True):
        # Levels arrive sorted by sklearn; use them verbatim so the
        # block of column indices for this group is contiguous.
        start = len(column_names)
        for level in levels:
            column_names.append(f"{cat_col}__{level}")
        feature_groups.append(
            FeatureGroup(
                raw_name=cat_col,
                column_indices=tuple(range(start, len(column_names))),
                is_categorical=True,
            )
        )

    for num_col in numeric_columns:
        idx = len(column_names)
        column_names.append(num_col)
        feature_groups.append(
            FeatureGroup(
                raw_name=num_col,
                column_indices=(idx,),
                is_categorical=False,
            )
        )

    for bin_col in binary_columns:
        idx = len(column_names)
        column_names.append(bin_col)
        feature_groups.append(
            FeatureGroup(
                raw_name=bin_col,
                column_indices=(idx,),
                is_categorical=False,
            )
        )

    for ind_col in indicator_columns:
        idx = len(column_names)
        column_names.append(ind_col)
        feature_groups.append(
            FeatureGroup(
                raw_name=ind_col,
                column_indices=(idx,),
                is_categorical=False,
            )
        )

    return EncodedFeatureSpace(
        column_names=tuple(column_names),
        feature_groups=tuple(feature_groups),
        _ohe=ohe,
        _categorical_columns=categorical_columns,
        _numeric_columns=numeric_columns,
        _binary_columns=binary_columns,
        _indicator_columns=indicator_columns,
    )


def _validate_frame(df: pd.DataFrame) -> None:
    """Assert the frame carries the post-cleaning HFP schema."""
    required = (
        list(CATEGORICAL_COLUMNS)
        + list(NUMERIC_COLUMNS)
        + list(BINARY_NUMERIC_COLUMNS)
        + list(INDICATOR_COLUMNS)
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "encoder requires the post-cleaning HFP schema; missing columns: "
            f"{missing}. Did you forget to call clean_for_modelling()?"
        )
    # Check categoricals have the Missing level handled (one nudge towards
    # catching un-cleaned input -- a NaN in a categorical here would silently
    # become its own one-hot level and break round-trip determinism).
    for cat in CATEGORICAL_COLUMNS:
        if df[cat].isna().any():
            raise ValueError(
                f"categorical column {cat!r} contains NaN; run "
                "replace_categorical_missing() (or clean_for_modelling()) first. "
                f"The {MISSING_CATEGORY_LABEL!r} sentinel is required for round-trip."
            )
