"""Analytic SHAP attribution for the L1 LR + RCS transparency model.

For an additive linear model ``logit(p) = b0 + sum_j beta_j * z_ij``
where ``z`` are the post-preprocessing standardized features, the
exact Shapley value of feature ``j`` for instance ``i`` is

    phi_ij = beta_j * (z_ij - E_train[z_j])

(Lundberg & Lee 2017 §3 "Linear models with independent features".)
For our LR pipeline the post-preprocessing standardization centres
``z`` at the training-set mean (StandardScaler behaviour), so
``E_train[z_j] = 0`` and the formula simplifies to
``phi_ij = beta_j * z_ij``. We compute it explicitly the long way for
robustness if a future preprocessing step changes the centring.

Two output views per ADR-013 §"LR + RCS attribution detail":

- **Per-spline-basis SHAP** -- the LR's *own* features, including the
  3 RCS basis columns per continuous feature. Used in the "LR-detail"
  figure for reviewers who want to see how the spline expansion
  uses each continuous feature.
- **Per-raw-feature SHAP (summed back)** -- spline-basis SHAP values
  summed across the 3 RCS columns of each continuous feature, plus
  one-hot dummy SHAP values summed across each categorical's encoded
  block. The cross-model comparison row uses this view.

The summed-back view is on log-odds scale (LR's natural scale) and is
*not* directly comparable to the probability-scale SHAP values
KernelSHAP produces against the calibrated model. That's the honest
caveat documented in ADR-013 §"Negative / open risks": the LR sanity
check confirms the per-feature *ranking*, not the absolute values.
For absolute-value comparison KernelSHAP on the calibrated LR gives
probability-scale numbers comparable to the other three models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cardiorisk.models.lr import LRModel

#: Marker the RCS transformer uses for spline-basis names.
#: Cross-references the ``get_feature_names_out()`` impl in
#: :class:`cardiorisk.features.spline.RestrictedCubicSpline`.
RCS_BASIS_SUFFIX_PREFIX: Final[str] = "_rcs"


@dataclass(frozen=True)
class LinearAttributionResult:
    """Per-row analytic SHAP for an LR model in two feature views."""

    #: Per-spline-basis log-odds SHAP. Columns in the order LR's
    #: post-preprocessing pipeline emits (after RCS expansion + OHE).
    shap_per_basis: npt.NDArray[np.float64]
    basis_feature_names: tuple[str, ...]

    #: Per-original-feature log-odds SHAP. RCS-expanded continuous
    #: features are summed back to one column; one-hot dummies are
    #: summed back per categorical.
    shap_per_raw: npt.NDArray[np.float64]
    raw_feature_names: tuple[str, ...]

    #: LR's intercept on log-odds scale. Per-row sum over
    #: ``shap_per_basis[i]`` plus this intercept reproduces
    #: ``logit(predict_proba(x_i)[1])`` exactly (modulo float).
    intercept: float

    @property
    def mean_abs_per_raw_feature(self) -> dict[str, float]:
        """Mean |log-odds SHAP| per raw HFP feature."""
        means = np.mean(np.abs(self.shap_per_raw), axis=0)
        return dict(zip(self.raw_feature_names, (float(m) for m in means), strict=True))


def attribute_lr(
    *,
    lr_model: LRModel | Pipeline,
    X_test: pd.DataFrame,
) -> LinearAttributionResult:
    """Compute exact analytic SHAP for an LR model.

    Parameters
    ----------
    lr_model
        Either a fitted :class:`~cardiorisk.models.lr.LRModel` (which
        wraps a Pipeline) or a fitted preprocessing-+-LR sklearn
        Pipeline directly. Either way the function reaches in for the
        ``preprocess`` and ``clf`` named steps.
    X_test
        Raw HFP DataFrame. Will be passed through the same
        preprocessing prefix the LR was fit with.
    """
    pipeline = _resolve_pipeline(lr_model)
    preprocess = pipeline.named_steps["preprocess"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    z = np.asarray(preprocess.transform(X_test), dtype=np.float64)
    coef = np.asarray(clf.coef_, dtype=np.float64).ravel()
    intercept = float(np.asarray(clf.intercept_, dtype=np.float64).ravel()[0])

    if z.shape[1] != coef.shape[0]:
        raise RuntimeError(
            "post-preprocessing dim mismatch: "
            f"transform produced {z.shape[1]} cols, "
            f"LR coef has {coef.shape[0]}"
        )

    shap_per_basis = z * coef[np.newaxis, :]

    basis_names = tuple(str(n) for n in preprocess.get_feature_names_out())
    if len(basis_names) != shap_per_basis.shape[1]:
        raise RuntimeError(
            "feature-name count mismatch: "
            f"{len(basis_names)} names vs {shap_per_basis.shape[1]} cols"
        )

    raw_groups = _group_basis_names_by_raw(basis_names)
    raw_names = tuple(raw_groups.keys())
    shap_per_raw = np.zeros((shap_per_basis.shape[0], len(raw_groups)), dtype=np.float64)
    for j, (_raw, indices) in enumerate(raw_groups.items()):
        shap_per_raw[:, j] = shap_per_basis[:, indices].sum(axis=1)

    return LinearAttributionResult(
        shap_per_basis=shap_per_basis,
        basis_feature_names=basis_names,
        shap_per_raw=shap_per_raw,
        raw_feature_names=raw_names,
        intercept=intercept,
    )


def _resolve_pipeline(lr_model: LRModel | Pipeline) -> Pipeline:
    """Return the underlying fitted preprocessing+clf Pipeline."""
    if isinstance(lr_model, LRModel):
        if not hasattr(lr_model, "pipeline_"):
            raise RuntimeError("LRModel must be fit before attribute_lr()")
        return lr_model.pipeline_
    if isinstance(lr_model, Pipeline):
        return lr_model
    raise TypeError(f"lr_model must be LRModel or sklearn Pipeline; got {type(lr_model).__name__}")


def _group_basis_names_by_raw(basis_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Map post-preprocessing column names to raw HFP feature names.

    Three patterns the LR pipeline emits, in priority order:

    1. ``"<raw>_rcs<k>"`` -- the k-th RCS basis term of a continuous
       feature. Maps to the raw continuous name (e.g. ``Age_rcs1``
       -> ``Age``).
    2. ``"<raw>_<level>"`` -- one-hot dummy of a categorical (sklearn
       OneHotEncoder default with ``verbose_feature_names_out=False``
       writes ``ChestPainType_ATA`` etc). Maps to the raw categorical
       name. We rely on the categorical column list from
       :mod:`cardiorisk.data.preprocess` to disambiguate.
    3. Plain raw name -- numeric continuous (linear basis term),
       binary numeric, or missingness indicator. Identity map.

    The returned dict preserves the *first occurrence* order of each
    raw feature, which gives a stable column ordering for the summed-
    back SHAP matrix.
    """
    from cardiorisk.data.preprocess import (
        BINARY_NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )
    from cardiorisk.explainability.encoder import INDICATOR_COLUMNS

    cat_set = set(CATEGORICAL_COLUMNS)
    num_set = set(NUMERIC_COLUMNS)
    bin_set = set(BINARY_NUMERIC_COLUMNS)
    ind_set = set(INDICATOR_COLUMNS)

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(basis_names):
        raw = _basis_name_to_raw(
            name,
            categorical_names=cat_set,
            numeric_names=num_set,
            binary_names=bin_set,
            indicator_names=ind_set,
        )
        groups.setdefault(raw, []).append(idx)
    return groups


def _basis_name_to_raw(
    name: str,
    *,
    categorical_names: set[str],
    numeric_names: set[str],
    binary_names: set[str],
    indicator_names: set[str],
) -> str:
    """Return the raw HFP feature name for one post-preprocessing column.

    Handles four name patterns:

    1. Direct match: ``"Age"`` -> ``"Age"``, ``"FastingBS"`` -> ``"FastingBS"``,
       ``"RestingBP_was_missing"`` -> identity. Numerics, binaries, indicators.
    2. RCS basis with raw name preserved: ``"Age_rcs1"`` -> ``"Age"``.
    3. RCS basis with positional fallback: ``"x0"`` / ``"x0_rcs1"`` ->
       ``NUMERIC_COLUMNS[0]``. The LR pipeline's nested SimpleImputer +
       RCS Pipeline drops the input column name metadata when the
       imputer returns a numpy array (sklearn ``set_output(transform=
       "pandas")`` is not enabled by ADR-008), so the RCS transformer
       falls back to ``x0..xN``. The mapping is positional and
       deterministic because :func:`make_lr_pipeline` configures the
       ``rcs_continuous`` transformer with ``NUMERIC_COLUMNS`` in
       that exact order.
    4. Categorical OHE dummy: ``"ChestPainType_ATA"`` -> ``"ChestPainType"``.
       Longest-prefix match disambiguates ``ExerciseAngina_Y`` vs
       ``ExerciseAngina_was_missing``.
    """
    from cardiorisk.data.preprocess import NUMERIC_COLUMNS

    if name in numeric_names | binary_names | indicator_names:
        return name

    if RCS_BASIS_SUFFIX_PREFIX in name:
        head, _ = name.rsplit(RCS_BASIS_SUFFIX_PREFIX, 1)
        if head in numeric_names:
            return head
        # Positional fallback: x<i>_rcs<k> -> NUMERIC_COLUMNS[i].
        if head.startswith("x") and head[1:].isdigit():
            idx = int(head[1:])
            if 0 <= idx < len(NUMERIC_COLUMNS):
                return NUMERIC_COLUMNS[idx]

    # Linear-term positional fallback: x<i> -> NUMERIC_COLUMNS[i].
    if name.startswith("x") and name[1:].isdigit():
        idx = int(name[1:])
        if 0 <= idx < len(NUMERIC_COLUMNS):
            return NUMERIC_COLUMNS[idx]

    candidates = sorted(
        (c for c in categorical_names if name.startswith(f"{c}_")),
        key=len,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise ValueError(
        f"could not map basis name {name!r} to any raw HFP feature; "
        "did the LR preprocessing pipeline change shape?"
    )
