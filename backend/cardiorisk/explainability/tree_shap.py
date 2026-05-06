"""TreeSHAP wrapper for XGBoost (Phase 2.5 sanity check).

ADR-013 §"Explainer per model" runs TreeSHAP additionally for
XGBoost. The cross-model headline is KernelSHAP-everywhere; TreeSHAP
is the *fast, exact* native algorithm for tree ensembles, used here
to sanity-check the KernelSHAP values via a side-by-side scatter
(``xgboost_<fold>_treeshap_vs_kernelshap.png``). If they agree
closely (Spearman rank correlation >= 0.85 by feature) the cross-model
KernelSHAP surface is reinforced; if they disagree substantially that
is itself a finding the model card flags.

Implementation notes:

- The on-disk artefact is ``CalibratedClassifierCV(FrozenEstimator(XGBoostModel))``.
  We dig in for the underlying XGBoost ``Booster`` (lives at
  ``cal.estimator.estimator.pipeline_.named_steps['clf'].booster_``
  via the path in :mod:`cardiorisk.models.xgboost_model`).
- TreeSHAP operates on the post-XGBoost-preprocessing matrix (the
  imputed + one-hot-encoded numeric matrix the booster actually
  trained on). We aggregate per-feature SHAP values back to raw HFP
  feature names using a name-to-raw mapping analogous to the LR
  attribution module's :func:`_basis_name_to_raw`.
- We explain the *uncalibrated* XGBoost output (log-odds), not the
  Platt-calibrated probability. This is the right comparison surface
  for "how does TreeSHAP-on-the-booster compare to KernelSHAP-on-the-
  calibrated-pipeline" because the Platt step is just a 1-d monotone
  rescaling -- it cannot change the per-feature SHAP *ranking*, only
  the absolute scale. The rank-correlation sanity check is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from cardiorisk.models.xgboost_model import XGBoostModel


@dataclass(frozen=True)
class TreeSHAPResult:
    """Per-row TreeSHAP values in raw HFP feature space."""

    shap_values_post: npt.NDArray[np.float64]
    post_feature_names: tuple[str, ...]

    shap_values_raw: npt.NDArray[np.float64]
    raw_feature_names: tuple[str, ...]

    expected_value: float

    @property
    def mean_abs_per_raw_feature(self) -> dict[str, float]:
        """Mean |log-odds SHAP| per raw HFP feature."""
        means = np.mean(np.abs(self.shap_values_raw), axis=0)
        return dict(zip(self.raw_feature_names, (float(m) for m in means), strict=True))


def explain_xgboost_with_tree_shap(
    *,
    calibrated_or_bare_xgb: BaseEstimator | XGBoostModel,
    X_test: pd.DataFrame,
) -> TreeSHAPResult:
    """Run TreeSHAP on the underlying XGBoost booster.

    Accepts either:

    - a fitted :class:`~cardiorisk.models.xgboost_model.XGBoostModel`,
    - a fitted preprocessing-+-XGB sklearn :class:`Pipeline`,
    - the calibrated artefact
      ``CalibratedClassifierCV(FrozenEstimator(XGBoostModel))`` that
      Phase 2.4's training driver writes to ``models/v1/``.
    """
    xgb_model = _resolve_xgboost(calibrated_or_bare_xgb)
    pipeline = xgb_model.pipeline_

    preprocess = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]

    X_post = np.asarray(preprocess.transform(X_test), dtype=np.float64)

    explainer = shap.TreeExplainer(clf)
    raw_values = explainer.shap_values(X_post)
    # XGBoost binary-class trees emit shape (n, n_features) for
    # log-odds output. Some shap versions return a list per class;
    # normalise.
    if isinstance(raw_values, list):
        # Take the positive class.
        raw_values = raw_values[1]
    shap_values_post = np.asarray(raw_values, dtype=np.float64)

    if shap_values_post.shape != X_post.shape:
        raise RuntimeError(
            f"TreeSHAP returned unexpected shape {shap_values_post.shape}; expected {X_post.shape}"
        )

    post_names = tuple(str(n) for n in preprocess.get_feature_names_out())

    raw_groups = _group_post_names_by_raw(post_names)
    raw_names = tuple(raw_groups.keys())
    shap_values_raw = np.zeros((shap_values_post.shape[0], len(raw_groups)), dtype=np.float64)
    for j, indices in enumerate(raw_groups.values()):
        shap_values_raw[:, j] = shap_values_post[:, indices].sum(axis=1)

    expected_value = float(np.asarray(explainer.expected_value).ravel()[0])
    return TreeSHAPResult(
        shap_values_post=shap_values_post,
        post_feature_names=post_names,
        shap_values_raw=shap_values_raw,
        raw_feature_names=raw_names,
        expected_value=expected_value,
    )


def _resolve_xgboost(obj: BaseEstimator | XGBoostModel) -> XGBoostModel:
    """Walk the calibration wrappers to find the fitted XGBoostModel."""
    if isinstance(obj, XGBoostModel):
        if not hasattr(obj, "pipeline_"):
            raise RuntimeError("XGBoostModel must be fit before TreeSHAP")
        return obj
    if isinstance(obj, CalibratedClassifierCV):
        # Sklearn 1.6+ stores per-fold base estimators; with FrozenEstimator
        # there's exactly one and it lives at
        # cal.calibrated_classifiers_[0].estimator (a FrozenEstimator),
        # which in turn wraps the XGBoostModel via .estimator.
        if not hasattr(obj, "calibrated_classifiers_"):
            raise RuntimeError("CalibratedClassifierCV is not fitted")
        inner = obj.calibrated_classifiers_[0].estimator
        if isinstance(inner, FrozenEstimator):
            return _resolve_xgboost(inner.estimator)
        return _resolve_xgboost(inner)
    if isinstance(obj, FrozenEstimator):
        return _resolve_xgboost(obj.estimator)
    raise TypeError(
        "could not extract XGBoostModel from "
        f"{type(obj).__name__}; expected XGBoostModel, FrozenEstimator, "
        "or CalibratedClassifierCV"
    )


def _group_post_names_by_raw(post_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Map XGBoost post-preprocessing column names to raw HFP feature names.

    The XGBoost pipeline (see :func:`cardiorisk.features.pipeline.make_xgboost_pipeline`)
    emits names with ``verbose_feature_names_out=False``:

    - One-hot dummies for categoricals (``ChestPainType_ATA`` etc.).
    - Continuous + binary numerics keep their raw HFP name.
    - Missingness indicators keep their raw ``<col>_was_missing`` name.

    Mapping rule mirrors :func:`linear_attribution._basis_name_to_raw`
    minus the RCS branch (XGB does not RCS-expand).
    """
    from cardiorisk.data.preprocess import (
        BINARY_NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )
    from cardiorisk.explainability.encoder import INDICATOR_COLUMNS

    pass_through_names = set(NUMERIC_COLUMNS) | set(BINARY_NUMERIC_COLUMNS) | set(INDICATOR_COLUMNS)

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(post_names):
        if name in pass_through_names:
            raw = name
        else:
            candidates = sorted(
                (c for c in CATEGORICAL_COLUMNS if name.startswith(f"{c}_")),
                key=len,
                reverse=True,
            )
            if not candidates:
                raise ValueError(f"could not map XGB post-name {name!r} to any raw HFP feature")
            raw = candidates[0]
        groups.setdefault(raw, []).append(idx)
    return groups
