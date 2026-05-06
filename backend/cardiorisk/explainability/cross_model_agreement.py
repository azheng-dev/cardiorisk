"""Cross-model SHAP feature-importance agreement (Spearman rank).

ADR-013 §"Cross-model agreement" + Phase 2.4 §8 Q4 ("do the four
models think different features matter?"). For each fold, build a
``(n_models, n_features)`` matrix of mean |SHAP| values per
(model, raw HFP feature), then compute the per-fold Spearman rank
correlation matrix across the model rows. Aggregate by averaging the
Spearman matrices across folds.

The matrix entries are correlations of *ranks* (which features each
model considers most important) rather than raw values, so the result
is invariant to the SHAP value's output scale (probability for
TabICL, log-odds for the other three after Platt). This is the
honest cross-model question -- "do the models agree on the ordering
of features by importance?" -- not "do they assign the same numbers
to feature attribution".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats


@dataclass(frozen=True)
class AgreementResult:
    """Spearman matrix of cross-model feature-importance agreement."""

    model_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    spearman_matrix: npt.NDArray[np.float64]


def compute_cross_model_agreement(
    *,
    mean_abs_per_model: Mapping[str, Mapping[str, float]],
) -> AgreementResult:
    """Per-fold Spearman rank-correlation matrix across the model rows.

    Parameters
    ----------
    mean_abs_per_model
        ``{model_name: {feature_name: mean |SHAP|}}``. All models must
        share the same set of feature names.
    """
    if not mean_abs_per_model:
        raise ValueError("mean_abs_per_model must not be empty")

    model_names = tuple(mean_abs_per_model.keys())
    first_features = tuple(next(iter(mean_abs_per_model.values())).keys())
    for m, d in mean_abs_per_model.items():
        if tuple(d.keys()) != first_features:
            raise ValueError(
                f"feature names mismatch for model {m!r}; "
                f"expected {first_features} got {tuple(d.keys())}"
            )

    matrix = np.array(
        [[d[f] for f in first_features] for d in mean_abs_per_model.values()],
        dtype=np.float64,
    )

    n_models = matrix.shape[0]
    spearman = np.eye(n_models, dtype=np.float64)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rho, _ = stats.spearmanr(matrix[i], matrix[j])
            spearman[i, j] = float(rho)
            spearman[j, i] = float(rho)

    return AgreementResult(
        model_names=model_names,
        feature_names=first_features,
        spearman_matrix=spearman,
    )


def aggregate_across_folds(
    *,
    per_fold: Sequence[AgreementResult],
) -> AgreementResult:
    """Mean of per-fold Spearman matrices, with the model + feature axes
    asserted identical.

    NaN entries (which happen if a fold's mean |SHAP| is constant
    across features for some model) are ignored via ``nanmean``.
    """
    if not per_fold:
        raise ValueError("per_fold must not be empty")

    model_names = per_fold[0].model_names
    feature_names = per_fold[0].feature_names
    for r in per_fold[1:]:
        if r.model_names != model_names:
            raise ValueError("model_names differ across folds")
        if r.feature_names != feature_names:
            raise ValueError("feature_names differ across folds")

    stack = np.stack([r.spearman_matrix for r in per_fold], axis=0)
    mean_matrix = np.nanmean(stack, axis=0)
    return AgreementResult(
        model_names=model_names,
        feature_names=feature_names,
        spearman_matrix=mean_matrix,
    )
