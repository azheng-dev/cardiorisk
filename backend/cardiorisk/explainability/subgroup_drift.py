"""Per-stratum mean |SHAP| deltas for the subgroup-drift audit.

ADR-013 §"Subgroup-drift scope" restricts this to **auditable strata
only**:

- *Sex strata*: only Cleveland and Hungarian have F counts above the
  ``min_stratum_size`` guard (97 and 81 respectively); LongBeachVA
  F=6 and Switzerland F=10 are below the guard and are skipped.
- *Age strata*: only strata with n >= 50 in the test slice. Cleveland
  ``50-69`` (n=206) and ``<50`` (n=87); Hungarian ``50-69`` (n=133)
  and ``<50`` (n=161); LongBeachVA ``50-69`` (n=165); Switzerland
  ``50-69`` (n=93). The ``>=70`` stratum is below the guard on every
  fold (10 / 0 / 16 / 5).

This mirrors the discipline Phase 2.3b's
:func:`cardiorisk.eval.subgroup.stratified_metrics` applies to AUROC
gaps: low-n strata return NA rather than a noisy delta.

Output: per (model, fold, stratum, raw_feature) mean |SHAP|. The
"drift" is the absolute difference between the per-stratum mean and
the overall test-slice mean, sorted descending by |delta|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Per-stratum minimum n for an audit to run. Matches Phase 2.3b's
#: ``min_stratum_size`` default in
#: :mod:`cardiorisk.eval.subgroup`.
DEFAULT_MIN_STRATUM_SIZE: Final[int] = 30


@dataclass(frozen=True)
class StratumDrift:
    """Per-stratum mean |SHAP| profile + drift vs the overall test slice."""

    stratum: str
    n: int
    mean_abs_per_feature: dict[str, float]
    delta_per_feature: dict[str, float]


@dataclass(frozen=True)
class SubgroupDriftResult:
    """Per-(grouping x stratum) drift, with the "below-guard" decision recorded."""

    grouping_name: str
    overall_n: int
    overall_mean_abs_per_feature: dict[str, float]
    by_stratum: tuple[StratumDrift, ...]
    skipped_strata: tuple[tuple[str, int], ...]


def compute_subgroup_drift(
    *,
    grouping_name: str,
    grouping_values: npt.NDArray[np.object_],
    shap_values_raw: npt.NDArray[np.float64],
    raw_feature_names: tuple[str, ...],
    min_stratum_size: int = DEFAULT_MIN_STRATUM_SIZE,
) -> SubgroupDriftResult:
    """Compute per-stratum mean |SHAP| + drift vs the overall test slice.

    Parameters
    ----------
    grouping_name
        Human-readable name for the grouping ("sex", "age_band").
    grouping_values
        Per-row stratum label, length matches ``shap_values_raw.shape[0]``.
    shap_values_raw
        Per-row, per-raw-feature SHAP values (e.g. from
        :class:`KernelSHAPResult.shap_values_raw`).
    raw_feature_names
        Column names for ``shap_values_raw``.
    min_stratum_size
        Strata with fewer rows than this are skipped (recorded in
        :attr:`SubgroupDriftResult.skipped_strata`).
    """
    if shap_values_raw.shape[1] != len(raw_feature_names):
        raise ValueError(
            f"shap_values_raw has {shap_values_raw.shape[1]} cols vs "
            f"{len(raw_feature_names)} feature names"
        )
    if shap_values_raw.shape[0] != grouping_values.shape[0]:
        raise ValueError(
            f"shap_values_raw rows={shap_values_raw.shape[0]} but "
            f"grouping_values rows={grouping_values.shape[0]}"
        )

    overall_n = shap_values_raw.shape[0]
    overall_mean = np.mean(np.abs(shap_values_raw), axis=0)
    overall_mean_dict = dict(zip(raw_feature_names, (float(v) for v in overall_mean), strict=True))

    by_stratum: list[StratumDrift] = []
    skipped: list[tuple[str, int]] = []

    # Stable iteration order (sort categorical labels lexicographically
    # so per-fold JSONs read consistently).
    for stratum in sorted(np.unique(grouping_values).tolist()):
        mask = grouping_values == stratum
        n = int(mask.sum())
        if n < min_stratum_size:
            skipped.append((str(stratum), n))
            continue
        stratum_mean = np.mean(np.abs(shap_values_raw[mask]), axis=0)
        delta = stratum_mean - overall_mean
        stratum_mean_dict = dict(
            zip(raw_feature_names, (float(v) for v in stratum_mean), strict=True)
        )
        delta_dict = dict(zip(raw_feature_names, (float(v) for v in delta), strict=True))
        by_stratum.append(
            StratumDrift(
                stratum=str(stratum),
                n=n,
                mean_abs_per_feature=stratum_mean_dict,
                delta_per_feature=delta_dict,
            )
        )

    return SubgroupDriftResult(
        grouping_name=grouping_name,
        overall_n=overall_n,
        overall_mean_abs_per_feature=overall_mean_dict,
        by_stratum=tuple(by_stratum),
        skipped_strata=tuple(skipped),
    )
