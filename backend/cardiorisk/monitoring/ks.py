"""Two-sample Kolmogorov-Smirnov drift sanity check (numeric features only).

PSI is the Phase-2.6 headline metric (per ADR-014); KS is a sanity-only
companion that surfaces a *significance* lens (p-value) which PSI alone
does not provide. Categorical features are skipped: KS is defined for
ordered distributions.

Thin wrapper around :func:`scipy.stats.ks_2samp`; the only value-add
over the bare scipy call is NaN handling and a single typed dataclass
return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats


@dataclass(frozen=True)
class KSResult:
    """One numeric two-sample KS test outcome."""

    statistic: float
    p_value: float
    n_ref: int
    n_cur: int


def ks_two_sample(
    *,
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
) -> KSResult:
    """Two-sample KS test on two numeric samples.

    NaNs are dropped from both inputs before the test (KS is undefined
    on missing values). If either dropped sample is empty, the test is
    not run and the result is reported as ``statistic=0, p_value=1`` —
    the "no evidence of drift" outcome — with the post-NaN-drop sample
    sizes preserved on the return so the caller can detect the
    degenerate case.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return KSResult(statistic=0.0, p_value=1.0, n_ref=int(ref.size), n_cur=int(cur.size))
    res = stats.ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return KSResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_ref=int(ref.size),
        n_cur=int(cur.size),
    )
