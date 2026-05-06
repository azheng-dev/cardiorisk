"""Population Stability Index (PSI) primitives.

PSI is the headline drift metric for Phase 2.6 (per ADR-014). For two
distributions over the same set of bins, with reference proportions
``p_ref`` and current proportions ``p_cur``,

.. math::

    \\mathrm{PSI} = \\sum_i (p_{\\mathrm{cur},i} - p_{\\mathrm{ref},i})
                          \\,\\ln\\!\\bigl(p_{\\mathrm{cur},i} / p_{\\mathrm{ref},i}\\bigr)

Industry convention (used everywhere from credit-risk models to
post-deployment monitoring tutorials) bands the result as:

- ``< 0.10``  → ``stable``
- ``0.10..0.25`` → ``moderate``
- ``>= 0.25`` → ``major``

ADR-014 explicitly notes that these thresholds are convention, not
proven from first principles, and surfaces the bin-count sensitivity in
the research doc.

Two helpers:

- :func:`psi_numeric` takes raw numeric arrays and a pre-computed
  edge vector (built once on the reference distribution by
  :mod:`cardiorisk.monitoring.reference`). Out-of-range current values
  are clipped into the outer bins, which mirrors the standard PSI
  treatment for production use (a value beyond the reference range still
  contributes to drift, it doesn't silently disappear).
- :func:`psi_categorical` takes per-level frequency dicts. Levels in
  ``cur`` not seen in ``ref`` are treated as a separate level with
  ``p_ref = epsilon`` (the standard ε-floor); levels in ``ref`` not seen
  in ``cur`` analogously get ``p_cur = epsilon``.

The ``epsilon`` floor (default ``1e-6``) prevents the ``log(0)``
singularity an empty reference or current bin would otherwise produce.
ADR-014 §"PSI hygiene" documents the choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

#: Default ε-floor for empty bins. Standard PSI convention.
DEFAULT_EPSILON: Final[float] = 1e-6

#: Severity band cut-points (inclusive lower bound on each band).
PSI_STABLE_MAX: Final[float] = 0.10
PSI_MODERATE_MAX: Final[float] = 0.25

SeverityBand = Literal["stable", "moderate", "major"]


def severity_band(psi: float) -> SeverityBand:
    """Industry-convention PSI severity classification.

    NaN PSI is reported as ``"major"`` so a degenerate computation
    (e.g. zero-variance reference) doesn't silently look ``"stable"``.
    """
    if not np.isfinite(psi):
        return "major"
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_MODERATE_MAX:
        return "moderate"
    return "major"


def _proportions_from_bins(
    counts: npt.NDArray[np.int64], epsilon: float
) -> npt.NDArray[np.float64]:
    """Convert raw bin counts to proportions with an ε-floor for empty bins.

    The ε-floor is applied *after* normalisation so the floored vector
    no longer sums to 1.0; this is the standard PSI treatment (the
    formula does not require normalised inputs, only that p_ref and
    p_cur are commensurate, which they are).
    """
    total = int(counts.sum())
    if total == 0:
        # Degenerate input: caller asked for proportions over an empty
        # sample. Return all-epsilon so the PSI formula is finite (the
        # PSI value will be 0 vs an identical all-epsilon reference,
        # which is the right answer: "no information either way").
        return np.full_like(counts, epsilon, dtype=np.float64)
    p = counts.astype(np.float64) / total
    return np.where(p == 0, epsilon, p)


def psi_from_proportions(
    p_ref: npt.NDArray[np.float64],
    p_cur: npt.NDArray[np.float64],
) -> float:
    """PSI from two pre-computed proportion vectors over the same bins.

    Both inputs must already have any ε-floor applied (so ``log(p)`` is
    finite for every bin). This split-out helper is what
    :func:`psi_numeric` and :func:`psi_categorical` end up calling; it
    is exposed for tests that want to assert closed-form behaviour
    without going through the binning machinery.
    """
    if p_ref.shape != p_cur.shape:
        raise ValueError(
            f"reference and current proportion vectors must align: {p_ref.shape} vs {p_cur.shape}"
        )
    if p_ref.size == 0:
        raise ValueError("empty proportion vectors")
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_from_counts(
    *,
    reference_counts: npt.NDArray[np.int64],
    current_counts: npt.NDArray[np.int64],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI from two integer count vectors over the same bins.

    Used by :mod:`cardiorisk.monitoring.drift` so the persisted reference
    bin counts in :class:`~cardiorisk.monitoring.reference.NumericReference`
    can be reused without re-binning. Both empty inputs short-circuit to
    0.0 (no signal in either direction).
    """
    if reference_counts.shape != current_counts.shape:
        raise ValueError(
            f"reference and current count vectors must align: "
            f"{reference_counts.shape} vs {current_counts.shape}"
        )
    if reference_counts.size == 0:
        return 0.0
    if reference_counts.sum() == 0 or current_counts.sum() == 0:
        return 0.0
    p_ref = _proportions_from_bins(reference_counts, epsilon)
    p_cur = _proportions_from_bins(current_counts, epsilon)
    return psi_from_proportions(p_ref, p_cur)


def psi_numeric(
    *,
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    edges: npt.NDArray[np.float64],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI for a numeric feature given pre-computed reference bin edges.

    Parameters
    ----------
    reference, current : array-like
        Raw numeric samples. NaNs are dropped before binning (PSI is
        ill-defined on a "missing" bin; the missingness story is told
        separately by the existing ``<col>_was_missing`` indicators).
    edges : ndarray, shape (n_bins + 1,)
        Bin edges, normally produced by
        :func:`cardiorisk.monitoring.reference.compute_quantile_edges`
        on the reference sample. Length-2 edges (single bin) is allowed
        and yields PSI = 0 by construction.
    epsilon : float
        ε-floor for empty bins. Default :data:`DEFAULT_EPSILON`.

    Returns
    -------
    float
        Non-negative PSI. NaN is impossible by construction once the
        ε-floor is applied; an empty reference or empty current sample
        returns 0.0 (no signal in either direction).

    Notes
    -----
    Out-of-range current values are clipped into the outer bins by
    ``np.histogram`` semantics: we use ``np.searchsorted`` with the
    ``"right"`` side and clip the resulting index into ``[0, n_bins-1]``.
    This mirrors what most production PSI implementations do — the alternative
    (silently dropping out-of-range values) hides drift.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"edges must be a 1-D array of length >= 2, got shape {edges.shape}")
    n_bins = int(edges.size) - 1

    if ref.size == 0 or cur.size == 0:
        return 0.0

    # Bin counts. Use np.searchsorted + clip so out-of-range values land
    # in the outer bins (rather than being dropped by np.histogram, which
    # is the historical PSI-implementation footgun).
    ref_idx = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, n_bins - 1)
    cur_idx = np.clip(np.searchsorted(edges, cur, side="right") - 1, 0, n_bins - 1)
    ref_counts = np.bincount(ref_idx, minlength=n_bins).astype(np.int64)
    cur_counts = np.bincount(cur_idx, minlength=n_bins).astype(np.int64)

    p_ref = _proportions_from_bins(ref_counts, epsilon)
    p_cur = _proportions_from_bins(cur_counts, epsilon)
    return psi_from_proportions(p_ref, p_cur)


def psi_categorical(
    *,
    reference_counts: Mapping[str, int],
    current_counts: Mapping[str, int],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI for a categorical feature given per-level frequency dicts.

    The two dicts need not share the same key set; the union is used,
    with missing keys assigned a count of zero (and therefore the
    ε-floor proportion). This is how novel-level drift in the current
    slice is surfaced — a category that did not exist in the reference
    contributes positively to PSI.

    Empty inputs return 0.0 (no signal). Identical level-frequency
    distributions return 0.0 exactly (modulo floating-point noise
    well below 1e-12 in practice).
    """
    levels = sorted({*reference_counts.keys(), *current_counts.keys()})
    if not levels:
        return 0.0
    ref_counts = np.asarray([reference_counts.get(level, 0) for level in levels], dtype=np.int64)
    cur_counts = np.asarray([current_counts.get(level, 0) for level in levels], dtype=np.int64)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return 0.0
    p_ref = _proportions_from_bins(ref_counts, epsilon)
    p_cur = _proportions_from_bins(cur_counts, epsilon)
    return psi_from_proportions(p_ref, p_cur)
