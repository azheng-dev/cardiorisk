"""Pick representative test patients for the local-explanation gallery.

ADR-013 §"Local-explanation gallery (per model x fold)" specifies four
archetypes per (model, fold) combination:

- **TP-high** -- highest predicted-risk correctly-predicted positive.
- **TP-low** -- lowest predicted-risk correctly-predicted positive
  (closest call that came out correct).
- **FN** -- lowest-confidence missed positive (most over-confident
  error: model predicted low risk, ground truth was positive).
- **FP** -- highest-confidence false alarm (model predicted high
  risk, ground truth was negative).

The four-archetype framing is the standard "where does the model
agree and disagree with reality" surface; it's how clinicians read
individual risk scores.

Decision threshold for TP/FN/FP/TN classification is fixed at 0.5
(the prediction is "high risk" iff the calibrated probability is
>= 0.5). This is *not* the same as the operating points the eval
harness reports (sens@85% spec, sens@90% spec) -- it is the
informational threshold for archetype selection. Per ADR-013 the
clinically-relevant thresholds (5% / 10% AusCVDRisk) are not used
here because they are below 0.5 and would treat almost every test
row as "high risk", which collapses the FP / TN distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Decision threshold for archetype classification.
DECISION_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class Archetype:
    """One representative test row + the model's prediction for it.

    Attributes
    ----------
    label : str
        One of ``"tp_high"``, ``"tp_low"``, ``"fn"``, ``"fp"``.
    test_index : int
        Index into the test slice (0-based), suitable for
        ``X_test.iloc[archetype.test_index]``.
    y_true : int
        Ground-truth label (0 or 1).
    y_proba : float
        Calibrated positive-class probability.
    """

    label: str
    test_index: int
    y_true: int
    y_proba: float


def pick_archetypes(
    *,
    y_true: npt.NDArray[np.int64] | npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    threshold: float = DECISION_THRESHOLD,
) -> list[Archetype]:
    """Pick the four representative test rows per ADR-013.

    Returns up to four :class:`Archetype` instances. Any archetype
    whose population is empty (e.g. a model that never produces a
    false positive on a particular fold) is silently omitted; the
    caller should expect *up to* 4 entries, not exactly 4.

    Parameters
    ----------
    y_true, y_proba
        Aligned arrays. Lengths must match.
    threshold
        Decision threshold for high/low risk classification. Default
        :data:`DECISION_THRESHOLD` = 0.5 per ADR-013.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_true.shape != y_proba.shape:
        raise ValueError(f"y_true and y_proba must align; got {y_true.shape} vs {y_proba.shape}")
    if y_true.size == 0:
        return []

    pred_positive = y_proba >= threshold
    tp_mask = (y_true == 1) & pred_positive
    fn_mask = (y_true == 1) & ~pred_positive
    fp_mask = (y_true == 0) & pred_positive

    out: list[Archetype] = []

    out.extend(
        _pick_one(
            label="tp_high",
            mask=tp_mask,
            y_true=y_true,
            y_proba=y_proba,
            select="argmax",
        )
    )
    out.extend(
        _pick_one(
            label="tp_low",
            mask=tp_mask,
            y_true=y_true,
            y_proba=y_proba,
            select="argmin",
        )
    )
    out.extend(
        _pick_one(
            label="fn",
            mask=fn_mask,
            y_true=y_true,
            y_proba=y_proba,
            select="argmin",
        )
    )
    out.extend(
        _pick_one(
            label="fp",
            mask=fp_mask,
            y_true=y_true,
            y_proba=y_proba,
            select="argmax",
        )
    )
    return out


def _pick_one(
    *,
    label: str,
    mask: npt.NDArray[np.bool_],
    y_true: npt.NDArray[np.int64],
    y_proba: npt.NDArray[np.float64],
    select: str,
) -> list[Archetype]:
    """Return ``[Archetype]`` if ``mask`` has any rows, else ``[]``."""
    if not mask.any():
        return []
    candidates = np.flatnonzero(mask)
    proba_subset = y_proba[candidates]
    if select == "argmax":
        local_idx = int(np.argmax(proba_subset))
    elif select == "argmin":
        local_idx = int(np.argmin(proba_subset))
    else:
        raise ValueError(f"select must be 'argmax' or 'argmin'; got {select!r}")
    test_idx = int(candidates[local_idx])
    return [
        Archetype(
            label=label,
            test_index=test_idx,
            y_true=int(y_true[test_idx]),
            y_proba=float(y_proba[test_idx]),
        )
    ]
