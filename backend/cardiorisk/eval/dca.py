"""Decision-Curve Analysis (DCA) per Vickers & Elkin (2006).

DCA quantifies *clinical utility* of a probabilistic prediction model
across the range of decision thresholds a clinician might adopt. For a
single threshold ``p_t``, the **net benefit** is::

    NB(p_t) = TP/N - (FP/N) * (p_t / (1 - p_t))

where ``TP`` and ``FP`` are computed by treating every patient with
predicted probability >= ``p_t`` as a positive prediction. The first
term is the rate of *correctly* identified positives; the second
penalises false positives by the harm-to-benefit ratio implied by the
threshold (``p_t / (1 - p_t)``).

Two reference policies bound the model's performance:

- **Treat all**: every patient is treated as positive. ``NB = prevalence -
  (1 - prevalence) * (p_t / (1 - p_t))``. Falls below zero quickly as
  ``p_t`` increases.
- **Treat none**: no one treated. ``NB = 0`` everywhere. Trivial.

A model is *clinically useful* at threshold ``p_t`` iff its net benefit
exceeds **both** treat-all and treat-none at that ``p_t``. Per
:doc:`../../../docs/research/04-revised-design.md` §5.1 we report net
benefit at the AusCVDRisk thresholds ``p_t = 5%`` (low/intermediate
boundary) and ``p_t = 10%`` (intermediate/high boundary).

Implementation: ~60 lines of formula + 30 lines of dataclass. We do not
pull the ``dcurves`` package — the math is short, the citation is the
documentation, and rolling our own keeps the dependency graph honest.

References
----------
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
  evaluating prediction models. Med Decis Making. 2006;26(6):565-574.
  https://pubmed.ncbi.nlm.nih.gov/17099194/
- Vickers AJ et al. A simple, step-by-step guide to interpreting
  decision curve analysis. Diagn Progn Res. 2019;3:18.
  https://diagnprognres.biomedcentral.com/articles/10.1186/s41512-019-0064-7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: Default threshold sweep for :func:`decision_curve` — every percent
#: from 1% to 99%. AusCVDRisk decision thresholds (5% and 10%) are
#: included.
DEFAULT_THRESHOLDS: Final[np.ndarray] = np.arange(0.01, 1.00, 0.01)

#: AusCVDRisk-aligned reporting thresholds called out in the design doc.
AUSCVDRISK_THRESHOLDS: Final[tuple[float, ...]] = (0.05, 0.10)


def _validate_inputs(
    y_true: npt.ArrayLike, y_proba: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_proba shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("y_true is empty")
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0 and 1; got {unique.tolist()}")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("y_proba must be in [0, 1]")
    return y.astype(np.int64, copy=False), p


def net_benefit(y_true: npt.ArrayLike, y_proba: npt.ArrayLike, threshold: float) -> float:
    """Net benefit of the model at a single ``threshold``.

    Treat patients with ``y_proba >= threshold`` as positive predictions,
    then::

        NB = TP/N - (FP/N) * (threshold / (1 - threshold))
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")
    y, p = _validate_inputs(y_true, y_proba)
    n = y.size
    predicted_positive = p >= threshold
    tp = int(((predicted_positive) & (y == 1)).sum())
    fp = int(((predicted_positive) & (y == 0)).sum())
    odds = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * odds


def net_benefit_treat_all(prevalence: float, threshold: float) -> float:
    """Net benefit of the 'treat everyone' policy at ``threshold``."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1]; got {prevalence}")
    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


@dataclass(frozen=True)
class DCACurve:
    """Net-benefit curves for the model and the two reference policies."""

    thresholds: np.ndarray
    net_benefit_model: np.ndarray
    net_benefit_treat_all: np.ndarray
    net_benefit_treat_none: np.ndarray  # always zero, kept for plotting symmetry

    def at(self, threshold: float) -> dict[str, float]:
        """Convenience: net benefit of all three policies at one threshold."""
        idx = int(np.argmin(np.abs(self.thresholds - threshold)))
        return {
            "model": float(self.net_benefit_model[idx]),
            "treat_all": float(self.net_benefit_treat_all[idx]),
            "treat_none": float(self.net_benefit_treat_none[idx]),
        }

    def is_useful_at(self, threshold: float) -> bool:
        """Model dominates both reference policies at this threshold."""
        nb = self.at(threshold)
        return nb["model"] > nb["treat_all"] and nb["model"] > nb["treat_none"]


def decision_curve(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    thresholds: npt.ArrayLike | None = None,
) -> DCACurve:
    """Compute the full decision curve (model + treat-all + treat-none).

    ``thresholds`` defaults to :data:`DEFAULT_THRESHOLDS` (1%-99%, step 1%).
    """
    y, p = _validate_inputs(y_true, y_proba)
    t = np.asarray(thresholds, dtype=np.float64) if thresholds is not None else DEFAULT_THRESHOLDS
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("all thresholds must be in (0, 1)")

    prevalence = float(y.mean())
    nb_model = np.array([net_benefit(y, p, float(ti)) for ti in t])
    nb_treat_all = np.array([net_benefit_treat_all(prevalence, float(ti)) for ti in t])
    nb_treat_none = np.zeros_like(t)
    return DCACurve(
        thresholds=t,
        net_benefit_model=nb_model,
        net_benefit_treat_all=nb_treat_all,
        net_benefit_treat_none=nb_treat_none,
    )
