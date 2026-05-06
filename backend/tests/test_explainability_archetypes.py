"""Tests for :mod:`cardiorisk.explainability.archetypes`."""

from __future__ import annotations

import numpy as np
import pytest

from cardiorisk.explainability.archetypes import (
    DECISION_THRESHOLD,
    Archetype,
    pick_archetypes,
)


def test_pick_all_four_archetypes_present() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.3, 0.55, 0.85, 0.4, 0.05, 0.7])
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    labels = {a.label for a in out}
    assert labels == {"tp_high", "tp_low", "fn", "fp"}


def test_tp_high_picks_argmax_proba_among_tps() -> None:
    y_true = np.array([1, 1, 1, 0])
    y_proba = np.array([0.9, 0.6, 0.55, 0.4])
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    tp_high = next(a for a in out if a.label == "tp_high")
    assert tp_high.test_index == 0
    assert tp_high.y_proba == 0.9
    assert tp_high.y_true == 1


def test_tp_low_picks_argmin_proba_among_tps() -> None:
    y_true = np.array([1, 1, 1, 0])
    y_proba = np.array([0.9, 0.6, 0.55, 0.4])
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    tp_low = next(a for a in out if a.label == "tp_low")
    assert tp_low.test_index == 2
    assert tp_low.y_proba == 0.55


def test_fn_picks_argmin_proba_among_fns() -> None:
    """FN = predicted-low ground-truth-positive; pick the most over-confident."""
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.9, 0.3, 0.1])
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    fn = next(a for a in out if a.label == "fn")
    assert fn.test_index == 2
    assert fn.y_proba == 0.1


def test_fp_picks_argmax_proba_among_fps() -> None:
    """FP = predicted-high ground-truth-negative; pick the highest-confidence false alarm."""
    y_true = np.array([0, 0, 0])
    y_proba = np.array([0.7, 0.85, 0.4])
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    fp = next(a for a in out if a.label == "fp")
    assert fp.test_index == 1
    assert fp.y_proba == 0.85


def test_missing_archetype_silently_skipped() -> None:
    """A model that never produces a false positive returns 3 archetypes, not 4."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.3, 0.2, 0.1])  # no FP (no negatives predicted high)
    out = pick_archetypes(y_true=y_true, y_proba=y_proba)
    labels = {a.label for a in out}
    assert "fp" not in labels
    assert {"tp_high", "fn"}.issubset(labels)


def test_empty_inputs_returns_empty_list() -> None:
    out = pick_archetypes(y_true=np.array([]), y_proba=np.array([]))
    assert out == []


def test_threshold_is_05_by_default() -> None:
    assert DECISION_THRESHOLD == 0.5


def test_archetype_dataclass_fields() -> None:
    a = Archetype(label="tp_high", test_index=3, y_true=1, y_proba=0.9)
    assert a.label == "tp_high"
    assert a.test_index == 3
    assert a.y_true == 1
    assert a.y_proba == 0.9


def test_pick_archetypes_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        pick_archetypes(y_true=np.array([1, 0]), y_proba=np.array([0.5]))
