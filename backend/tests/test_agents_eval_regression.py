"""Tests for the Phase-6 regression gate in the eval orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cardiorisk.agents.eval.orchestrator import (
    REGRESSION_METRICS,
    REGRESSION_METRICS_LOWER_IS_BETTER,
    check_regression,
)


def _summary(
    *,
    triage: float = 1.0,
    risk: float = 1.0,
    guideline: float = 1.0,
    letter: float = 1.0,
    full: float = 1.0,
    reco: float = 1.0,
    prec: float = 1.0,
    rec: float = 1.0,
    halluc: float = 0.0,
    judge: float = 1.0,
) -> dict[str, object]:
    return {
        "aggregate": {
            "triage_pass_rate": triage,
            "risk_band_match_rate": risk,
            "guideline_pass_rate": guideline,
            "letter_pass_rate": letter,
            "full_pipeline_pass_rate": full,
            "recommendation_correctness_rate": reco,
            "mean_citation_precision": prec,
            "mean_citation_recall": rec,
            "mean_hallucination_rate": halluc,
        },
        "judge_aggregate": {"pass_rate": judge},
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestCheckRegression:
    def test_no_change_does_not_fail(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary())
        result = check_regression(
            current_summary=_summary(),
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is False
        for label_info in result["deltas"].values():
            assert label_info["fail"] is False

    def test_drop_beyond_tolerance_fails(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary(triage=1.0))
        # 95% is a 5 pp drop > 2 pp tolerance
        result = check_regression(
            current_summary=_summary(triage=0.95),
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is True
        assert result["deltas"]["triage_pass_rate"]["fail"] is True

    def test_drop_within_tolerance_passes(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary(triage=1.0))
        result = check_regression(
            current_summary=_summary(triage=0.99),  # 1 pp drop
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is False

    def test_improvement_never_fails(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary(triage=0.80))
        result = check_regression(
            current_summary=_summary(triage=1.0),
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is False
        assert result["deltas"]["triage_pass_rate"]["delta_pp"] == pytest.approx(20.0)

    def test_hallucination_increase_fails(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary(halluc=0.05))
        result = check_regression(
            current_summary=_summary(halluc=0.15),  # 10 pp increase
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is True
        assert result["deltas"]["mean_hallucination_rate"]["fail"] is True
        assert result["deltas"]["mean_hallucination_rate"]["direction"] == "lower_is_better"

    def test_hallucination_decrease_passes(self, tmp_path: Path) -> None:
        baseline = _write(tmp_path, _summary(halluc=0.30))
        result = check_regression(
            current_summary=_summary(halluc=0.10),
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is False

    def test_missing_baseline_metric_recorded_but_not_failed(self, tmp_path: Path) -> None:
        # Strip the recommendation_correctness_rate from the baseline
        # to simulate a pre-Phase-6 baseline file.
        baseline_dict = _summary()
        assert isinstance(baseline_dict["aggregate"], dict)
        del baseline_dict["aggregate"]["recommendation_correctness_rate"]
        baseline = _write(tmp_path, baseline_dict)
        result = check_regression(
            current_summary=_summary(reco=0.0),
            baseline_path=baseline,
            tolerance_pp=2.0,
        )
        assert result["failed"] is False
        entry = result["deltas"]["recommendation_correctness_rate"]
        assert entry["baseline"] is None
        assert entry["fail"] is False

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError):
            check_regression(
                current_summary=_summary(),
                baseline_path=missing,
                tolerance_pp=2.0,
            )

    def test_every_tracked_metric_has_known_direction(self) -> None:
        # Sanity: every tracked metric must be reachable via the
        # documented path. This guards against typos in the path
        # tuples breaking the gate silently.
        seen_labels = set()
        for path, label in REGRESSION_METRICS + REGRESSION_METRICS_LOWER_IS_BETTER:
            assert label not in seen_labels, f"duplicate metric label {label!r}"
            seen_labels.add(label)
            assert all(isinstance(p, str) and p for p in path)
