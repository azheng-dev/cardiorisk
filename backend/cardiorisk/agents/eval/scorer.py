"""Per-case + aggregate scoring for the Phase-4 agent eval.

Scoring philosophy (mirrors Phase 3.3's discipline):

- Every metric is binary at the per-case level. The aggregate is a
  rate (e.g. ``triage_pass_rate = mean(triage_pass)``).
- The risk band is a coarse sanity gate, not a regression test —
  Phase 2.4 documented the LODO-fold-to-fold variance. The
  aggregate reports a per-band confusion matrix so a recruiter can
  see the structure of any miss-class.
- Latency numbers (per-stage + total wall-clock) are reported
  diagnostically alongside the band-match table; they're not in
  the pass/fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cardiorisk.agents.state import AgentState

from .loader import AgentEvalCase

# Stage names (kept in sync with AgentStage values).
STAGES = ("triage", "risk", "guideline", "letter")

# Band labels (kept in sync with RiskResult.risk_band literal).
BANDS = ("low", "intermediate", "high")


@dataclass(frozen=True)
class StageReport:
    """Per-stage pass/fail + small numeric diagnostics."""

    stage: str
    passed: bool
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseReport:
    """Per-case full report: 4 stage reports + meta."""

    id: str
    tag: str
    expected_risk_band: str
    observed_risk_band: str | None
    band_match: bool
    n_verified_claims: int
    n_suppressed_claims: int
    letter_word_count: int
    triage: StageReport
    risk: StageReport
    guideline: StageReport
    letter: StageReport
    total_duration_ms: float
    sanity_flags_observed: tuple[str, ...]
    sanity_flags_missing: tuple[str, ...]


def _stage_duration_ms(state: AgentState, stage: str) -> float:
    for entry in state.audit:
        if entry.stage.value == stage:
            return entry.duration_ms
    return 0.0


def _word_count(text: str) -> int:
    return len(text.split())


def score_case(case: AgentEvalCase, state: AgentState) -> CaseReport:
    """Score one case's terminal :class:`AgentState` against the case spec."""
    triage = state.triage
    risk = state.risk
    guideline = state.guideline
    letter = state.letter

    # ----- triage
    sanity_observed = tuple(triage.sanity_flags) if triage else ()
    sanity_missing = tuple(f for f in case.expected_sanity_flags if f not in sanity_observed)
    triage_pass = triage is not None and bool(triage.summary) and not sanity_missing
    triage_report = StageReport(
        stage="triage",
        passed=triage_pass,
        duration_ms=_stage_duration_ms(state, "triage"),
        detail={
            "summary_present": bool(triage and triage.summary),
            "expected_flags": list(case.expected_sanity_flags),
            "observed_flags": list(sanity_observed),
            "missing_expected_flags": list(sanity_missing),
        },
    )

    # ----- risk
    observed_band = risk.risk_band if risk else None
    band_match = bool(risk and risk.risk_band == case.expected_risk_band)
    risk_report = StageReport(
        stage="risk",
        passed=band_match,
        duration_ms=_stage_duration_ms(state, "risk"),
        detail={
            "expected_band": case.expected_risk_band,
            "observed_band": observed_band,
            "probability": risk.probability if risk else None,
            "model_artefact_present": (risk.model_artefact_present if risk else None),
        },
    )

    # ----- guideline
    n_verified = len(guideline.answer.verified_claims) if guideline else 0
    n_suppressed = len(guideline.answer.suppressed_claims) if guideline else 0
    guideline_pass = n_verified >= case.expected_min_verified_claims
    guideline_report = StageReport(
        stage="guideline",
        passed=guideline_pass,
        duration_ms=_stage_duration_ms(state, "guideline"),
        detail={
            "n_verified_claims": n_verified,
            "n_suppressed_claims": n_suppressed,
            "expected_min_verified_claims": case.expected_min_verified_claims,
            "is_refusal": guideline.answer.is_refusal if guideline else None,
        },
    )

    # ----- letter
    word_count = _word_count(letter.draft) if letter else 0
    letter_pass = word_count >= case.expected_letter_min_words
    letter_report = StageReport(
        stage="letter",
        passed=letter_pass,
        duration_ms=_stage_duration_ms(state, "letter"),
        detail={
            "word_count": word_count,
            "expected_min_words": case.expected_letter_min_words,
            "n_citations": len(letter.citations) if letter else 0,
            "n_redacted_claims": len(letter.redacted_claims) if letter else 0,
        },
    )

    total_ms = sum(
        r.duration_ms for r in (triage_report, risk_report, guideline_report, letter_report)
    )

    return CaseReport(
        id=case.id,
        tag=case.tag,
        expected_risk_band=case.expected_risk_band,
        observed_risk_band=observed_band,
        band_match=band_match,
        n_verified_claims=n_verified,
        n_suppressed_claims=n_suppressed,
        letter_word_count=word_count,
        triage=triage_report,
        risk=risk_report,
        guideline=guideline_report,
        letter=letter_report,
        total_duration_ms=total_ms,
        sanity_flags_observed=sanity_observed,
        sanity_flags_missing=sanity_missing,
    )


@dataclass(frozen=True)
class AggregateReport:
    """Cross-case roll-up."""

    n_cases: int
    triage_pass_rate: float
    risk_band_match_rate: float
    guideline_pass_rate: float
    letter_pass_rate: float
    full_pipeline_pass_rate: float
    median_total_duration_ms: float
    p95_total_duration_ms: float
    confusion_matrix: dict[str, dict[str, int]]
    per_tag: dict[str, dict[str, float | int]]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p * (len(s) - 1))))
    return s[k]


def aggregate_reports(reports: list[CaseReport]) -> AggregateReport:
    """Roll up per-case reports into the headline numbers."""
    n = len(reports)
    if n == 0:
        return AggregateReport(
            n_cases=0,
            triage_pass_rate=0.0,
            risk_band_match_rate=0.0,
            guideline_pass_rate=0.0,
            letter_pass_rate=0.0,
            full_pipeline_pass_rate=0.0,
            median_total_duration_ms=0.0,
            p95_total_duration_ms=0.0,
            confusion_matrix={},
            per_tag={},
        )

    triage_pass = [r.triage.passed for r in reports]
    risk_pass = [r.risk.passed for r in reports]
    guideline_pass = [r.guideline.passed for r in reports]
    letter_pass = [r.letter.passed for r in reports]
    durations = [r.total_duration_ms for r in reports]

    # Confusion matrix (expected -> observed)
    confusion: dict[str, dict[str, int]] = {b: dict.fromkeys(BANDS, 0) for b in BANDS}
    for r in reports:
        if r.observed_risk_band in BANDS:
            confusion[r.expected_risk_band][r.observed_risk_band] += 1

    # Per-tag breakdown
    tags: dict[str, list[CaseReport]] = {}
    for r in reports:
        tags.setdefault(r.tag, []).append(r)
    per_tag: dict[str, dict[str, float | int]] = {}
    for tag, rs in sorted(tags.items()):
        per_tag[tag] = {
            "n": len(rs),
            "triage_pass_rate": _mean([r.triage.passed for r in rs]),
            "risk_band_match_rate": _mean([r.risk.passed for r in rs]),
            "guideline_pass_rate": _mean([r.guideline.passed for r in rs]),
            "letter_pass_rate": _mean([r.letter.passed for r in rs]),
        }

    return AggregateReport(
        n_cases=n,
        triage_pass_rate=_mean(triage_pass),
        risk_band_match_rate=_mean(risk_pass),
        guideline_pass_rate=_mean(guideline_pass),
        letter_pass_rate=_mean(letter_pass),
        full_pipeline_pass_rate=_mean(
            [
                t and r and g and lt
                for t, r, g, lt in zip(
                    triage_pass, risk_pass, guideline_pass, letter_pass, strict=True
                )
            ]
        ),
        median_total_duration_ms=_percentile(durations, 0.5),
        p95_total_duration_ms=_percentile(durations, 0.95),
        confusion_matrix=confusion,
        per_tag=per_tag,
    )


def _mean(values: list[bool]) -> float:
    return float(sum(1 for v in values if v) / len(values)) if values else 0.0


def report_to_dict(report: CaseReport) -> dict[str, Any]:
    """Convert a :class:`CaseReport` to JSON-safe dict."""
    return asdict(report)


__all__ = [
    "BANDS",
    "STAGES",
    "AggregateReport",
    "CaseReport",
    "StageReport",
    "aggregate_reports",
    "report_to_dict",
    "score_case",
]
