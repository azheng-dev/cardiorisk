"""Per-case + aggregate scoring for the Phase-4 / Phase-6 agent eval.

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

Phase-6 additions (ADR-019, four new metrics):

- ``citation_precision`` per case = fraction of (verified-claim, cited-
  chunk) pairs where the cited chunk is in the retrieved set. ``1.0``
  when the generator never invented a chunk_id, ``0.0`` when every
  citation is phantom. Mean across cases is the headline.
- ``citation_recall`` per case = fraction of verified claims that have
  at least one citation pointing into the retrieved set. ``1.0`` when
  every verified claim is grounded, ``0.0`` if the generator emitted
  text without citations. Mean across cases is the headline.
- ``recommendation_correctness`` per case = boolean, the letter draft
  contains a keyword from the expected ``recommendation_family``
  keyword table (case-insensitive). Mean is the headline.
- ``hallucination_rate`` per case = fraction of claims (verified +
  suppressed) that the NLI verifier suppressed for an evidence-related
  reason (``'phantom_citation' | 'no_passage_entails' | 'no_citation'``).
  Lower is better. This is *not* the same as citation precision: a
  claim with a phantom citation can still be suppressed by the
  verifier, which is the right behaviour; the hallucination rate
  tracks "how often did the LLM try" rather than "how often did the
  system ship the hallucination".
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

# Phase-6 recommendation-family -> keyword-family table (ADR-019 §4).
# A letter draft is credited for a family if it contains AT LEAST ONE
# keyword from that family's list (case-insensitive substring). The
# table is intentionally conservative: each family's keyword list is
# small enough that random text doesn't trigger false positives, but
# wide enough to allow reasonable phrasing variation by the LLM.
RECOMMENDATION_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lifestyle_only": (
        "lifestyle",
        "diet",
        "exercise",
        "physical activity",
        "smoking cessation",
        "weight",
    ),
    "lifestyle_plus_review": (
        "lifestyle",
        "review",
        "follow-up",
        "follow up",
        "reassess",
        "recheck",
    ),
    "statin_consider": (
        "statin",
        "consider statin",
        "lipid-lowering",
        "lipid lowering",
    ),
    "statin_plus_bp": (
        "statin",
        "blood pressure",
        "antihypertens",
        "bp control",
        "ace inhibitor",
        "arb",
    ),
    "statin_plus_bp_plus_referral": (
        "statin",
        "blood pressure",
        "refer",
        "referral",
        "cardiology",
        "cardiologist",
    ),
    "specialist_referral_urgent": (
        "urgent",
        "refer",
        "referral",
        "cardiology",
        "cardiologist",
        "emergency",
    ),
    "refusal_no_recommendation": (
        "i do not have the supporting guidance",
        "unable to recommend",
        "insufficient evidence",
        "cannot provide a recommendation",
        "no specific recommendation",
    ),
}

# Suppression reasons that we count as evidence-side hallucination
# attempts. ``'no_citation'`` is included because emitting an
# uncited claim is itself a fabrication attempt: the LLM was asked
# to cite and didn't.
HALLUCINATION_SUPPRESSION_REASONS: frozenset[str] = frozenset(
    {"phantom_citation", "no_passage_entails", "no_citation"}
)


@dataclass(frozen=True)
class StageReport:
    """Per-stage pass/fail + small numeric diagnostics."""

    stage: str
    passed: bool
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseReport:
    """Per-case full report: 4 stage reports + meta + Phase-6 metrics."""

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
    # Phase 6 (ADR-019)
    expected_recommendation_family: str
    recommendation_correct: bool
    citation_precision: float
    citation_recall: float
    hallucination_rate: float


def _stage_duration_ms(state: AgentState, stage: str) -> float:
    for entry in state.audit:
        if entry.stage.value == stage:
            return entry.duration_ms
    return 0.0


def _word_count(text: str) -> int:
    return len(text.split())


def _citation_precision_recall(state: AgentState) -> tuple[float, float]:
    """Compute (precision, recall) of cited chunk_ids against the
    retrieval set.

    Definitions:

    - Precision = supported pairs / total cited pairs across all
      verified claims. A "pair" is (verified_claim, cited_chunk_id),
      flattening both headline + supporting citations.
      ``1.0`` if no claims were cited (vacuous) — we only mark this
      down if there *were* citations and some were phantom.
    - Recall = verified claims with >= 1 cited chunk in the retrieved
      set / total verified claims. ``1.0`` if no verified claims
      (vacuous).
    """
    guideline = state.guideline
    if guideline is None:
        return 1.0, 1.0
    retrieved_ids = {rc.chunk.chunk_id for rc in guideline.answer.retrieved}
    claims = guideline.answer.verified_claims
    if not claims:
        return 1.0, 1.0

    total_pairs = 0
    supported_pairs = 0
    claims_with_at_least_one_supported = 0
    for claim in claims:
        cited = (claim.headline_chunk_id, *claim.supporting_chunk_ids)
        cited = tuple(c for c in cited if c)
        any_supported = False
        for chunk_id in cited:
            total_pairs += 1
            if chunk_id in retrieved_ids:
                supported_pairs += 1
                any_supported = True
        if any_supported:
            claims_with_at_least_one_supported += 1

    precision = (supported_pairs / total_pairs) if total_pairs else 1.0
    recall = claims_with_at_least_one_supported / len(claims)
    return precision, recall


def _hallucination_rate(state: AgentState) -> float:
    """Fraction of total claims (verified + suppressed) that were
    suppressed for an evidence-side reason. Refusals (zero claims
    total) score 0.0 since there's nothing to fabricate.
    """
    guideline = state.guideline
    if guideline is None:
        return 0.0
    verified = guideline.answer.verified_claims
    suppressed = guideline.answer.suppressed_claims
    total = len(verified) + len(suppressed)
    if total == 0:
        return 0.0
    bad = sum(1 for s in suppressed if s.reason in HALLUCINATION_SUPPRESSION_REASONS)
    return bad / total


def _recommendation_correct(state: AgentState, family: str) -> bool:
    """True iff the letter draft contains a keyword from the expected
    recommendation family's keyword table (case-insensitive substring).
    Refusal cases are scored against the refusal keyword table — a
    correct refusal draft says "I do not have the supporting guidance"
    (or similar).
    """
    letter = state.letter
    if letter is None:
        return False
    draft = letter.draft.lower()
    keywords = RECOMMENDATION_FAMILY_KEYWORDS.get(family, ())
    return any(k.lower() in draft for k in keywords)


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

    # Phase-6 metrics
    citation_precision, citation_recall = _citation_precision_recall(state)
    hallucination = _hallucination_rate(state)
    recommendation_correct = _recommendation_correct(state, case.expected_recommendation_family)

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
        expected_recommendation_family=case.expected_recommendation_family,
        recommendation_correct=recommendation_correct,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        hallucination_rate=hallucination,
    )


@dataclass(frozen=True)
class AggregateReport:
    """Cross-case roll-up (Phase 4 + Phase 6 metrics)."""

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
    # Phase 6 (ADR-019)
    recommendation_correctness_rate: float
    mean_citation_precision: float
    mean_citation_recall: float
    mean_hallucination_rate: float


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
            recommendation_correctness_rate=0.0,
            mean_citation_precision=0.0,
            mean_citation_recall=0.0,
            mean_hallucination_rate=0.0,
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

    # Per-tag breakdown (now also includes the Phase-6 metrics)
    tags: dict[str, list[CaseReport]] = {}
    for r in reports:
        tags.setdefault(r.tag, []).append(r)
    per_tag: dict[str, dict[str, float | int]] = {}
    for tag, rs in sorted(tags.items()):
        per_tag[tag] = {
            "n": len(rs),
            "triage_pass_rate": _mean_bool([r.triage.passed for r in rs]),
            "risk_band_match_rate": _mean_bool([r.risk.passed for r in rs]),
            "guideline_pass_rate": _mean_bool([r.guideline.passed for r in rs]),
            "letter_pass_rate": _mean_bool([r.letter.passed for r in rs]),
            "recommendation_correctness_rate": _mean_bool([r.recommendation_correct for r in rs]),
            "mean_citation_precision": _mean_float([r.citation_precision for r in rs]),
            "mean_citation_recall": _mean_float([r.citation_recall for r in rs]),
            "mean_hallucination_rate": _mean_float([r.hallucination_rate for r in rs]),
        }

    return AggregateReport(
        n_cases=n,
        triage_pass_rate=_mean_bool(triage_pass),
        risk_band_match_rate=_mean_bool(risk_pass),
        guideline_pass_rate=_mean_bool(guideline_pass),
        letter_pass_rate=_mean_bool(letter_pass),
        full_pipeline_pass_rate=_mean_bool(
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
        recommendation_correctness_rate=_mean_bool([r.recommendation_correct for r in reports]),
        mean_citation_precision=_mean_float([r.citation_precision for r in reports]),
        mean_citation_recall=_mean_float([r.citation_recall for r in reports]),
        mean_hallucination_rate=_mean_float([r.hallucination_rate for r in reports]),
    )


def _mean_bool(values: list[bool]) -> float:
    return float(sum(1 for v in values if v) / len(values)) if values else 0.0


def _mean_float(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# Back-compat alias: the original public name was ``_mean`` and took a
# list of bools. Some external code (and the figures module's older
# tests) may import it. The new code paths use ``_mean_bool`` /
# ``_mean_float`` directly so the intent is clear at the call site.
_mean = _mean_bool


def report_to_dict(report: CaseReport) -> dict[str, Any]:
    """Convert a :class:`CaseReport` to JSON-safe dict."""
    return asdict(report)


__all__ = [
    "BANDS",
    "HALLUCINATION_SUPPRESSION_REASONS",
    "RECOMMENDATION_FAMILY_KEYWORDS",
    "STAGES",
    "AggregateReport",
    "CaseReport",
    "StageReport",
    "aggregate_reports",
    "report_to_dict",
    "score_case",
]
