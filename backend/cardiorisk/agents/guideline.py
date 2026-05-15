"""Guideline agent.

Builds a clinician-style question from the patient + risk context and
calls the Phase 3.3 :class:`CitationGenerator` for a citation-mandatory
answer. The agent does not re-implement any of the citation contract
— it sits on top of ``CitationGenerator`` so the Phase 3.3 verifier
discipline (parsed claims, NLI verification, suppression with typed
reasons) flows through unchanged.

The question template intentionally does NOT include free-form patient
context (PHI-shaped strings) — only the risk band. This keeps the
LLM prompt corpus-bounded: the LLM should answer "what does the
guideline say for a patient in this risk band" rather than "should
this specific person take statins". The "for this patient" framing
lives in the letter agent, where the LLM is asked to assemble a
referral letter from already-cited claims.
"""

from __future__ import annotations

from cardiorisk.rag.generation.generator import CitationGenerator

from .retries import TransientAgentError, with_retries
from .state import GuidelineResult, PatientInput, RiskResult

#: Number of `with_retries` attempts on the LLM/NLI hot path.
DEFAULT_MAX_ATTEMPTS: int = 3


def build_question(*, patient: PatientInput, risk: RiskResult) -> str:
    """Construct the corpus-bounded clinician-style question.

    Phase-4 design choice: the question references *the risk band*,
    not the specific patient. The CitationGenerator's prompt
    explicitly tells the LLM to answer from the retrieved passages
    only; framing the question as "for a patient with high CVD
    risk" guides retrieval into the most useful section of the
    guideline corpus.
    """
    band_phrasing = {
        "high": "high",
        "intermediate": "intermediate",
        "low": "low",
    }[risk.risk_band]
    age_band = "older" if patient.Age >= 65 else "middle-aged" if patient.Age >= 45 else "younger"
    sex_word = {"M": "male", "F": "female"}[patient.Sex]
    return (
        f"What do the Australian CVD-risk guidelines recommend for "
        f"an {age_band} {sex_word} adult assessed at {band_phrasing} "
        f"5-year absolute CVD risk (model probability "
        f"{risk.probability:.1%}, calibrated)?"
    )


def run_guideline(
    *,
    patient: PatientInput,
    risk: RiskResult,
    generator: CitationGenerator,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[GuidelineResult, int]:
    """Run the guideline agent.

    Returns the :class:`GuidelineResult` and the attempt count from
    the retry loop (folded into the audit log by the graph node).

    The retry policy treats *all* CitationGenerator exceptions as
    transient — the underlying LLM and NLI calls are the only thing
    that should be retried, and Phase 3.3's MockLLM never raises, so
    in CI the loop runs once. Real-LLM 5xx and timeout paths are the
    Phase 6 thing this catches.
    """
    question = build_question(patient=patient, risk=risk)

    def _call() -> GuidelineResult:
        try:
            answer = generator.generate(question)
        except Exception as exc:  # pragma: no cover - exercised by Phase 6
            raise TransientAgentError(str(exc)) from exc

        verified_n = len(answer.verified_claims)
        suppressed_n = len(answer.suppressed_claims)
        if answer.is_refusal:
            summary = (
                f"Guideline answer refused (no passage entailed any "
                f"claim with probability >= verifier threshold). "
                f"({verified_n} verified, {suppressed_n} suppressed.)"
            )
        else:
            summary = (
                f"Guideline answer: {verified_n} verified claim(s), {suppressed_n} suppressed."
            )
        return GuidelineResult(question=question, answer=answer, summary=summary)

    result, attempts = with_retries(_call, max_attempts=max_attempts)
    return result, attempts


__all__ = ["DEFAULT_MAX_ATTEMPTS", "build_question", "run_guideline"]
