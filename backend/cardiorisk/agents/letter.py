"""Letter agent.

Drafts a referral letter from the verified guideline claims + risk
context. Phase 4 ships a *deterministic template-based* letter
drafter rather than a second LLM round — Phase 3.3's
:class:`CitationGenerator` already enforces the citation contract
on the guideline answer, and the letter agent's job is to
*re-format* those verified claims into a referral-letter shape
without inventing any new clinical content.

The Phase 6 plan revisits this and either:

- Adds a second LLM call (+verifier) that paraphrases the verified
  claims into letter prose, with the same suppression contract.
- Or keeps the deterministic template if the eval shows the
  template-based letters are clinically acceptable to a clinician
  reviewer.

Either way the public surface (:func:`run_letter`) is stable.
"""

from __future__ import annotations

from .state import GuidelineResult, LetterResult, PatientInput, RiskResult


def _format_band(risk: RiskResult) -> str:
    return {
        "high": "high (>=10% 5-year absolute risk)",
        "intermediate": "intermediate (5-10% 5-year absolute risk)",
        "low": "low (<=5% 5-year absolute risk)",
    }[risk.risk_band]


def _format_attribution(attr_label: str) -> str:
    return attr_label.replace("_", " ").lower()


def run_letter(
    *,
    patient: PatientInput,
    risk: RiskResult,
    guideline: GuidelineResult,
) -> LetterResult:
    """Draft the referral letter.

    The output letter is a structured template:

    1. Opening (referrer + patient identifier scaffolding; Phase 4
       leaves real PII placeholders).
    2. Risk summary (model + calibrated probability + band).
    3. Top contributing factors (top 3 risk attributions).
    4. Guideline recommendations (one bullet per verified claim,
       cited with the chunk_id the claim was verified against).
    5. Suppression note if any claims were dropped.

    Citations re-use the chunk_ids from the guideline answer's
    verified claims; the letter never introduces a citation that
    didn't already pass the Phase 3.3 verifier. ``redacted_claims``
    records every guideline claim that *was* verified but didn't
    fit the letter template (rare; happens when a claim is too
    fragmentary to render as a recommendation bullet).
    """
    # Risk summary
    band_str = _format_band(risk)
    fact_lines: list[str] = []
    for attr in risk.top_attributions[:3]:
        sign = "+" if attr.contribution >= 0 else "-"
        fact_lines.append(
            f"  - {_format_attribution(attr.feature)} "
            f"({sign}{abs(attr.contribution):.2f} contribution)"
        )

    # Recommendation bullets — only include claims that came back
    # verified AND have a non-empty headline_chunk_id (so the
    # citation is real). Anything else goes to redacted_claims.
    rec_lines: list[str] = []
    cited: list[str] = []
    redacted: list[str] = []
    for claim in guideline.answer.verified_claims:
        if not claim.headline_chunk_id:
            redacted.append(claim.text)
            continue
        rec_lines.append(f"  - {claim.text} [{claim.headline_chunk_id}]")
        cited.append(claim.headline_chunk_id)

    suppressed = guideline.answer.suppressed_claims
    suppression_note = ""
    if suppressed:
        reasons = sorted({s.reason for s in suppressed})
        suppression_note = (
            f"\nNote: {len(suppressed)} draft claim(s) were suppressed "
            f"by the citation verifier (reasons: {', '.join(reasons)})."
        )

    body = (
        f"Dear Colleague,\n\n"
        f"I am referring this {patient.Age}-year-old "
        f"{'male' if patient.Sex == 'M' else 'female'} patient for "
        f"specialist cardiovascular review.\n\n"
        f"Risk assessment\n"
        f"  - Calibrated 5-year absolute CVD-risk probability: "
        f"{risk.probability:.1%} ({band_str}).\n"
        f"  - Model: {risk.model_name} "
        f"({'real artefact' if risk.model_artefact_present else 'deterministic stand-in'}).\n"
        f"Top contributing factors:\n"
        f"{chr(10).join(fact_lines) if fact_lines else '  - (no per-case attributions available)'}\n\n"
        f"Guideline-based considerations\n"
        f"{chr(10).join(rec_lines) if rec_lines else '  - (no verified guideline recommendations available)'}\n"
        f"{suppression_note}\n\n"
        f"Yours sincerely,\nCardioRisk Co-Pilot (synthetic; not for clinical use)"
    )

    summary = (
        f"Letter drafted: {len(rec_lines)} cited recommendation(s); "
        f"{len(redacted)} claim(s) redacted; "
        f"{len(suppressed)} prior-stage suppression(s)."
    )

    return LetterResult(
        draft=body,
        citations=tuple(cited),
        redacted_claims=tuple(redacted),
        summary=summary,
    )


__all__ = ["run_letter"]
