"""Tests for the deterministic letter agent."""

from __future__ import annotations

from cardiorisk.agents.letter import run_letter
from cardiorisk.agents.state import (
    GuidelineResult,
    PatientInput,
    RiskAttribution,
    RiskResult,
)
from cardiorisk.rag.generation.generator import (
    GeneratedAnswer,
    SuppressedClaim,
    VerifiedClaim,
)


def _patient() -> PatientInput:
    return PatientInput(
        Age=72,
        Sex="M",
        ChestPainType="ASY",
        RestingBP=160,
        Cholesterol=290,
        FastingBS=1,
        RestingECG="ST",
        MaxHR=100,
        ExerciseAngina="Y",
        Oldpeak=2.4,
        ST_Slope="Flat",
    )


def _risk(band: str = "high", probability: float = 0.18) -> RiskResult:
    return RiskResult(
        probability=probability,
        risk_band=band,  # type: ignore[arg-type]
        threshold_high=0.10,
        threshold_low=0.05,
        model_name="tabicl",
        model_artefact_present=True,
        top_attributions=(
            RiskAttribution(feature="Age", contribution=0.5),
            RiskAttribution(feature="ST_Slope", contribution=-0.4),
            RiskAttribution(feature="ChestPainType", contribution=0.3),
        ),
        summary="risk 18%",
    )


def _verified(text: str, chunk_id: str = "c1") -> VerifiedClaim:
    return VerifiedClaim(
        text=text,
        headline_chunk_id=chunk_id,
        headline_score=0.95,
        supporting_chunk_ids=(chunk_id,),
        supporting_scores=(0.95,),
    )


def _suppressed(text: str, reason: str = "no_passage_entails") -> SuppressedClaim:
    return SuppressedClaim(
        text=text,
        cited_chunk_ids=("c9",),
        best_entailment=0.05,
        reason=reason,
    )


def _guideline(
    *,
    verified: tuple[VerifiedClaim, ...] = (),
    suppressed: tuple[SuppressedClaim, ...] = (),
) -> GuidelineResult:
    answer = GeneratedAnswer(
        query="...",
        raw_llm_text="...",
        is_refusal=not verified,
        verified_claims=verified,
        suppressed_claims=suppressed,
    )
    return GuidelineResult(question="q", answer=answer, summary="x")


class TestRunLetter:
    def test_basic_letter_drafts_with_one_verified_claim(self) -> None:
        guideline = _guideline(verified=(_verified("Pharmacotherapy is recommended."),))
        out = run_letter(patient=_patient(), risk=_risk(), guideline=guideline)
        assert "Dear Colleague" in out.draft
        assert "Pharmacotherapy is recommended." in out.draft
        assert "[c1]" in out.draft
        assert out.citations == ("c1",)
        assert out.redacted_claims == ()

    def test_letter_renders_band_and_probability(self) -> None:
        guideline = _guideline(verified=(_verified("Refer."),))
        out = run_letter(patient=_patient(), risk=_risk(probability=0.27), guideline=guideline)
        assert "27.0%" in out.draft
        assert "high" in out.draft

    def test_letter_records_suppression_note(self) -> None:
        guideline = _guideline(
            verified=(_verified("Refer."),),
            suppressed=(
                _suppressed("Phantom claim 1", reason="phantom_citation"),
                _suppressed("Hallucinated claim 2", reason="no_passage_entails"),
            ),
        )
        out = run_letter(patient=_patient(), risk=_risk(), guideline=guideline)
        assert "suppressed by the citation verifier" in out.draft
        assert "phantom_citation" in out.draft
        assert "no_passage_entails" in out.draft

    def test_letter_handles_empty_verified_claims(self) -> None:
        guideline = _guideline(verified=())
        out = run_letter(patient=_patient(), risk=_risk(), guideline=guideline)
        assert "no verified guideline recommendations" in out.draft
        assert out.citations == ()

    def test_letter_redacts_claims_without_chunk_id(self) -> None:
        # An empty headline_chunk_id should send the claim to the
        # redacted pile rather than the cited bullets.
        weird = VerifiedClaim(
            text="An orphan claim with no chunk id.",
            headline_chunk_id="",
            headline_score=0.0,
            supporting_chunk_ids=(),
            supporting_scores=(),
        )
        guideline = _guideline(verified=(_verified("ok."), weird))
        out = run_letter(patient=_patient(), risk=_risk(), guideline=guideline)
        assert "An orphan claim" in out.redacted_claims[0]
        assert "An orphan claim" not in out.draft

    def test_letter_summary_counts_match(self) -> None:
        guideline = _guideline(verified=(_verified("a."), _verified("b.", "c2")))
        out = run_letter(patient=_patient(), risk=_risk(), guideline=guideline)
        assert "2 cited recommendation" in out.summary
        assert "0 claim(s) redacted" in out.summary

    def test_letter_disclaimer_present(self) -> None:
        out = run_letter(
            patient=_patient(),
            risk=_risk(),
            guideline=_guideline(verified=(_verified("ok."),)),
        )
        assert "synthetic" in out.draft.lower()
        assert "not for clinical use" in out.draft.lower()
