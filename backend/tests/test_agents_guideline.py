"""Tests for the guideline agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cardiorisk.agents.guideline import build_question, run_guideline
from cardiorisk.agents.state import (
    PatientInput,
    RiskAttribution,
    RiskResult,
)
from cardiorisk.rag.generation.generator import (
    CitationGenerator,
    GeneratedAnswer,
    VerifiedClaim,
)
from cardiorisk.rag.generation.llm import MockLLMClient
from cardiorisk.rag.generation.nli import EntailmentResult
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


def _patient(**overrides: object) -> PatientInput:
    base = {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 240,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Up",
    }
    base.update(overrides)
    return PatientInput(**base)  # type: ignore[arg-type]


def _risk(band: str = "high", probability: float = 0.18) -> RiskResult:
    return RiskResult(
        probability=probability,
        risk_band=band,  # type: ignore[arg-type]
        threshold_high=0.10,
        threshold_low=0.05,
        model_name="mock-risk-v1",
        model_artefact_present=False,
        top_attributions=(RiskAttribution(feature="Age", contribution=0.5),),
        summary=f"risk={probability}",
    )


def _chunk(chunk_id: str, text: str, doc_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        strategy="token",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
        text=text,
        n_tokens=len(text.split()),
    )


@dataclass
class _StubPipeline:
    chunks: list[Chunk]

    def retrieve(
        self, query: str, *, top_k: int = 5, with_rerank: bool = False
    ) -> list[RetrievedChunk]:
        del query, with_rerank
        return [
            RetrievedChunk(
                chunk=c,
                score=1.0 - i * 0.1,
                rrf_score=1.0 - i * 0.1,
                vector_rank=i + 1,
                bm25_rank=i + 1,
                rerank_score=None,
            )
            for i, c in enumerate(self.chunks[:top_k])
        ]


class _AlwaysEntails:
    name = "always-entail"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        del premise, hypothesis
        return EntailmentResult(p_entailment=0.99, p_neutral=0.005, p_contradiction=0.005)

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


class TestBuildQuestion:
    def test_question_includes_band_age_and_sex(self) -> None:
        q = build_question(patient=_patient(Age=72), risk=_risk(band="high"))
        assert "high" in q
        assert "older" in q
        assert "male" in q

    def test_band_phrasing_branches(self) -> None:
        for band in ("high", "intermediate", "low"):
            q = build_question(patient=_patient(), risk=_risk(band=band))
            assert band in q

    def test_age_band_branches(self) -> None:
        assert "older" in build_question(patient=_patient(Age=70), risk=_risk())
        assert "middle-aged" in build_question(patient=_patient(Age=50), risk=_risk())
        assert "younger" in build_question(patient=_patient(Age=30), risk=_risk())


class TestRunGuideline:
    def test_returns_verified_claims_when_all_entail(self) -> None:
        chunks = [
            _chunk(
                "c1",
                "Pharmacotherapy is recommended for high CVD risk patients.",
                doc_id="nvdpa",
            ),
            _chunk(
                "c2",
                "Lifestyle interventions remain first-line for primary prevention.",
                doc_id="racgp",
            ),
        ]
        gen = CitationGenerator(
            retrieval_pipeline=_StubPipeline(chunks),  # type: ignore[arg-type]
            llm_client=MockLLMClient(),
            nli_verifier=_AlwaysEntails(),
        )
        result, attempts = run_guideline(patient=_patient(), risk=_risk(), generator=gen)
        assert attempts == 1
        assert result.question
        assert isinstance(result.answer, GeneratedAnswer)
        assert len(result.answer.verified_claims) >= 1
        assert "verified" in result.summary

    def test_summary_calls_out_refusal_path(self) -> None:
        # No chunks → CitationGenerator returns a refusal
        gen = CitationGenerator(
            retrieval_pipeline=_StubPipeline([]),  # type: ignore[arg-type]
            llm_client=MockLLMClient(),
            nli_verifier=_AlwaysEntails(),
        )
        result, _ = run_guideline(patient=_patient(), risk=_risk(), generator=gen)
        assert result.answer.is_refusal is True
        assert "refused" in result.summary.lower()

    def test_verified_claims_carry_chunk_ids(self) -> None:
        chunks = [
            _chunk(
                "c1",
                "Pharmacotherapy is recommended for high CVD risk patients.",
                doc_id="nvdpa",
            ),
        ]
        gen = CitationGenerator(
            retrieval_pipeline=_StubPipeline(chunks),  # type: ignore[arg-type]
            llm_client=MockLLMClient(),
            nli_verifier=_AlwaysEntails(),
        )
        result, _ = run_guideline(patient=_patient(), risk=_risk(), generator=gen)
        assert all(isinstance(c, VerifiedClaim) for c in result.answer.verified_claims)
        for c in result.answer.verified_claims:
            assert c.headline_chunk_id == "c1"
