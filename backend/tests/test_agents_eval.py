"""Tests for the Phase 4 agent eval (loader + scorer + orchestrator)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cardiorisk.agents.eval import (
    aggregate_reports,
    load_cases,
    run_eval,
    score_case,
)
from cardiorisk.agents.eval.figures import render_all
from cardiorisk.agents.eval.loader import AgentEvalCase
from cardiorisk.agents.eval.scorer import RECOMMENDATION_FAMILY_KEYWORDS
from cardiorisk.agents.state import (
    AgentState,
    AuditEntry,
    GuidelineResult,
    LetterResult,
    PatientInput,
    RiskAttribution,
    RiskResult,
    TriageResult,
)
from cardiorisk.rag.generation.generator import (
    CitationGenerator,
    GeneratedAnswer,
    SuppressedClaim,
    VerifiedClaim,
)
from cardiorisk.rag.generation.llm import MockLLMClient
from cardiorisk.rag.generation.nli import EntailmentResult, MockNLIVerifier
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[2]


def _patient(**overrides: object) -> PatientInput:
    base: dict[str, object] = {
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


def _verified(text: str, chunk_id: str = "c1") -> VerifiedClaim:
    return VerifiedClaim(
        text=text,
        headline_chunk_id=chunk_id,
        headline_score=0.95,
        supporting_chunk_ids=(chunk_id,),
        supporting_scores=(0.95,),
    )


# ----------------------------------------------------------------- loader
class TestLoadCases:
    def test_loads_full_case_set(self) -> None:
        cases = load_cases(repo_root=REPO_ROOT)
        assert len(cases) >= 100  # Phase 6 locks at 100
        for c in cases:
            assert c.id.startswith("a")
            assert c.expected_risk_band in ("low", "intermediate", "high")
            assert c.tag in (
                "high_risk",
                "intermediate_risk",
                "low_risk",
                "borderline",
                "extreme_case",
                "data_quality",
                "refusal",
            )
            # Phase-6 field: every case has a recommendation family
            assert c.expected_recommendation_family in {
                "lifestyle_only",
                "lifestyle_plus_review",
                "statin_consider",
                "statin_plus_bp",
                "statin_plus_bp_plus_referral",
                "specialist_referral_urgent",
                "refusal_no_recommendation",
            }

    def test_tag_filter(self) -> None:
        cases = load_cases(repo_root=REPO_ROOT, tag_filter="high_risk")
        assert all(c.tag == "high_risk" for c in cases)
        assert len(cases) > 0

    def test_limit(self) -> None:
        cases = load_cases(repo_root=REPO_ROOT, limit=5)
        assert len(cases) == 5

    def test_validates_against_schema(self, tmp_path: Path) -> None:
        bad = tmp_path / "cases.jsonl"
        bad.write_text(
            json.dumps({"id": "bad", "patient": {}, "expected_risk_band": "x"}) + "\n",
            encoding="utf-8",
        )
        from jsonschema import ValidationError

        with pytest.raises(ValidationError):
            load_cases(cases_path=bad, repo_root=REPO_ROOT)


# ----------------------------------------------------------------- scorer
def _retrieved(chunk_ids: Sequence[str]) -> tuple[RetrievedChunk, ...]:
    out: list[RetrievedChunk] = []
    for i, cid in enumerate(chunk_ids):
        chunk = Chunk(
            chunk_id=cid,
            doc_id="doc",
            strategy="token",
            char_start=0,
            char_end=10,
            page_start=1,
            page_end=1,
            text=f"text-{cid}",
            n_tokens=2,
        )
        out.append(
            RetrievedChunk(
                chunk=chunk,
                score=1.0 - i * 0.1,
                rrf_score=1.0 - i * 0.1,
                vector_rank=i + 1,
                bm25_rank=i + 1,
                rerank_score=None,
            )
        )
    return tuple(out)


def _make_state(
    *,
    case: AgentEvalCase,
    band: str = "high",
    sanity_flags: tuple[str, ...] = (),
    n_verified: int = 1,
    n_suppressed: int = 0,
    letter_words: int = 80,
    letter_text: str | None = None,
    retrieved_chunk_ids: Sequence[str] | None = None,
    verified_cite_ids: Sequence[str] | None = None,
    suppression_reason: str = "no_passage_entails",
) -> AgentState:
    triage = TriageResult(
        normalised_patient=case.patient,
        sanity_flags=sanity_flags,
        summary="ok",
    )
    risk = RiskResult(
        probability=0.42,
        risk_band=band,  # type: ignore[arg-type]
        threshold_high=0.10,
        threshold_low=0.05,
        model_name="mock-risk-v1",
        model_artefact_present=False,
        top_attributions=(RiskAttribution(feature="Age", contribution=0.5),),
        summary="risk",
    )
    # Default: verified claims cite c0..c{n-1}, all of which are in the
    # retrieved set => citation_precision / recall = 1.0.
    cite_ids = (
        tuple(verified_cite_ids)
        if verified_cite_ids is not None
        else tuple(f"c{i}" for i in range(n_verified))
    )
    retrieved = _retrieved(
        retrieved_chunk_ids
        if retrieved_chunk_ids is not None
        else [f"c{i}" for i in range(max(n_verified, 1))]
    )
    verified_claims = tuple(
        _verified(f"Claim {i}.", cite_ids[i] if i < len(cite_ids) else f"c{i}")
        for i in range(n_verified)
    )
    answer = GeneratedAnswer(
        query="...",
        raw_llm_text="...",
        is_refusal=False,
        verified_claims=verified_claims,
        suppressed_claims=tuple(
            SuppressedClaim(
                text=f"S{i}",
                cited_chunk_ids=(),
                best_entailment=0.0,
                reason=suppression_reason,
            )
            for i in range(n_suppressed)
        ),
        retrieved=retrieved,
    )
    guideline = GuidelineResult(question="q", answer=answer, summary="g")
    if letter_text is None:
        letter_text = "x " * letter_words
    letter = LetterResult(draft=letter_text.strip(), citations=("c1",), summary="l")
    audit = tuple(
        AuditEntry(
            stage=stage,  # type: ignore[arg-type]
            started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            completed_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            duration_ms=10.0,
        )
        for stage in ("triage", "risk", "guideline", "letter")
    )
    return AgentState(
        case_id=case.id,
        patient=case.patient,
        triage=triage,
        risk=risk,
        guideline=guideline,
        letter=letter,
        audit=audit,
    )


class TestScoreCase:
    def test_all_stages_pass_for_clean_state(self) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="high",
            expected_min_verified_claims=1,
            expected_letter_min_words=60,
            expected_sanity_flags=(),
            tag="high_risk",
            rationale="...",
        )
        state = _make_state(case=case, band="high", n_verified=2, letter_words=80)
        report = score_case(case, state)
        assert report.triage.passed
        assert report.risk.passed
        assert report.guideline.passed
        assert report.letter.passed
        assert report.band_match
        assert report.n_verified_claims == 2

    def test_band_mismatch_fails_risk_only(self) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="low",
            expected_min_verified_claims=1,
            expected_letter_min_words=60,
            expected_sanity_flags=(),
            tag="low_risk",
            rationale="...",
        )
        state = _make_state(case=case, band="high", n_verified=2)
        report = score_case(case, state)
        assert report.triage.passed
        assert not report.risk.passed
        assert report.guideline.passed
        assert report.letter.passed

    def test_missing_expected_sanity_flag_fails_triage(self) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="high",
            expected_min_verified_claims=1,
            expected_letter_min_words=60,
            expected_sanity_flags=("cholesterol_missing_sentinel",),
            tag="data_quality",
            rationale="...",
        )
        state = _make_state(case=case, sanity_flags=())
        report = score_case(case, state)
        assert not report.triage.passed
        assert "cholesterol_missing_sentinel" in report.sanity_flags_missing

    def test_short_letter_fails_letter(self) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="high",
            expected_min_verified_claims=1,
            expected_letter_min_words=200,
            expected_sanity_flags=(),
            tag="high_risk",
            rationale="...",
        )
        state = _make_state(case=case, letter_words=50)
        report = score_case(case, state)
        assert not report.letter.passed


# ----------------------------------------------------------------- Phase 6 metrics
class TestPhase6Metrics:
    """Citation precision / recall, recommendation correctness, hallucination."""

    def _case(
        self,
        family: str = "statin_consider",
        tag: str = "intermediate_risk",
        band: str = "intermediate",
    ) -> AgentEvalCase:
        return AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band=band,
            expected_min_verified_claims=1,
            expected_letter_min_words=10,
            expected_sanity_flags=(),
            tag=tag,
            rationale="...",
            expected_recommendation_family=family,
        )

    def test_perfect_citations_score_one(self) -> None:
        case = self._case()
        state = _make_state(case=case, n_verified=2, retrieved_chunk_ids=("c0", "c1"))
        r = score_case(case, state)
        assert r.citation_precision == pytest.approx(1.0)
        assert r.citation_recall == pytest.approx(1.0)
        assert r.hallucination_rate == pytest.approx(0.0)

    def test_phantom_citation_lowers_precision_and_recall(self) -> None:
        case = self._case()
        # Two claims; first cites c0 (in retrieved), second cites c99 (phantom).
        state = _make_state(
            case=case,
            n_verified=2,
            retrieved_chunk_ids=("c0",),
            verified_cite_ids=("c0", "c99"),
        )
        r = score_case(case, state)
        assert r.citation_precision == pytest.approx(0.5)
        assert r.citation_recall == pytest.approx(0.5)

    def test_no_verified_no_suppressed_is_vacuous_one(self) -> None:
        case = self._case(family="refusal_no_recommendation", tag="refusal", band="low")
        state = _make_state(case=case, n_verified=0, n_suppressed=0)
        r = score_case(case, state)
        assert r.citation_precision == pytest.approx(1.0)
        assert r.citation_recall == pytest.approx(1.0)
        assert r.hallucination_rate == pytest.approx(0.0)

    def test_suppression_counts_as_hallucination(self) -> None:
        case = self._case()
        # 1 verified + 1 suppressed (no_passage_entails) => 1/2 hallucinated
        state = _make_state(case=case, n_verified=1, n_suppressed=1)
        r = score_case(case, state)
        assert r.hallucination_rate == pytest.approx(0.5)

    def test_suppression_other_reasons_not_counted(self) -> None:
        case = self._case()
        # 'parser_error' is not in HALLUCINATION_SUPPRESSION_REASONS
        state = _make_state(
            case=case,
            n_verified=1,
            n_suppressed=1,
            suppression_reason="parser_error",
        )
        r = score_case(case, state)
        assert r.hallucination_rate == pytest.approx(0.0)

    def test_recommendation_correct_when_keyword_present(self) -> None:
        case = self._case(family="statin_plus_bp")
        keywords = RECOMMENDATION_FAMILY_KEYWORDS["statin_plus_bp"]
        letter = f"Consider {keywords[0]} therapy and review {keywords[1]}."
        state = _make_state(case=case, n_verified=1, letter_text=letter)
        r = score_case(case, state)
        assert r.recommendation_correct is True

    def test_recommendation_wrong_when_no_keyword(self) -> None:
        case = self._case(family="lifestyle_only")
        state = _make_state(case=case, n_verified=1, letter_text="Please continue current care.")
        r = score_case(case, state)
        assert r.recommendation_correct is False

    def test_refusal_family_credits_canonical_refusal_text(self) -> None:
        case = self._case(family="refusal_no_recommendation", tag="refusal", band="low")
        # Canonical refusal text from the generator
        letter = "I do not have the supporting guidance for that question."
        state = _make_state(case=case, n_verified=0, letter_text=letter)
        r = score_case(case, state)
        assert r.recommendation_correct is True

    def test_aggregate_rolls_up_phase_6_metrics(self) -> None:
        case = self._case(family="statin_plus_bp")
        keyword = RECOMMENDATION_FAMILY_KEYWORDS["statin_plus_bp"][0]
        good_letter = f"Recommend {keyword} therapy."
        bad_letter = "Please continue current care."
        reports = [
            score_case(
                case,
                _make_state(case=case, n_verified=2, letter_text=good_letter),
            ),
            score_case(
                case,
                _make_state(
                    case=case,
                    n_verified=2,
                    verified_cite_ids=("c0", "c99"),
                    retrieved_chunk_ids=("c0",),
                    letter_text=bad_letter,
                ),
            ),
        ]
        agg = aggregate_reports(reports)
        assert agg.recommendation_correctness_rate == pytest.approx(0.5)
        assert agg.mean_citation_precision == pytest.approx(0.75)  # (1.0 + 0.5) / 2
        assert agg.mean_citation_recall == pytest.approx(0.75)
        assert agg.mean_hallucination_rate == pytest.approx(0.0)


# ----------------------------------------------------------------- aggregate
class TestAggregate:
    def test_aggregate_pass_rates_correct(self) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="high",
            expected_min_verified_claims=1,
            expected_letter_min_words=60,
            expected_sanity_flags=(),
            tag="high_risk",
            rationale="...",
        )
        # 2 high-pass + 1 band-miss
        reports = [
            score_case(case, _make_state(case=case, band="high")),
            score_case(case, _make_state(case=case, band="high")),
            score_case(case, _make_state(case=case, band="low")),
        ]
        agg = aggregate_reports(reports)
        assert agg.n_cases == 3
        assert agg.risk_band_match_rate == pytest.approx(2 / 3)
        assert agg.full_pipeline_pass_rate == pytest.approx(2 / 3)
        assert agg.confusion_matrix["high"]["high"] == 2
        assert agg.confusion_matrix["high"]["low"] == 1


# ----------------------------------------------------------------- figures
class TestFigures:
    def test_render_all_writes_three_pngs(self, tmp_path: Path) -> None:
        case = AgentEvalCase(
            id="a000",
            patient=_patient(),
            expected_risk_band="high",
            expected_min_verified_claims=1,
            expected_letter_min_words=60,
            expected_sanity_flags=(),
            tag="high_risk",
            rationale="...",
        )
        reports = [score_case(case, _make_state(case=case, band="high"))]
        agg = aggregate_reports(reports)
        out = render_all(agg, tmp_path)
        for k in ("per_stage_pass_rate", "risk_band_confusion", "per_tag_pass_rate"):
            assert out[k].exists()
            assert out[k].stat().st_size > 0


# ----------------------------------------------------------------- orchestrator
def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
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


class _AlwaysEntails(MockNLIVerifier):
    name: str = "always-entail"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        del premise, hypothesis
        return EntailmentResult(p_entailment=0.99, p_neutral=0.005, p_contradiction=0.005)

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


class TestRunEvalEndToEnd:
    def test_smoke_run_writes_outputs(self, tmp_path: Path) -> None:
        cases = load_cases(repo_root=REPO_ROOT, limit=2)
        gen = CitationGenerator(
            retrieval_pipeline=_StubPipeline(  # type: ignore[arg-type]
                [_chunk("c1", "Pharmacotherapy is recommended for high-risk patients.")]
            ),
            llm_client=MockLLMClient(),
            nli_verifier=_AlwaysEntails(),
        )
        out_dir = tmp_path / "reports"
        fig_dir = tmp_path / "figs"
        summary = run_eval(
            generator=gen,
            cases=cases,
            is_smoke=True,
            output_dir=out_dir,
            figures_dir=fig_dir,
        )
        # is_smoke=True nests under smoke/ by orchestrator design
        per_case_path = out_dir / "smoke" / "per_case.json"
        agg_path = out_dir / "smoke" / "aggregate.json"
        assert per_case_path.exists()
        assert agg_path.exists()
        loaded = json.loads(agg_path.read_text(encoding="utf-8"))
        assert loaded["aggregate"]["n_cases"] == 2
        assert summary["aggregate"]["n_cases"] == 2
        # figures
        for name in ("per_stage_pass_rate", "risk_band_confusion", "per_tag_pass_rate"):
            assert (fig_dir / "smoke" / f"{name}.png").exists()
