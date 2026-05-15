"""End-to-end tests for the FastAPI Phase 4 surface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from cardiorisk.api import build_app
from cardiorisk.rag.generation.generator import CitationGenerator
from cardiorisk.rag.generation.llm import MockLLMClient
from cardiorisk.rag.generation.nli import EntailmentResult
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


def _patient() -> dict[str, object]:
    return {
        "Age": 72,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 290,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 100,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.4,
        "ST_Slope": "Flat",
    }


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


def _generator() -> CitationGenerator:
    chunks = [
        _chunk(
            "c1",
            "Pharmacotherapy is recommended for high-risk patients.",
            doc_id="nvdpa",
        ),
        _chunk(
            "c2",
            "Lifestyle interventions are first-line for prevention.",
            doc_id="racgp",
        ),
    ]
    return CitationGenerator(
        retrieval_pipeline=_StubPipeline(chunks),  # type: ignore[arg-type]
        llm_client=MockLLMClient(),
        nli_verifier=_AlwaysEntails(),
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app(generator=_generator()))


class TestHealth:
    def test_healthz_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestCreateCase:
    def test_create_case_returns_triage_interrupt(self, client: TestClient) -> None:
        r = client.post(
            "/v1/cases",
            json={"case_id": "c1", "patient": _patient()},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["case_id"] == "c1"
        assert body["triage"] is not None
        assert body["next_interrupt"]["stage"] == "triage"
        assert body["terminated"] is False

    def test_create_case_rejects_invalid_patient(self, client: TestClient) -> None:
        bad = _patient()
        bad["Sex"] = "X"
        r = client.post("/v1/cases", json={"case_id": "bad", "patient": bad})
        assert r.status_code == 422

    def test_create_case_duplicate_id_returns_409(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "dup", "patient": _patient()})
        r = client.post("/v1/cases", json={"case_id": "dup", "patient": _patient()})
        assert r.status_code == 409


class TestDecide:
    def test_approve_advances_to_next_stage(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c2", "patient": _patient()})
        r = client.post(
            "/v1/cases/c2/decide",
            json={"decision": {"status": "approve"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["next_interrupt"]["stage"] == "risk"
        assert body["risk"] is not None
        assert len(body["decisions"]) == 1

    def test_full_happy_path_runs_to_termination(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c3", "patient": _patient()})
        approve = {"decision": {"status": "approve"}}
        for _ in range(4):
            r = client.post("/v1/cases/c3/decide", json=approve)
            assert r.status_code == 200
        body = r.json()
        assert body["next_interrupt"] is None
        assert body["terminated"] is False  # graph reached END normally
        assert body["letter"] is not None
        assert len(body["decisions"]) == 4
        assert len(body["audit"]) == 4

    def test_reject_sets_terminated_with_reason(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c4", "patient": _patient()})
        r = client.post(
            "/v1/cases/c4/decide",
            json={"decision": {"status": "reject", "reason": "incomplete history"}},
        )
        body = r.json()
        assert body["terminated"] is True
        assert "triage" in (body["termination_reason"] or "")

    def test_decide_rejects_invalid_decision_status(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c5", "patient": _patient()})
        r = client.post(
            "/v1/cases/c5/decide",
            json={"decision": {"status": "shrug"}},
        )
        assert r.status_code == 422

    def test_decide_for_unknown_case_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/v1/cases/does-not-exist/decide",
            json={"decision": {"status": "approve"}},
        )
        assert r.status_code == 404

    def test_decide_after_termination_returns_409(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c6", "patient": _patient()})
        approve = {"decision": {"status": "approve"}}
        for _ in range(4):
            client.post("/v1/cases/c6/decide", json=approve)
        # No active interrupt now.
        r = client.post("/v1/cases/c6/decide", json=approve)
        assert r.status_code == 409


class TestGetCase:
    def test_returns_full_state(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c7", "patient": _patient()})
        r = client.get("/v1/cases/c7")
        assert r.status_code == 200
        body = r.json()
        assert body["case_id"] == "c7"
        assert body["next_interrupt"]["stage"] == "triage"

    def test_get_unknown_case_returns_404(self, client: TestClient) -> None:
        r = client.get("/v1/cases/does-not-exist")
        assert r.status_code == 404

    def test_get_after_full_run_has_no_interrupt(self, client: TestClient) -> None:
        client.post("/v1/cases", json={"case_id": "c8", "patient": _patient()})
        for _ in range(4):
            client.post("/v1/cases/c8/decide", json={"decision": {"status": "approve"}})
        r = client.get("/v1/cases/c8")
        body = r.json()
        assert body["next_interrupt"] is None
        assert body["letter"] is not None
