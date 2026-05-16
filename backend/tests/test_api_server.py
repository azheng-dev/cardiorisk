"""End-to-end tests for the FastAPI Phase 4 surface (reshaped in Phase 7).

Covers the API contract the Phase 5.3 frontend already assumes:

- Endpoints under ``/v1/agents/cases``.
- ``case_id`` optional on create (server mints ``c{8-hex}``).
- Flat ``DecideRequest`` shape: ``{stage, status, note?}``.
- Response shape: ``status`` / ``next_stage`` / ``trace_id`` / flat
  ``decisions``.
- ``X-Trace-Id`` response header round-trips per case.
- Triage gate auto-approved on create; UI lands on the risk gate.
"""

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
    def test_create_auto_approves_triage_and_lands_on_risk(self, client: TestClient) -> None:
        r = client.post(
            "/v1/agents/cases",
            json={"case_id": "c1", "patient": _patient()},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["case_id"] == "c1"
        # Triage ran + was auto-approved; we should be paused on
        # the risk gate now.
        assert body["status"] == "awaiting_decision"
        assert body["next_stage"] == "risk"
        assert body["triage"] is not None
        assert body["risk"] is not None
        # One auto-approved triage decision.
        assert len(body["decisions"]) == 1
        assert body["decisions"][0] == {
            "stage": "triage",
            "status": "approved",
            "note": None,
            "timestamp": body["decisions"][0]["timestamp"],
        }
        # trace_id always populated (mock sentinel in CI; real Langfuse otherwise)
        assert body["trace_id"] is not None
        assert r.headers.get("x-trace-id") == body["trace_id"]

    def test_create_mints_case_id_when_omitted(self, client: TestClient) -> None:
        r = client.post("/v1/agents/cases", json={"patient": _patient()})
        assert r.status_code == 201
        body = r.json()
        assert body["case_id"].startswith("c")
        assert len(body["case_id"]) == 9  # 'c' + 8 hex

    def test_create_rejects_invalid_patient(self, client: TestClient) -> None:
        bad = _patient()
        bad["Sex"] = "X"
        r = client.post("/v1/agents/cases", json={"case_id": "bad", "patient": bad})
        assert r.status_code == 422

    def test_create_duplicate_id_returns_409(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "dup", "patient": _patient()})
        r = client.post("/v1/agents/cases", json={"case_id": "dup", "patient": _patient()})
        assert r.status_code == 409


class TestDecide:
    def test_approve_risk_advances_to_guideline(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c2", "patient": _patient()})
        r = client.post(
            "/v1/agents/cases/c2/decide",
            json={"stage": "risk", "status": "approved"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "awaiting_decision"
        assert body["next_stage"] == "guideline"
        assert body["guideline"] is not None
        # 2 decisions: triage (auto) + risk (user)
        assert len(body["decisions"]) == 2
        assert body["decisions"][-1]["stage"] == "risk"
        assert body["decisions"][-1]["status"] == "approved"

    def test_full_happy_path_runs_to_complete(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c3", "patient": _patient()})
        # Triage auto-approved on create. UI sends 3 approves
        # (risk → guideline → letter) to drive to completion.
        for stage in ("risk", "guideline", "letter"):
            r = client.post(
                "/v1/agents/cases/c3/decide",
                json={"stage": stage, "status": "approved"},
            )
            assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        assert body["next_stage"] is None
        assert body["letter"] is not None
        # 4 decisions total: triage (auto) + 3 user
        assert len(body["decisions"]) == 4
        assert len(body["audit"]) == 4

    def test_reject_sets_status_rejected(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c4", "patient": _patient()})
        r = client.post(
            "/v1/agents/cases/c4/decide",
            json={"stage": "risk", "status": "rejected", "note": "incomplete history"},
        )
        body = r.json()
        assert body["status"] == "rejected"
        assert body["next_stage"] is None
        # The rejection note round-trips on the decisions list.
        assert body["decisions"][-1]["note"] == "incomplete history"

    def test_decide_rejects_invalid_status(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c5", "patient": _patient()})
        r = client.post(
            "/v1/agents/cases/c5/decide",
            json={"stage": "risk", "status": "shrug"},
        )
        assert r.status_code == 422

    def test_decide_for_unknown_case_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/v1/agents/cases/does-not-exist/decide",
            json={"stage": "risk", "status": "approved"},
        )
        assert r.status_code == 404

    def test_decide_after_completion_returns_409(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c6", "patient": _patient()})
        for stage in ("risk", "guideline", "letter"):
            client.post(
                "/v1/agents/cases/c6/decide",
                json={"stage": stage, "status": "approved"},
            )
        # No active interrupt now.
        r = client.post(
            "/v1/agents/cases/c6/decide",
            json={"stage": "letter", "status": "approved"},
        )
        assert r.status_code == 409

    def test_decide_preserves_trace_id_header(self, client: TestClient) -> None:
        create = client.post("/v1/agents/cases", json={"case_id": "c7", "patient": _patient()})
        trace_id = create.json()["trace_id"]
        r = client.post(
            "/v1/agents/cases/c7/decide",
            json={"stage": "risk", "status": "approved"},
        )
        assert r.headers.get("x-trace-id") == trace_id


class TestGetCase:
    def test_returns_full_state(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c8", "patient": _patient()})
        r = client.get("/v1/agents/cases/c8")
        assert r.status_code == 200
        body = r.json()
        assert body["case_id"] == "c8"
        assert body["status"] == "awaiting_decision"
        assert body["next_stage"] == "risk"

    def test_get_unknown_case_returns_404(self, client: TestClient) -> None:
        r = client.get("/v1/agents/cases/does-not-exist")
        assert r.status_code == 404

    def test_get_after_full_run_has_no_next_stage(self, client: TestClient) -> None:
        client.post("/v1/agents/cases", json={"case_id": "c9", "patient": _patient()})
        for stage in ("risk", "guideline", "letter"):
            client.post(
                "/v1/agents/cases/c9/decide",
                json={"stage": stage, "status": "approved"},
            )
        r = client.get("/v1/agents/cases/c9")
        body = r.json()
        assert body["status"] == "complete"
        assert body["next_stage"] is None
        assert body["letter"] is not None


class TestCors:
    def test_options_preflight_includes_allow_origin(self, client: TestClient) -> None:
        r = client.options(
            "/v1/agents/cases",
            headers={
                "origin": "https://example.com",
                "access-control-request-method": "POST",
            },
        )
        # CORSMiddleware answers preflight requests with 200.
        assert r.status_code == 200
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}
