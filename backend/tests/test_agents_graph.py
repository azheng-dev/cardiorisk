"""End-to-end tests for the LangGraph 4-agent graph.

Uses the same stub-pipeline pattern as ``test_rag_generation_generator``
to keep the dependency surface tiny: no embedder + no NLI weights, just
the deterministic mocks. Verifies:

- The graph pauses at every HITL gate and resumes correctly.
- Approve / edit / reject decisions all propagate through the audit
  log + the conditional edges.
- The final state carries the expected sequence of stages, decisions,
  and audit entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from cardiorisk.agents.graph import build_graph, latest_interrupt
from cardiorisk.agents.state import (
    AgentState,
    ApproveDecision,
    EditDecision,
    PatientInput,
    RejectDecision,
)
from cardiorisk.rag.generation.generator import CitationGenerator
from cardiorisk.rag.generation.llm import MockLLMClient
from cardiorisk.rag.generation.nli import EntailmentResult
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


# ----------------------------------------------------------------- helpers
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


def _make_generator() -> CitationGenerator:
    chunks = [
        _chunk(
            "c1",
            "Pharmacotherapy is recommended for patients at high CVD risk.",
            doc_id="nvdpa",
        ),
        _chunk(
            "c2",
            "Lifestyle interventions remain first-line for cardiovascular prevention.",
            doc_id="racgp",
        ),
    ]
    return CitationGenerator(
        retrieval_pipeline=_StubPipeline(chunks),  # type: ignore[arg-type]
        llm_client=MockLLMClient(),
        nli_verifier=_AlwaysEntails(),
    )


def _empty_models(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


# ----------------------------------------------------------------- tests
class TestGraphHappyPath:
    def test_full_run_through_all_four_stages(self, tmp_path: Path) -> None:
        graph = build_graph(generator=_make_generator())
        models_dir = _empty_models(tmp_path)
        del models_dir  # risk agent uses MODELS_V1_DIR by default; it's fine if empty

        config = cast(RunnableConfig, {"configurable": {"thread_id": "case-001"}})
        init = AgentState(case_id="case-001", patient=_patient()).model_dump()

        # Stage 1: triage runs, then graph pauses at triage_review
        graph.invoke(cast(Any, init), config=config)
        snap = graph.get_state(config)
        itr = latest_interrupt(snap)
        assert itr is not None
        assert itr["stage"] == "triage"

        approve = ApproveDecision().model_dump()
        graph.invoke(Command(resume=approve), config=config)
        snap = graph.get_state(config)
        itr = latest_interrupt(snap)
        assert itr is not None
        assert itr["stage"] == "risk"

        graph.invoke(Command(resume=approve), config=config)
        snap = graph.get_state(config)
        itr = latest_interrupt(snap)
        assert itr is not None
        assert itr["stage"] == "guideline"

        graph.invoke(Command(resume=approve), config=config)
        snap = graph.get_state(config)
        itr = latest_interrupt(snap)
        assert itr is not None
        assert itr["stage"] == "letter"

        # Final approve takes us past the last review to END.
        final_state_dict = graph.invoke(Command(resume=approve), config=config)
        snap = graph.get_state(config)
        assert latest_interrupt(snap) is None

        # State carries 4 stages + 4 approve decisions + 4 audit entries.
        assert final_state_dict["triage"] is not None
        assert final_state_dict["risk"] is not None
        assert final_state_dict["guideline"] is not None
        assert final_state_dict["letter"] is not None
        assert len(final_state_dict["decisions"]) == 4
        assert len(final_state_dict["audit"]) == 4


class TestGraphRejectionPath:
    @pytest.mark.parametrize(
        "stages_to_pass,reject_stage",
        [
            (0, "triage"),
            (1, "risk"),
            (2, "guideline"),
            (3, "letter"),
        ],
    )
    def test_reject_at_any_stage_terminates(
        self, tmp_path: Path, stages_to_pass: int, reject_stage: str
    ) -> None:
        del tmp_path  # risk agent's mock fallback works with no models
        graph = build_graph(generator=_make_generator())
        config = cast(RunnableConfig, {"configurable": {"thread_id": f"case-rej-{reject_stage}"}})
        init = AgentState(case_id=f"case-rej-{reject_stage}", patient=_patient()).model_dump()

        approve = ApproveDecision().model_dump()
        reject = RejectDecision(reason=f"unit test reject at {reject_stage}").model_dump()

        graph.invoke(cast(Any, init), config=config)
        for _ in range(stages_to_pass):
            graph.invoke(Command(resume=approve), config=config)

        result = graph.invoke(Command(resume=reject), config=config)
        assert result.get("terminated") is True
        assert reject_stage in (result.get("termination_reason") or "")
        # The audit + decisions should reflect every stage we ran through.
        assert len(result["decisions"]) == stages_to_pass + 1


class TestGraphEditPath:
    def test_edit_decision_at_triage_overrides_summary(self) -> None:
        graph = build_graph(generator=_make_generator())
        config = cast(RunnableConfig, {"configurable": {"thread_id": "case-edit-1"}})
        init = AgentState(case_id="case-edit-1", patient=_patient()).model_dump()

        graph.invoke(cast(Any, init), config=config)

        edit = EditDecision(edits={"summary": "edited triage summary"}).model_dump()
        graph.invoke(Command(resume=edit), config=config)
        snap = graph.get_state(config)

        # The patched triage.summary should be visible in the resumed state.
        triage_payload = snap.values["triage"]
        # State.values contains pydantic models if reconstituted; the dict form
        # also works (LangGraph round-trips through JSON depending on adapter).
        if hasattr(triage_payload, "summary"):
            assert triage_payload.summary == "edited triage summary"
        else:
            assert triage_payload["summary"] == "edited triage summary"

    def test_edit_decision_at_risk_is_recorded_but_artefact_unchanged(self) -> None:
        graph = build_graph(generator=_make_generator())
        config = cast(RunnableConfig, {"configurable": {"thread_id": "case-edit-2"}})
        init = AgentState(case_id="case-edit-2", patient=_patient()).model_dump()

        approve = ApproveDecision().model_dump()
        graph.invoke(cast(Any, init), config=config)
        graph.invoke(Command(resume=approve), config=config)  # past triage

        # Try to edit the risk artefact.
        edit = EditDecision(edits={"probability": 0.99, "risk_band": "high"}).model_dump()
        graph.invoke(Command(resume=edit), config=config)
        snap = graph.get_state(config)

        # The decision is in the audit but the risk artefact stays
        # whatever the model emitted (Phase-4 design: the risk artefact
        # is model-derived and intentionally not user-editable).
        risk_payload = snap.values["risk"]
        prob = (
            risk_payload.probability
            if hasattr(risk_payload, "probability")
            else risk_payload["probability"]
        )
        assert prob != 0.99
