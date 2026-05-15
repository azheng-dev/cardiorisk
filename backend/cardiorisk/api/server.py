"""FastAPI app factory + router.

Use :func:`build_app` to construct a fresh app instance with a custom
``CitationGenerator`` (for tests). The ``app`` module-level attribute
is the production handle that uvicorn imports::

    uvicorn cardiorisk.api.server:app --host 0.0.0.0 --port 8000

Production wires the real BGE-M3 + bge-reranker + DeBERTa NLI
verifier; tests construct their own with ``MockLLMClient`` /
``MockNLIVerifier`` / a stub pipeline.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI, HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cardiorisk.agents.graph import (
    build_graph,
    latest_interrupt,
    state_from_dict,
)
from cardiorisk.agents.state import AgentState
from cardiorisk.rag.generation.generator import CitationGenerator

from .schemas import (
    CaseCreate,
    CaseStateResponse,
    DecideRequest,
    DecideResponse,
    InterruptPayload,
)

API_PREFIX = "/v1"


def _payload_to_interrupt(itr: dict[str, Any] | None) -> InterruptPayload | None:
    if itr is None:
        return None
    stage_value = itr.get("stage")
    if stage_value is None:
        return None
    return InterruptPayload(stage=stage_value, artefact=itr.get("artefact"))


def build_app(
    *,
    generator: CitationGenerator,
    risk_model_name: str | None = None,
    risk_held_out_source: str | None = None,
    title: str = "CardioRisk Co-Pilot — Phase 4 agent API",
    version: str = "0.4.0",
) -> FastAPI:
    """Construct a FastAPI app bound to a fresh graph + checkpointer.

    The checkpointer is :class:`InMemorySaver`; cases live for the
    lifetime of the process. Phase 7 will swap in a Postgres-backed
    saver. The generator is supplied by the caller so production
    code wires the real LLM + NLI stack while tests inject mocks.
    """
    checkpointer = InMemorySaver()
    risk_kwargs: dict[str, Any] = {}
    if risk_model_name is not None:
        risk_kwargs["risk_model_name"] = risk_model_name
    if risk_held_out_source is not None:
        risk_kwargs["risk_held_out_source"] = risk_held_out_source
    graph = build_graph(generator=generator, checkpointer=checkpointer, **risk_kwargs)

    app = FastAPI(
        title=title,
        version=version,
        description=(
            "Synthetic-data-only research artefact. The /v1 surface "
            "exposes the 4-agent CardioRisk graph with HITL gates "
            "(triage → risk → guideline → letter). Not for clinical use."
        ),
    )

    def _config_for(case_id: str) -> RunnableConfig:
        # LangGraph's RunnableConfig is a TypedDict; building one
        # via cast keeps mypy happy without importing every key.
        return cast(RunnableConfig, {"configurable": {"thread_id": case_id}})

    @app.post(
        f"{API_PREFIX}/cases",
        response_model=CaseStateResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_case(payload: CaseCreate) -> CaseStateResponse:
        config = _config_for(payload.case_id)
        existing = graph.get_state(config)
        if existing.values:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"case {payload.case_id!r} already exists",
            )
        init = AgentState(case_id=payload.case_id, patient=payload.patient).model_dump()
        graph.invoke(cast(Any, init), config=config)
        snap = graph.get_state(config)
        state = state_from_dict(snap.values)
        return CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )

    @app.post(
        f"{API_PREFIX}/cases/{{case_id}}/decide",
        response_model=DecideResponse,
    )
    def decide(case_id: str, body: DecideRequest) -> DecideResponse:
        config = _config_for(case_id)
        snap = graph.get_state(config)
        if not snap.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"case {case_id!r} not found",
            )
        if latest_interrupt(snap) is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"case {case_id!r} has no active review gate",
            )
        decision_payload = body.decision.model_dump()
        graph.invoke(Command(resume=decision_payload), config=config)
        snap = graph.get_state(config)
        state = state_from_dict(snap.values)
        base = CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )
        return DecideResponse(**base.model_dump())

    @app.get(
        f"{API_PREFIX}/cases/{{case_id}}",
        response_model=CaseStateResponse,
    )
    def get_case(case_id: str) -> CaseStateResponse:
        config = _config_for(case_id)
        snap = graph.get_state(config)
        if not snap.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"case {case_id!r} not found",
            )
        state = state_from_dict(snap.values)
        return CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


__all__ = ["API_PREFIX", "build_app"]
