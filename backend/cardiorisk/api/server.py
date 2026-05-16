"""FastAPI app factory + router (Phase 4; reshaped in Phase 7).

Use :func:`build_app` to construct a fresh app instance with a custom
``CitationGenerator`` (for tests). The ``app`` module-level attribute
is the production handle that uvicorn imports::

    uvicorn cardiorisk.api.server:app --host 0.0.0.0 --port 8000

Production wires the real BGE-M3 + bge-reranker + DeBERTa NLI
verifier; tests construct their own with ``MockLLMClient`` /
``MockNLIVerifier`` / a stub pipeline.

Phase 7 changes (ADR-024):

- Endpoints moved from ``/v1`` to ``/v1/agents`` so the FE client
  (which already assumed this prefix) works against the live API.
- CORS middleware wired in (origins read from ``CORS_ALLOW_ORIGINS``).
- Sentry initialised (env-var gated; no-op without DSN).
- Each :func:`create_case` opens a Langfuse root span and:
  - auto-mints a ``case_id`` if the caller omitted one,
  - stamps the resulting trace_id onto :class:`AgentState`,
  - emits the trace_id back as an ``X-Trace-Id`` response header,
  - auto-approves the (deterministic, rule-based) triage gate so the
    UI lands on the first clinically-meaningful gate (risk) without
    a wasted round-trip.
- :class:`DecideRequest` uses the flat ``{stage, status, note?}`` shape
  the frontend sends.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cardiorisk.agents.graph import (
    build_graph,
    latest_interrupt,
    state_from_dict,
)
from cardiorisk.agents.state import (
    AgentStage,
    AgentState,
    ApproveDecision,
    EditDecision,
    RejectDecision,
)
from cardiorisk.observability import (
    init_sentry,
    new_trace_id,
    start_root_span,
)
from cardiorisk.rag.generation.generator import CitationGenerator
from cardiorisk.settings import get_settings

from .schemas import (
    CaseCreate,
    CaseStateResponse,
    DecideRequest,
    DecideResponse,
    InterruptPayload,
)

#: Public path prefix. The Phase 5.3 frontend client already calls
#: under ``/v1/agents/cases``; Phase 7 aligns the backend with that.
API_PREFIX = "/v1/agents"


def _payload_to_interrupt(itr: dict[str, Any] | None) -> InterruptPayload | None:
    if itr is None:
        return None
    stage_value = itr.get("stage")
    if stage_value is None:
        return None
    return InterruptPayload(stage=stage_value, artefact=itr.get("artefact"))


def _mint_case_id() -> str:
    """Mint a fresh case id when the caller omits one.

    Format: ``c{8-char-hex}`` — short enough for URLs, long enough
    to collide only every ~4 billion cases.
    """
    return f"c{uuid4().hex[:8]}"


def _decision_from_request(body: DecideRequest) -> ApproveDecision | EditDecision | RejectDecision:
    """Translate the flat FE-shape ``DecideRequest`` to the internal
    :class:`Decision` discriminated union the graph expects.

    Maps:
    - ``approved`` → :class:`ApproveDecision`
    - ``edited``   → :class:`EditDecision` (with empty ``edits``;
      partial edits land in Phase 5.3+)
    - ``rejected`` → :class:`RejectDecision` (``note`` becomes ``reason``)
    """
    if body.status == "approved":
        return ApproveDecision(note=body.note) if body.note else ApproveDecision()
    if body.status == "edited":
        return EditDecision(note=body.note) if body.note else EditDecision()
    # rejected: the schema requires a non-empty reason
    reason = body.note or "rejected via API"
    return RejectDecision(reason=reason, note=body.note)


def build_app(
    *,
    generator: CitationGenerator,
    risk_model_name: str | None = None,
    risk_held_out_source: str | None = None,
    title: str = "CardioRisk Co-Pilot — Phase 7 agent API",
    version: str = "0.7.0",
) -> FastAPI:
    """Construct a FastAPI app bound to a fresh graph + checkpointer.

    The checkpointer is :class:`InMemorySaver`; cases live for the
    lifetime of the process. Phase 8 will swap in a Supabase-backed
    saver. The generator is supplied by the caller so production
    code wires the real LLM + NLI stack while tests inject mocks.
    """
    settings = get_settings()
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
            "Synthetic-data-only research artefact. The /v1/agents "
            "surface exposes the 4-agent CardioRisk graph with HITL "
            "gates (triage → risk → guideline → letter). Not for "
            "clinical use."
        ),
    )

    # CORS — required for the Vercel-hosted frontend to call the
    # HF Spaces-hosted backend cross-origin. ``CORS_ALLOW_ORIGINS=*``
    # is the dev default; production sets it to the Vercel origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    # Sentry — no-op when SENTRY_DSN is unset (CI, local dev).
    init_sentry(app=app)

    def _config_for(case_id: str) -> RunnableConfig:
        # LangGraph's RunnableConfig is a TypedDict; building one
        # via cast keeps mypy happy without importing every key.
        return cast(RunnableConfig, {"configurable": {"thread_id": case_id}})

    def _resume_with(case_id: str, payload: dict[str, Any]) -> None:
        graph.invoke(Command(resume=payload), config=_config_for(case_id))

    @app.post(
        f"{API_PREFIX}/cases",
        response_model=CaseStateResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_case(payload: CaseCreate, response: Response) -> CaseStateResponse:
        case_id = payload.case_id or _mint_case_id()
        config = _config_for(case_id)
        existing = graph.get_state(config)
        if existing.values:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"case {case_id!r} already exists",
            )

        # Open the per-case Langfuse root span so every agent node
        # nests under one trace. Auto-approve the rule-based triage
        # gate so the UI lands on the risk screen on first paint.
        with start_root_span(name=f"case[{case_id}]", case_id=case_id) as trace_id:
            init = AgentState(
                case_id=case_id,
                patient=payload.patient,
                trace_id=trace_id or new_trace_id(),
            ).model_dump()
            graph.invoke(cast(Any, init), config=config)
            # Auto-approve triage so the UI's first interactive
            # screen is the risk dashboard (no dedicated triage
            # screen exists; the triage agent is deterministic and
            # rule-based, so a clinician adds no value at this gate).
            snap_after_triage = graph.get_state(config)
            interrupt_after_triage = latest_interrupt(snap_after_triage)
            if (
                interrupt_after_triage is not None
                and interrupt_after_triage.get("stage") == AgentStage.triage.value
            ):
                _resume_with(case_id, ApproveDecision(actor="system-auto").model_dump())

        snap = graph.get_state(config)
        state = state_from_dict(snap.values)
        body = CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )
        if body.trace_id:
            response.headers["X-Trace-Id"] = body.trace_id
        return body

    @app.post(
        f"{API_PREFIX}/cases/{{case_id}}/decide",
        response_model=DecideResponse,
    )
    def decide(case_id: str, body: DecideRequest, response: Response) -> DecideResponse:
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
        decision = _decision_from_request(body)
        graph.invoke(Command(resume=decision.model_dump()), config=config)
        snap = graph.get_state(config)
        state = state_from_dict(snap.values)
        base = CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )
        if base.trace_id:
            response.headers["X-Trace-Id"] = base.trace_id
        return DecideResponse(**base.model_dump())

    @app.get(
        f"{API_PREFIX}/cases/{{case_id}}",
        response_model=CaseStateResponse,
    )
    def get_case(case_id: str, response: Response) -> CaseStateResponse:
        config = _config_for(case_id)
        snap = graph.get_state(config)
        if not snap.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"case {case_id!r} not found",
            )
        state = state_from_dict(snap.values)
        body = CaseStateResponse.from_state(
            state, next_interrupt=_payload_to_interrupt(latest_interrupt(snap))
        )
        if body.trace_id:
            response.headers["X-Trace-Id"] = body.trace_id
        return body

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


__all__ = ["API_PREFIX", "build_app"]
