"""Phase 4 FastAPI surface for the agent graph.

This module is a thin adapter between HTTP and
:mod:`cardiorisk.agents.graph`. It exposes three endpoints:

- ``POST /v1/cases`` — create a new case from a :class:`PatientInput`
  payload. Returns ``{"case_id": ..., "next_stage": "triage"}`` with
  the freshly-paused triage_review interrupt waiting on a decision.
- ``POST /v1/cases/{case_id}/decide`` — submit an Approve / Edit /
  Reject decision for whichever review gate is currently paused.
  Returns the resulting state (with either the next interrupt or
  ``terminated=True``).
- ``GET /v1/cases/{case_id}`` — fetch the latest state of an
  in-flight or completed case (read-only).

Dependency wiring: the graph + checkpointer + ``CitationGenerator`` are
constructed once at app startup using :func:`build_app`. Tests inject
their own (with mock LLM/NLI/retrieval) via
``build_app(generator=...)``. There are no DB calls here — everything
lives in the LangGraph :class:`InMemorySaver` for Phase 4. Phase 7
revisits with a Postgres-backed checkpointer.

Phase 4 deliberately ships *no auth*. The disclaimer banner in the
README is the security model: the API serves synthetic patients from
a single research artefact; do not point this at real PHI.
"""

from .schemas import (
    CaseCreate,
    CaseStateResponse,
    DecideRequest,
    DecideResponse,
    InterruptPayload,
)
from .server import build_app

__all__ = [
    "CaseCreate",
    "CaseStateResponse",
    "DecideRequest",
    "DecideResponse",
    "InterruptPayload",
    "build_app",
]
