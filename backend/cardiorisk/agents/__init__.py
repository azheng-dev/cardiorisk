"""Phase 4 — LangGraph 4-agent orchestration with HITL gates.

This package wires the v1 risk model (Phase 2.3b-2.5) and the
citation-mandatory generator (Phase 3.3) into a 4-agent workflow under
LangGraph. Each agent transition is gated by a structured
human-in-the-loop (HITL) decision (approve / edit / reject). The graph
is paused at every gate via ``langgraph.types.interrupt``; the FastAPI
surface (``cardiorisk.api``) resumes the graph with a structured
``Decision``.

Module map::

    state.py          AgentState (pydantic) + Decision union types +
                      AgentStage enum + AgentArtefact union
    triage.py         Validates + normalises a (synthetic) patient
                      payload to the HFP feature schema; emits
                      sanity flags
    risk.py           Loads the v1 TabICL artefact (joblib) when
                      present, otherwise a deterministic mock
                      classifier; emits calibrated probability +
                      top-k SHAP-style attributions
    guideline.py      Builds a clinician-style question from
                      patient + risk context; calls
                      ``CitationGenerator.generate``; passes the
                      verified answer + suppression audit through
                      the HITL gate
    letter.py         Drafts a referral letter using the verified
                      claims + risk + attributions; runs the same
                      verifier-in-the-loop discipline as
                      ``CitationGenerator``
    retries.py        ``with_retries`` decorator (tenacity-backed)
                      + ``CircuitBreaker`` (in-house: 3 consecutive
                      failures → open for 60s; deterministic clock
                      hook for tests)
    graph.py          ``StateGraph`` wiring; HITL interrupts after
                      every agent node; ``MemorySaver`` checkpointer

Design rationale + rejected alternatives are documented in
:mod:`docs/adr/018-agent-orchestration.md` and
:mod:`docs/research/15-agent-design.md`.
"""
