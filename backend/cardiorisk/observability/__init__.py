"""Observability primitives — Langfuse + Sentry, both env-var gated.

Every helper here is **a no-op when the relevant credential is
unset**. CI runs without any keys; production wires them in via the
platform env. Modules that want a span call :func:`observe_node` or
:func:`record_generation`; modules that want error tracking call
:func:`init_sentry` once at app boot. Nothing in this package raises
when credentials are missing — the wrapper either resolves to the
real SDK call or to a stub that swallows the call.

See [ADR-024](../../../docs/adr/024-observability-free-tier.md) for
the binding decision (Langfuse Cloud Hobby + Sentry Free + Vercel
Web Analytics + Speed Insights — all permanent free tier per AGENTS
§4) and `docs/research/20-observability-design.md` for the design
rationale.
"""

from __future__ import annotations

from .langfuse import (
    LANGFUSE_GENERATION_INPUT_LIMIT,
    LANGFUSE_GENERATION_OUTPUT_LIMIT,
    flush_langfuse,
    get_current_trace_id,
    get_langfuse_client,
    new_trace_id,
    observe_node,
    record_generation,
    start_root_span,
)
from .sentry import init_sentry, scrub_patient

__all__ = [
    "LANGFUSE_GENERATION_INPUT_LIMIT",
    "LANGFUSE_GENERATION_OUTPUT_LIMIT",
    "flush_langfuse",
    "get_current_trace_id",
    "get_langfuse_client",
    "init_sentry",
    "new_trace_id",
    "observe_node",
    "record_generation",
    "scrub_patient",
    "start_root_span",
]
