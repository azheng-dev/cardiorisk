"""Langfuse wrappers — single source of truth for LLM tracing.

The Langfuse SDK ships its own ``@observe`` decorator and
``get_client()`` factory. We wrap those so:

1. **Missing credentials are not an error.** When ``LANGFUSE_*`` env
   vars are unset (CI, forks, any local dev that hasn't opted in),
   every helper here resolves to a no-op. ``get_langfuse_client()``
   returns ``None``; ``observe_node`` becomes an identity decorator;
   ``record_generation`` returns silently; ``start_root_span`` yields
   a stub context manager.

2. **Trace IDs are always synthesised.** Even in no-op mode we mint
   a deterministic ``mock-trace-<8-hex>`` id so the API can hand a
   trace_id to the UI without a network round-trip. When Langfuse
   is enabled, the real OTel trace_id replaces the mock.

3. **The SDK import is lazy.** We never import ``langfuse`` at
   module load — only inside the helpers, gated on
   :attr:`Settings.langfuse_enabled`. Pytest collection stays fast
   and the OTel background thread doesn't get installed in CI.

Pinned to Langfuse Python SDK v3+/v4 (the OTel-based rewrite). v2
is end-of-life as of 2026-03 and is not API-compatible.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from cardiorisk.settings import get_settings

#: Cap the per-generation prompt + completion attached to a span. The
#: Langfuse Cloud Hobby tier allows large strings, but the eval
#: harness can run thousands of generations a day — capping keeps the
#: storage bill nominal and the spans readable.
LANGFUSE_GENERATION_INPUT_LIMIT: int = 8_000
LANGFUSE_GENERATION_OUTPUT_LIMIT: int = 2_000

F = TypeVar("F", bound=Callable[..., Any])


def get_langfuse_client() -> Any | None:
    """Return the lazy Langfuse singleton or ``None`` if disabled.

    Cached implicitly by the SDK's own ``get_client`` (it's idempotent).
    Returns ``None`` when credentials are missing so callers can
    short-circuit without try/except.
    """
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import get_client as _get_client
    except ImportError:
        return None
    try:
        return _get_client()
    except Exception:  # pragma: no cover - SDK init failure must not break app
        return None


def observe_node(*, stage: str) -> Callable[[F], F]:
    """Wrap an agent-graph node so its span shows up in Langfuse.

    The wrapper:

    - Tags the span ``name=agent.<stage>`` and ``metadata.stage=<stage>``
      so the Langfuse UI groups runs by stage at a glance.
    - Passes through arguments + return value untouched.
    - In no-op mode, becomes an identity decorator (no behaviour change).

    Usage::

        @observe_node(stage="triage")
        def triage_node(state: AgentState) -> dict[str, Any]:
            ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_langfuse_client()
            if client is None:
                return fn(*args, **kwargs)
            case_id = _extract_case_id(args, kwargs)
            try:
                with client.start_as_current_observation(
                    as_type="span",
                    name=f"agent.{stage}",
                    metadata={"stage": stage, "case_id": case_id},
                ) as span:
                    result = fn(*args, **kwargs)
                    # The node returns a dict of state updates; surface
                    # an "ok" output marker rather than dumping state
                    # (state can be large; the per-stage artefacts will
                    # be uploaded explicitly elsewhere if we ever need
                    # them).
                    span.update(
                        output={"keys": sorted(result.keys()) if isinstance(result, dict) else None}
                    )
                    return result
            except Exception:  # pragma: no cover - never let observability break the call
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def record_generation(
    *,
    model: str,
    prompt: str,
    completion: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    client_name: str,
) -> None:
    """Emit a Langfuse ``generation`` event for one LLM call.

    Called from inside each LLM client's ``generate()`` *after* the
    usage totals are updated. The span attaches model, token counts,
    cost, and (capped) prompt + completion bodies — matching the
    Langfuse generation schema used by every native integration.

    No-op when Langfuse is disabled.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        # Use the manual start_observation path so we don't change
        # the active OTel context — generations are leaves, not parents.
        gen = client.start_observation(
            as_type="generation",
            name=f"llm.{client_name}",
            model=model,
            input=prompt[:LANGFUSE_GENERATION_INPUT_LIMIT],
            metadata={"client_name": client_name},
        )
        gen.update(
            output=completion[:LANGFUSE_GENERATION_OUTPUT_LIMIT],
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            cost_details={
                "input": None,
                "output": None,
                "total": float(cost_usd),
            },
        )
        gen.end()
    except Exception:  # pragma: no cover - observability MUST NOT break the LLM call
        return


@contextmanager
def start_root_span(*, name: str, case_id: str | None = None) -> Iterator[str | None]:
    """Open a root span for a case run and yield its trace_id.

    Used by:
    - :func:`cardiorisk.api.server.create_case` — wrap one HTTP
      request so the FastAPI route shows up as a single trace with
      every agent node nested inside.
    - :func:`cardiorisk.agents.eval.orchestrator.run_eval` — wrap
      each case so the 100-case batch run produces 100 sibling
      traces, each with the same trace_id stamped on the response.

    Yields ``None`` when Langfuse is disabled (callers fall back to
    :func:`new_trace_id` for a synthetic id).
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata={"case_id": case_id} if case_id else {},
        ) as span:
            trace_id = _safe_trace_id(span)
            yield trace_id
    except Exception:  # pragma: no cover - observability never breaks the call
        yield None


def get_current_trace_id() -> str | None:
    """Return the active Langfuse trace_id, or ``None``.

    Reads the OTel context. Useful in deep-nested code that wants to
    return the trace_id back to the caller without threading a span
    object through every function.
    """
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        tid = client.get_current_trace_id()
    except Exception:  # pragma: no cover
        return None
    return tid if isinstance(tid, str) and tid else None


def new_trace_id() -> str:
    """Mint a synthetic trace_id for no-op mode.

    Format: ``mock-trace-<16-hex>``. Prefixed so the UI can render a
    muted "Local mock — no remote trace" badge instead of a broken
    Langfuse deep-link.
    """
    return f"mock-trace-{secrets.token_hex(8)}"


def flush_langfuse() -> None:
    """Force-flush pending Langfuse events.

    Eval scripts call this at the end of the run so short-lived
    processes (CI, ``--smoke`` runs) don't lose data when they exit
    before the background thread drains.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # pragma: no cover
        return


def _extract_case_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Best-effort case_id extraction for span metadata.

    The agent-graph nodes take an ``AgentState`` as their first
    positional argument; we read ``.case_id`` if present. Falls back
    to ``None`` so the wrapper still works on call sites that don't
    fit the pattern (eval orchestrator passes ``case_id`` as a
    keyword).
    """
    if "case_id" in kwargs and isinstance(kwargs["case_id"], str):
        return kwargs["case_id"]
    if args:
        case_id = getattr(args[0], "case_id", None)
        if isinstance(case_id, str):
            return case_id
    return None


def _safe_trace_id(span: Any) -> str | None:
    """Read the trace_id off a Langfuse span without breaking on shape drift."""
    for attr in ("trace_id", "_trace_id"):
        val = getattr(span, attr, None)
        if isinstance(val, str) and val:
            return val
    # v3/v4 stores it on the OTel span context.
    ctx = getattr(span, "_otel_span", None)
    if ctx is not None:
        ctx_ctx = getattr(ctx, "context", None) or getattr(ctx, "get_span_context", lambda: None)()
        trace_id_int = getattr(ctx_ctx, "trace_id", None)
        if isinstance(trace_id_int, int) and trace_id_int:
            return f"{trace_id_int:032x}"
    return get_current_trace_id()


__all__ = [
    "LANGFUSE_GENERATION_INPUT_LIMIT",
    "LANGFUSE_GENERATION_OUTPUT_LIMIT",
    "flush_langfuse",
    "get_current_trace_id",
    "get_langfuse_client",
    "new_trace_id",
    "observe_node",
    "record_generation",
    "start_root_span",
]
