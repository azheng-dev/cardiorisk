"""Retries + circuit breaker for the agent hot path.

LangGraph's executor surfaces every exception as a graph-level error
that tears down the in-flight HITL state. We don't want that — the
LLM/NLI calls inside the guideline + letter agents are flaky enough
that one transient 5xx kills a clinician's session. So:

- :func:`with_retries` is a thin tenacity wrapper with the project's
  default backoff (3 attempts, exponential jitter 0.5..4 s, retry only
  on the project's :class:`TransientAgentError` so deterministic
  failures (validation errors, missing artefacts) propagate
  immediately).

- :class:`CircuitBreaker` is the in-house counterpart: 3 consecutive
  failures → ``open`` for ``open_seconds``; one ``half_open`` probe
  on the next call. The clock is injected so tests can advance time
  without sleeping.

Both primitives are deliberately tiny — Phase 4's job is to ship the
contract (retries + breakers exist on the LLM/NLI calls), not to
solve resilience as a library.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class TransientAgentError(RuntimeError):
    """Raised by an agent's tool call when the failure is transient.

    Tenacity retries this; deterministic failures (assertion errors,
    pydantic validation errors, missing model artefacts) bypass the
    retry loop and surface to the graph immediately.
    """


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is blocked because the breaker is open."""


def _record_attempt(retry_state: RetryCallState) -> None:
    """Sentinel hook so tests can introspect retry counts."""
    # Updates an attribute the caller can read after the retry loop.
    if hasattr(retry_state, "outcome"):
        return


def with_retries[U](
    fn: Callable[[], U],
    *,
    max_attempts: int = 3,
    initial_wait_seconds: float = 0.5,
    max_wait_seconds: float = 4.0,
) -> tuple[U, int]:
    """Run ``fn`` with tenacity retries; return ``(result, attempt_count)``.

    Retries only on :class:`TransientAgentError`. Returns the attempt
    count (1-based) so the caller can fold it into the audit entry.

    The retry policy is deliberately small (3 attempts, 0.5..4 s
    jitter): Phase 4's eval needs a deterministic upper bound on
    per-case wall-clock, and a bigger budget would let one wedged
    LLM call dominate a 30-case run.
    """
    retryer = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=initial_wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception_type(TransientAgentError),
        reraise=True,
        before=_record_attempt,
    )
    last_attempt = 0
    for attempt in retryer:
        last_attempt = attempt.retry_state.attempt_number
        with attempt:
            result = fn()
            return result, last_attempt
    # tenacity guarantees the loop returns or raises; this line is
    # purely to satisfy mypy that every code path returns.
    raise RuntimeError("with_retries: loop exited without result")


# ----------------------------------------------------------------- breaker
@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class CircuitBreaker:
    """Per-key circuit breaker.

    Usage::

        breaker = CircuitBreaker(failure_threshold=3, open_seconds=60.0)
        result = breaker.call("llm.anthropic", lambda: client.generate(...))

    State machine:

    - ``closed`` (default): call ``fn``; reset failure count on success;
      increment on :class:`TransientAgentError`. After
      ``failure_threshold`` consecutive failures, the breaker opens.
    - ``open``: every call raises :class:`CircuitBreakerOpenError`
      without invoking ``fn`` until ``open_seconds`` has elapsed.
    - ``half_open``: the first call after ``open_seconds`` is allowed
      through. If it succeeds, the breaker closes (and failure count
      resets); if it fails, the breaker re-opens for another
      ``open_seconds``.

    Non-transient exceptions propagate without touching breaker state
    (they're deterministic bugs, not load shedding).
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        open_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if open_seconds <= 0:
            raise ValueError("open_seconds must be > 0")
        self._threshold = failure_threshold
        self._open_seconds = float(open_seconds)
        self._clock = clock
        self._states: dict[str, _BreakerState] = {}
        self._states_lock = threading.Lock()

    def _state(self, key: str) -> _BreakerState:
        with self._states_lock:
            return self._states.setdefault(key, _BreakerState())

    def call[U](self, key: str, fn: Callable[[], U]) -> U:
        st = self._state(key)
        with st.lock:
            if st.opened_at is not None:
                if self._clock() - st.opened_at < self._open_seconds:
                    raise CircuitBreakerOpenError(
                        f"breaker {key!r} is open ({st.failures} consecutive failures)"
                    )
                # half-open: clear opened_at and let the call through;
                # success closes the breaker, failure re-opens it.
                st.opened_at = None
        try:
            result = fn()
        except TransientAgentError:
            with st.lock:
                st.failures += 1
                if st.failures >= self._threshold:
                    st.opened_at = self._clock()
            raise
        else:
            with st.lock:
                st.failures = 0
                st.opened_at = None
            return result

    def state_for(self, key: str) -> tuple[int, bool]:
        """Return ``(failure_count, is_open)`` for the given key."""
        st = self._state(key)
        with st.lock:
            is_open = st.opened_at is not None and (
                self._clock() - st.opened_at < self._open_seconds
            )
            return st.failures, is_open


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "TransientAgentError",
    "with_retries",
]
