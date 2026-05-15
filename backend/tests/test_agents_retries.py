"""Tests for ``cardiorisk.agents.retries``: tenacity retry + circuit breaker."""

from __future__ import annotations

import time

import pytest

from cardiorisk.agents.retries import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    TransientAgentError,
    with_retries,
)


class TestWithRetries:
    def test_succeeds_first_try(self) -> None:
        calls = {"n": 0}

        def _fn() -> str:
            calls["n"] += 1
            return "ok"

        result, attempts = with_retries(_fn, max_attempts=3, initial_wait_seconds=0.0)
        assert result == "ok"
        assert attempts == 1
        assert calls["n"] == 1

    def test_retries_on_transient_then_succeeds(self) -> None:
        calls = {"n": 0}

        def _fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientAgentError("blip")
            return "ok"

        result, attempts = with_retries(_fn, max_attempts=5, initial_wait_seconds=0.0)
        assert result == "ok"
        assert attempts == 3

    def test_exhausts_attempts_and_reraises(self) -> None:
        def _fn() -> str:
            raise TransientAgentError("boom")

        with pytest.raises(TransientAgentError):
            with_retries(_fn, max_attempts=2, initial_wait_seconds=0.0)

    def test_does_not_retry_on_non_transient(self) -> None:
        calls = {"n": 0}

        def _fn() -> str:
            calls["n"] += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError):
            with_retries(_fn, max_attempts=5, initial_wait_seconds=0.0)
        assert calls["n"] == 1


class TestCircuitBreaker:
    def test_opens_after_threshold_consecutive_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, open_seconds=10.0)

        def _fail() -> None:
            raise TransientAgentError("boom")

        for _ in range(3):
            with pytest.raises(TransientAgentError):
                cb.call("k", _fail)

        # Fourth call should be short-circuited by the open breaker
        with pytest.raises(CircuitBreakerOpenError):
            cb.call("k", _fail)

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, open_seconds=10.0)

        def _fail() -> None:
            raise TransientAgentError("boom")

        for _ in range(2):
            with pytest.raises(TransientAgentError):
                cb.call("k", _fail)

        assert cb.call("k", lambda: "ok") == "ok"

        for _ in range(2):
            with pytest.raises(TransientAgentError):
                cb.call("k", _fail)
        assert cb.call("k", lambda: "ok") == "ok"

    def test_per_key_isolation(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, open_seconds=10.0)

        def _fail() -> None:
            raise TransientAgentError("boom")

        for _ in range(2):
            with pytest.raises(TransientAgentError):
                cb.call("a", _fail)
        with pytest.raises(CircuitBreakerOpenError):
            cb.call("a", _fail)

        assert cb.call("b", lambda: "ok") == "ok"

    def test_half_open_probe_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, open_seconds=0.05)

        def _fail() -> None:
            raise TransientAgentError("boom")

        for _ in range(2):
            with pytest.raises(TransientAgentError):
                cb.call("k", _fail)
        with pytest.raises(CircuitBreakerOpenError):
            cb.call("k", _fail)

        time.sleep(0.06)
        assert cb.call("k", lambda: "ok") == "ok"
        assert cb.call("k", lambda: "ok") == "ok"

    def test_non_transient_error_does_not_count_against_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, open_seconds=10.0)

        def _bug() -> None:
            raise ValueError("not transient")

        for _ in range(5):
            with pytest.raises(ValueError):
                cb.call("k", _bug)
        # Breaker stays closed; the failures are not transient.
        failures, is_open = cb.state_for("k")
        assert failures == 0
        assert is_open is False
