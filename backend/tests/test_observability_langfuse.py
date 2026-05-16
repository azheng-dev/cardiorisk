"""Tests for the Langfuse wrappers.

The key invariant: every helper is a no-op when ``LANGFUSE_*`` env
vars are unset. CI runs without keys; production wires them in.
Helpers that go through the SDK only fire when both keys are set.
"""

from __future__ import annotations

from typing import Any

import pytest

from cardiorisk.observability.langfuse import (
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
from cardiorisk.settings import get_settings


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every test to start with Langfuse disabled.

    Individual tests opt in by re-setting the env vars + clearing the
    settings cache. This keeps the suite hermetic — no test ever
    reaches the real Langfuse SDK by accident.
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    get_settings.cache_clear()


class TestNoOpMode:
    def test_get_client_returns_none_without_keys(self) -> None:
        assert get_langfuse_client() is None

    def test_record_generation_swallows_call(self) -> None:
        # Must not raise + must not return anything (None is fine).
        record_generation(
            model="gemini-2.5-flash",
            prompt="hello",
            completion="world",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            client_name="mock-llm",
        )

    def test_observe_node_is_identity_decorator(self) -> None:
        @observe_node(stage="triage")
        def inner(_state: Any) -> dict[str, str]:
            return {"k": "v"}

        out = inner({"case_id": "c1"})
        assert out == {"k": "v"}

    def test_observe_node_preserves_callable_metadata(self) -> None:
        @observe_node(stage="risk")
        def named(_state: Any) -> int:
            return 7

        assert named.__name__ == "named"

    def test_start_root_span_yields_none(self) -> None:
        with start_root_span(name="case", case_id="c1") as trace_id:
            assert trace_id is None

    def test_get_current_trace_id_returns_none(self) -> None:
        assert get_current_trace_id() is None

    def test_flush_is_a_no_op(self) -> None:
        flush_langfuse()  # must not raise


class TestSynthTraceId:
    def test_new_trace_id_has_mock_prefix(self) -> None:
        tid = new_trace_id()
        assert tid.startswith("mock-trace-")
        assert len(tid) > len("mock-trace-")

    def test_new_trace_id_is_unique(self) -> None:
        ids = {new_trace_id() for _ in range(64)}
        assert len(ids) == 64


class TestEnabledMode:
    """Smoke checks for the enabled path using monkeypatched SDK doubles.

    We don't reach real Langfuse — every test injects a fake client
    via monkeypatch so the SDK init contract is exercised without a
    network call.
    """

    def test_observe_node_wraps_when_client_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        class FakeSpan:
            def __enter__(self) -> FakeSpan:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def update(self, **kwargs: Any) -> None:
                calls.append({"update": kwargs})

        class FakeClient:
            def start_as_current_observation(self, **kwargs: Any) -> FakeSpan:
                calls.append({"open": kwargs})
                return FakeSpan()

            def get_current_trace_id(self) -> str:
                return "deadbeef"

            def flush(self) -> None:
                calls.append({"flush": True})

        monkeypatch.setattr(
            "cardiorisk.observability.langfuse.get_langfuse_client",
            lambda: FakeClient(),
        )

        @observe_node(stage="guideline")
        def inner(state: Any) -> dict[str, str]:
            return {"a": "1", "b": "2"}

        result = inner({"case_id": "c-42"})
        assert result == {"a": "1", "b": "2"}
        # one open + one update at the end
        assert any("open" in c for c in calls)
        assert any(
            "update" in c and c["update"].get("output", {}).get("keys") == ["a", "b"] for c in calls
        )

    def test_record_generation_caps_input_and_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class FakeGen:
            def update(self, **kwargs: Any) -> None:
                captured["update"] = kwargs

            def end(self) -> None:
                captured["ended"] = True

        class FakeClient:
            def start_observation(self, **kwargs: Any) -> FakeGen:
                captured["start"] = kwargs
                return FakeGen()

        monkeypatch.setattr(
            "cardiorisk.observability.langfuse.get_langfuse_client",
            lambda: FakeClient(),
        )

        big_prompt = "x" * (LANGFUSE_GENERATION_INPUT_LIMIT + 200)
        big_completion = "y" * (LANGFUSE_GENERATION_OUTPUT_LIMIT + 100)
        record_generation(
            model="gemini-2.5-flash",
            prompt=big_prompt,
            completion=big_completion,
            input_tokens=300,
            output_tokens=200,
            cost_usd=0.0042,
            client_name="gemini",
        )
        assert len(captured["start"]["input"]) == LANGFUSE_GENERATION_INPUT_LIMIT
        assert len(captured["update"]["output"]) == LANGFUSE_GENERATION_OUTPUT_LIMIT
        assert captured["update"]["usage_details"]["input"] == 300
        assert captured["update"]["usage_details"]["output"] == 200
        assert captured["update"]["cost_details"]["total"] == pytest.approx(0.0042)
        assert captured["ended"] is True

    def test_record_generation_sdk_errors_are_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BoomClient:
            def start_observation(self, **_kwargs: Any) -> Any:
                raise RuntimeError("network down")

        monkeypatch.setattr(
            "cardiorisk.observability.langfuse.get_langfuse_client",
            lambda: BoomClient(),
        )

        # Must not raise.
        record_generation(
            model="gemini-2.5-flash",
            prompt="p",
            completion="c",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            client_name="gemini",
        )

    def test_observe_node_sdk_errors_fall_back_to_inner_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BoomClient:
            def start_as_current_observation(self, **_kwargs: Any) -> Any:
                raise RuntimeError("OTel context broken")

        monkeypatch.setattr(
            "cardiorisk.observability.langfuse.get_langfuse_client",
            lambda: BoomClient(),
        )

        @observe_node(stage="letter")
        def inner(state: Any) -> dict[str, int]:
            return {"v": 1}

        # The decorated function must still return its value even when
        # the SDK blows up — observability never breaks the call.
        assert inner({"case_id": "c1"}) == {"v": 1}
