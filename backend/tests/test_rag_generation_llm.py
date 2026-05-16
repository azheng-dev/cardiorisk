"""Tests for LLM client wrappers."""

from __future__ import annotations

import os

import pytest

from cardiorisk.rag.generation.llm import (
    PRICE_TABLE_USD_PER_1K,
    AnthropicLLMClient,
    BaseLLMClient,
    GeminiLLMClient,
    GroqLLMClient,
    LLMMessage,
    MockLLMClient,
    OpenAILLMClient,
    UsageTotals,
    deterministic_seed,
    estimate_cost_usd,
    get_llm_client,
)
from cardiorisk.rag.generation.prompts import PromptPassage, render_citation_prompt


def _passage(chunk_id: str, text: str) -> PromptPassage:
    return PromptPassage(
        chunk_id=chunk_id,
        doc_id="doc",
        page_start=1,
        page_end=1,
        text=text,
    )


def test_mock_client_satisfies_protocol() -> None:
    client: BaseLLMClient = MockLLMClient()
    assert isinstance(client, BaseLLMClient)
    assert client.name == "mock-llm"


def test_mock_client_emits_one_sentence_per_passage() -> None:
    prompt = render_citation_prompt(
        question="What is the entry age for risk assessment?",
        passages=[
            _passage(
                "fixture_racgp_cvd:p1:c1",
                "Adults aged 45 to 79 should have CVD risk calculated every two years.",
            ),
            _passage(
                "fixture_nvdpa_quickref:p1:c2",
                "For people with diabetes, begin assessment at age 35.",
            ),
        ],
    )
    out = MockLLMClient(max_passages=2).generate([LLMMessage("user", prompt)])
    assert "[fixture_racgp_cvd:p1:c1]" in out
    assert "[fixture_nvdpa_quickref:p1:c2]" in out


def test_mock_client_refuses_when_no_passages() -> None:
    out = MockLLMClient().generate(
        [LLMMessage("user", "# Question\nfoo\n# Available passages\n\n")]
    )
    assert "I do not have the supporting guidance" in out
    assert "[REFUSE]" in out


def test_mock_client_is_deterministic_at_temp_zero() -> None:
    prompt = render_citation_prompt(
        question="Q",
        passages=[_passage("a:1", "Alpha sentence one. Alpha sentence two.")],
    )
    a = MockLLMClient().generate([LLMMessage("user", prompt)])
    b = MockLLMClient().generate([LLMMessage("user", prompt)])
    assert a == b


def test_get_llm_client_dispatches_mock() -> None:
    client = get_llm_client("mock")
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown llm client"):
        get_llm_client("transmogrifier")


def test_anthropic_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMClient()


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLMClient()


def test_deterministic_seed_stable() -> None:
    assert deterministic_seed("hello", salt="x") == deterministic_seed("hello", salt="x")
    assert deterministic_seed("a") != deterministic_seed("b")


def test_anthropic_client_does_not_consume_real_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sanity: the constructor must not import anthropic when the key is missing.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    with pytest.raises(RuntimeError):
        AnthropicLLMClient()


# ----------------------------------------------------------------- Phase 6: cost accounting + new clients
class TestUsageAndCost:
    def test_mock_client_tracks_usage_per_call(self) -> None:
        client = MockLLMClient()
        assert client.usage.n_calls == 0
        client.generate([LLMMessage("user", "hello world")])
        assert client.usage.n_calls == 1
        assert client.usage.input_tokens > 0
        assert client.usage.cost_usd == 0.0

    def test_usage_totals_accumulates(self) -> None:
        u = UsageTotals()
        u.add(input_tokens=10, output_tokens=5, cost_usd=0.001)
        u.add(input_tokens=20, output_tokens=8, cost_usd=0.002)
        snap = u.snapshot()
        assert snap["n_calls"] == 2
        assert snap["input_tokens"] == 30
        assert snap["output_tokens"] == 13
        assert snap["cost_usd"] == pytest.approx(0.003)

    def test_estimate_cost_usd_known_model(self) -> None:
        # Gemini 2.5 Flash: 0.000075/1K in, 0.0003/1K out
        cost = estimate_cost_usd("gemini-2.5-flash", input_tokens=1000, output_tokens=1000)
        assert cost == pytest.approx(0.000075 + 0.0003)

    def test_estimate_cost_usd_unknown_model_is_zero(self) -> None:
        assert estimate_cost_usd("not-a-real-model", 1000, 1000) == 0.0

    def test_price_table_lists_every_default_model(self) -> None:
        # Models that the factory will hand back when called with the
        # default arguments must all have a row in the price table.
        for model in (
            "mock-llm",
            "gemini-2.5-flash",
            "llama-3.3-70b-versatile",
            "claude-sonnet-4-5-20251022",
            "gpt-4o-mini",
        ):
            assert model in PRICE_TABLE_USD_PER_1K, model


class TestGeminiClient:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiLLMClient()

    def test_factory_dispatches_to_gemini_when_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-only-construct")
        client = get_llm_client("gemini")
        assert isinstance(client, GeminiLLMClient)

    def test_falls_back_to_google_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-only-construct")
        client = GeminiLLMClient()
        assert client.usage.n_calls == 0


class TestGroqClient:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            GroqLLMClient()

    def test_factory_dispatches_to_groq_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-only-construct")
        client = get_llm_client("groq")
        assert isinstance(client, GroqLLMClient)
