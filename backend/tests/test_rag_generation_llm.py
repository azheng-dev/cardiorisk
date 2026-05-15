"""Tests for LLM client wrappers."""

from __future__ import annotations

import os

import pytest

from cardiorisk.rag.generation.llm import (
    AnthropicLLMClient,
    BaseLLMClient,
    LLMMessage,
    MockLLMClient,
    OpenAILLMClient,
    deterministic_seed,
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
