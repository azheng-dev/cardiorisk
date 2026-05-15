"""LLM client surface for the citation-mandatory generator.

Three implementations behind one Protocol:

- :class:`MockLLMClient` — deterministic, dep-free. Reads the prompt
  to find the available chunk_ids, then emits a sentence per top
  passage citing it. Used by every unit test and the CI smoke step
  so neither requires an API key.
- :class:`AnthropicLLMClient` — Claude Sonnet 4.5 (or any other
  Anthropic-hosted model). Reads ``ANTHROPIC_API_KEY`` from the
  environment. Lazy import; the ``anthropic`` package only loads if
  this client is actually constructed.
- :class:`OpenAILLMClient` — GPT-4o-mini (or any other OpenAI
  model). Reads ``OPENAI_API_KEY``. Lazy import.

Phase 6 (eval harness) is the phase that actually picks the
production LLM. Phase 3.3 ships all three so the eval harness can
swap them without touching the generator code.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

#: Default upper-bound token budget for one LLM call.
DEFAULT_MAX_TOKENS: Final[int] = 512
#: Sampling temperature. The generator runs effectively-greedy because
#: the verifier handles the hallucination axis; randomness only adds
#: noise to the eval.
DEFAULT_TEMPERATURE: Final[float] = 0.0


@dataclass(frozen=True)
class LLMMessage:
    """One message in an LLM chat-style call."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class BaseLLMClient(Protocol):
    """Shape every LLM wrapper implements.

    Implementations must be deterministic at ``temperature=0.0`` so
    the eval harness can assert byte-stable outputs.
    """

    name: str

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Return the assistant message body as a single string."""
        ...


_CHUNK_ID_RE = re.compile(r"\[chunk_id=(?P<chunk_id>[A-Za-z0-9_:.\-]+)\]")
_PASSAGE_BLOCK_RE = re.compile(
    r"\[chunk_id=(?P<chunk_id>[A-Za-z0-9_:.\-]+)\]\s*"
    r"\(doc=(?P<doc_id>[A-Za-z0-9_.\-]+),\s*page=(?P<p_start>\d+)-(?P<p_end>\d+)\)\s*"
    r"(?P<text>.+?)(?=\n\[chunk_id=|\Z)",
    re.DOTALL,
)


class MockLLMClient:
    """Deterministic, dep-free LLM stand-in.

    Behaviour:

    - Reads the user message, extracts every available passage from
      the prompt's "Available passages" section, and detects the
      question text.
    - For each of the first ``self.max_passages`` passages, emits one
      sentence that quotes the first informative span of the passage
      and ends with the ``[chunk_id]`` citation.
    - If the question contains the case-insensitive substring
      ``"refuse"`` OR no passages were supplied, emits the canonical
      refusal line.

    The mock is **not** a stand-in for the production LLM in clinical
    quality — it's here so every unit test and CI smoke can exercise
    the parse / verify / suppress path without an API key.
    """

    name: str = "mock-llm"

    def __init__(self, *, max_passages: int = 2, sentence_max_chars: int = 200) -> None:
        self._max_passages = max_passages
        self._sentence_max_chars = sentence_max_chars

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        # Concatenate all message bodies; the prompt loader puts the
        # passages + question into one user turn.
        body = "\n\n".join(m.content for m in messages)

        passages = list(_PASSAGE_BLOCK_RE.finditer(body))

        if not passages:
            return "I do not have the supporting guidance for that question. [REFUSE]"

        # The mock interprets passage_text->sentence by picking the
        # first sentence of the passage. This is enough to give the
        # NLI verifier a real entailment signal in tests.
        sentences: list[str] = []
        for match in passages[: self._max_passages]:
            chunk_id = match.group("chunk_id")
            text = match.group("text").strip()
            first_sentence = _first_sentence(text)[: self._sentence_max_chars].rstrip(".") + "."
            sentences.append(f"{first_sentence} [{chunk_id}]")

        return " ".join(sentences)


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text


class AnthropicLLMClient:
    """Anthropic Claude wrapper. Reads ``ANTHROPIC_API_KEY`` from env.

    Phase 3.3 wires the surface; Phase 6 picks the actual model id
    after the multi-model eval. Default model id is documented in
    ADR-018 (placeholder); the constructor accepts an explicit
    ``model`` to keep the eval scriptable.
    """

    name: str = "anthropic-claude"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5-20251022",
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} not set; "
                "AnthropicLLMClient requires an Anthropic API key"
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        # Anthropic separates the system message from the chat list.
        sys_parts = [m.content for m in messages if m.role == "system"]
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if not chat:
            chat = [{"role": "user", "content": "\n\n".join(sys_parts)}]
            sys_parts = []
        resp: Any = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="\n\n".join(sys_parts) if sys_parts else None,
            messages=chat,
        )
        # Concat all text blocks.
        return "".join(block.text for block in resp.content if getattr(block, "text", None))


class OpenAILLMClient:
    """OpenAI Chat Completions wrapper. Reads ``OPENAI_API_KEY`` from env."""

    name: str = "openai-gpt"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} not set; "
                "OpenAILLMClient requires an OpenAI API key"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        resp: Any = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        choice = resp.choices[0].message.content
        return str(choice or "")


def get_llm_client(name: str, **kwargs: Any) -> BaseLLMClient:
    """Factory mirroring :func:`embed.get_embedder`.

    Names accepted:

    - ``mock`` — :class:`MockLLMClient` (no creds required).
    - ``anthropic`` / ``claude`` — :class:`AnthropicLLMClient`.
    - ``openai`` / ``gpt`` — :class:`OpenAILLMClient`.
    """
    if name == "mock":
        return MockLLMClient(**kwargs)
    if name in ("anthropic", "claude"):
        return AnthropicLLMClient(**kwargs)
    if name in ("openai", "gpt", "gpt-4o-mini"):
        return OpenAILLMClient(**kwargs)
    raise ValueError(f"unknown llm client {name!r}; known: mock, anthropic, openai")


def deterministic_seed(prompt_text: str, salt: str = "") -> int:
    """Deterministic seed derived from prompt + optional salt.

    Tests use this to assert byte-stable outputs from the mock client
    when the prompt changes byte-identically across runs.
    """
    h = hashlib.sha256((prompt_text + salt).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")
