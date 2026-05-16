"""LLM client surface for the citation-mandatory generator.

Five implementations behind one Protocol (Phase 6, ADR-019):

- :class:`MockLLMClient` — deterministic, dep-free. Reads the prompt
  to find the available chunk_ids, then emits a sentence per top
  passage citing it. Used by every unit test and the CI smoke step
  so neither requires an API key. **Headline floor cell in Phase 6.**
- :class:`GeminiLLMClient` — Google ``gemini-2.5-flash`` (or any
  Gemini family member). Reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
  Free tier per ADR-024. **Headline production cell in Phase 6.**
- :class:`GroqLLMClient` — Groq-hosted Llama-3.3-70B (or any model
  Groq exposes). Reads ``GROQ_API_KEY``. Free tier per ADR-024.
  Off-by-default; opt-in third cell for the multi-model A/B.
- :class:`AnthropicLLMClient` — Claude Sonnet 4.5. Paid; kept in
  the codebase as an opt-in for users who already pay Anthropic,
  but **not** in the default config (free-tier constraint).
- :class:`OpenAILLMClient` — GPT-4o-mini. Same paid/opt-in story.

Phase 6 (eval harness) is the phase that actually picks the
production LLM. Phase 3.3 wired the surface; Phase 6 added Gemini +
Groq + the token / USD accounting layer.

Cost accounting
---------------
Every client tracks its own running token + USD totals via
:class:`UsageTotals`. The mock client reports zeros; live clients
read token usage from the SDK response and apply the price table
in :data:`PRICE_TABLE_USD_PER_1K`. The eval orchestrator reads
``client.usage`` after every call and rolls the deltas into the
per-cell cost line.
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

#: Price table for per-1K-token cost accounting. Source: vendor
#: pricing pages as of 2026-05-16. Free-tier models use the **paid
#: post-free-tier** rate so the eval cost line is honest if the
#: caller ever exceeds the free quota.
#:
#: Format: model_id -> (input_usd_per_1k, output_usd_per_1k).
PRICE_TABLE_USD_PER_1K: Final[dict[str, tuple[float, float]]] = {
    # Google Gemini (post-free-tier paid rate)
    "gemini-2.5-flash": (0.000075, 0.0003),
    "gemini-2.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.005),
    # Groq (post-free-tier paid rate; the free tier covers ~6K TPM/day
    # and the paid post-rate is published per-1M-tokens; converted)
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    # Anthropic (kept for opt-in users who already pay Anthropic)
    "claude-sonnet-4-5-20251022": (0.003, 0.015),
    # OpenAI (kept for opt-in users who already pay OpenAI)
    "gpt-4o-mini": (0.00015, 0.0006),
    # Mock — exact zero
    "mock-llm": (0.0, 0.0),
}


@dataclass
class UsageTotals:
    """Running totals for one client instance."""

    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, *, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.n_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd

    def snapshot(self) -> dict[str, float | int]:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up ``model`` in :data:`PRICE_TABLE_USD_PER_1K` and return the
    USD cost of one call with the given token counts. Unknown models
    cost 0.0 (and are flagged by the orchestrator's diagnostics)."""
    rate_in, rate_out = PRICE_TABLE_USD_PER_1K.get(model, (0.0, 0.0))
    return (input_tokens / 1000.0) * rate_in + (output_tokens / 1000.0) * rate_out


@dataclass(frozen=True)
class LLMMessage:
    """One message in an LLM chat-style call."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class BaseLLMClient(Protocol):
    """Shape every LLM wrapper implements.

    Implementations must be deterministic at ``temperature=0.0`` so
    the eval harness can assert byte-stable outputs (for the mock)
    and reproducible-enough outputs (for live LLMs at temp 0).

    Every implementation also exposes :attr:`usage` — a mutable
    :class:`UsageTotals` the call sites can read after every
    ``generate()`` to track tokens + USD.
    """

    name: str
    usage: UsageTotals

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
        self.usage = UsageTotals()

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
            text = "I do not have the supporting guidance for that question. [REFUSE]"
        else:
            # The mock interprets passage_text->sentence by picking the
            # first sentence of the passage. This is enough to give the
            # NLI verifier a real entailment signal in tests.
            sentences: list[str] = []
            for match in passages[: self._max_passages]:
                chunk_id = match.group("chunk_id")
                passage_text = match.group("text").strip()
                first = _first_sentence(passage_text)[: self._sentence_max_chars].rstrip(".") + "."
                sentences.append(f"{first} [{chunk_id}]")
            text = " ".join(sentences)

        # Even the mock tracks tokens (whitespace-split estimate) so
        # the orchestrator's cost line shows the same shape across
        # mock and live cells. Cost is exactly zero by definition.
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = len(text.split())
        self.usage.add(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=0.0)
        return text


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text


def _approx_tokens(text: str) -> int:
    """Fallback token estimator (~4 chars / token) for SDKs that don't
    report usage. Used only when the live response lacks token counts.
    """
    return max(1, len(text) // 4)


class GeminiLLMClient:
    """Google Gemini wrapper. Reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).

    Default model: ``gemini-2.5-flash`` — the free-tier production
    headline cell in Phase 6 (ADR-019, ADR-024). The free tier was
    ~10 RPM / 250 K TPM / 250 RPD as of 2026-05-16; the eval harness
    is well under that budget for the 100-case set.

    Uses the ``google-genai`` (1.x) SDK. Reports input/output token
    counts from ``response.usage_metadata`` so the cost line is real.
    """

    name: str = "gemini"

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} (or GOOGLE_API_KEY) not set; "
                "GeminiLLMClient requires a Google AI Studio key"
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._genai = genai
        self.usage = UsageTotals()

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        # Gemini chat: collapse system + user into one prompt. The
        # generator's prompt already wraps the system instructions
        # inside the single rendered template, so this is lossless.
        contents = "\n\n".join(m.content for m in messages)
        from google.genai import types as gtypes

        config = gtypes.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        resp: Any = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        text = str(getattr(resp, "text", "") or "")
        # Token usage from response.usage_metadata; fall back to estimator.
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
        else:
            input_tokens = _approx_tokens(contents)
            output_tokens = _approx_tokens(text)
        cost = estimate_cost_usd(self._model, input_tokens, output_tokens)
        self.usage.add(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        return text


class GroqLLMClient:
    """Groq-hosted OpenAI-compatible wrapper. Reads ``GROQ_API_KEY``.

    Default model: ``llama-3.3-70b-versatile`` — the opt-in second
    live cell for the Phase 6 multi-model A/B. Off by default in
    every CI step; flipped on locally by setting ``GROQ_API_KEY``.
    Uses the standard ``openai`` SDK pointed at Groq's
    ``https://api.groq.com/openai/v1`` base URL.
    """

    name: str = "groq"

    def __init__(
        self,
        *,
        model: str = "llama-3.3-70b-versatile",
        api_key_env: str = "GROQ_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} not set; "
                "GroqLLMClient requires a Groq API key"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self._model = model
        self.usage = UsageTotals()

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        chat_messages: Any = [{"role": m.role, "content": m.content} for m in messages]
        resp: Any = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_messages,
        )
        text = str(resp.choices[0].message.content or "")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            input_tokens = sum(_approx_tokens(m.content) for m in messages)
            output_tokens = _approx_tokens(text)
        cost = estimate_cost_usd(self._model, input_tokens, output_tokens)
        self.usage.add(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        return text


class AnthropicLLMClient:
    """Anthropic Claude wrapper. Reads ``ANTHROPIC_API_KEY`` from env.

    **Paid only.** Kept in the codebase for opt-in users who already
    pay Anthropic; the default Phase 6 config (ADR-024 free-tier)
    does NOT include this client.
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
        self.usage = UsageTotals()

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
        text = "".join(block.text for block in resp.content if getattr(block, "text", None))
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        else:
            input_tokens = sum(_approx_tokens(m.content) for m in messages)
            output_tokens = _approx_tokens(text)
        cost = estimate_cost_usd(self._model, input_tokens, output_tokens)
        self.usage.add(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        return text


class OpenAILLMClient:
    """OpenAI Chat Completions wrapper. Reads ``OPENAI_API_KEY`` from env.

    **Paid only.** Kept in the codebase for opt-in users who already
    pay OpenAI; not in the default Phase 6 config (ADR-024).
    """

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
        self.usage = UsageTotals()

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        chat_messages: Any = [{"role": m.role, "content": m.content} for m in messages]
        resp: Any = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_messages,
        )
        text = str(resp.choices[0].message.content or "")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            input_tokens = sum(_approx_tokens(m.content) for m in messages)
            output_tokens = _approx_tokens(text)
        cost = estimate_cost_usd(self._model, input_tokens, output_tokens)
        self.usage.add(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        return text


def get_llm_client(name: str, **kwargs: Any) -> BaseLLMClient:
    """Factory mirroring :func:`embed.get_embedder`.

    Names accepted (Phase 6, ADR-019):

    - ``mock`` — :class:`MockLLMClient` (no creds; the deterministic
      floor cell in CI and the regression baseline).
    - ``gemini`` / ``google`` — :class:`GeminiLLMClient`
      (``gemini-2.5-flash`` default; free tier).
    - ``groq`` — :class:`GroqLLMClient` (``llama-3.3-70b-versatile``
      default; free tier; opt-in third cell).
    - ``anthropic`` / ``claude`` — :class:`AnthropicLLMClient` (paid).
    - ``openai`` / ``gpt`` — :class:`OpenAILLMClient` (paid).
    """
    if name == "mock":
        return MockLLMClient(**kwargs)
    if name in ("gemini", "google"):
        return GeminiLLMClient(**kwargs)
    if name == "groq":
        return GroqLLMClient(**kwargs)
    if name in ("anthropic", "claude"):
        return AnthropicLLMClient(**kwargs)
    if name in ("openai", "gpt", "gpt-4o-mini"):
        return OpenAILLMClient(**kwargs)
    raise ValueError(f"unknown llm client {name!r}; known: mock, gemini, groq, anthropic, openai")


def deterministic_seed(prompt_text: str, salt: str = "") -> int:
    """Deterministic seed derived from prompt + optional salt.

    Tests use this to assert byte-stable outputs from the mock client
    when the prompt changes byte-identically across runs.
    """
    h = hashlib.sha256((prompt_text + salt).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")
