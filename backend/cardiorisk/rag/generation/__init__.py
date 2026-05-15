"""Phase 3.3 citation-mandatory generation layer.

Generator wiring (top-down):

- :mod:`.generator` — :class:`CitationGenerator.generate(query)`
  orchestrates the full retrieve-prompt-parse-verify-suppress loop.
  This is the only public entry point Phase 4's LangGraph nodes call.
- :mod:`.llm` — :class:`BaseLLMClient` Protocol +
  :class:`MockLLMClient` (deterministic, used by CI + unit tests) +
  :class:`AnthropicLLMClient` / :class:`OpenAILLMClient` (gated on
  ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``; never imported in CI).
- :mod:`.prompts` — file-backed prompt loader. Templates live next
  to this package as plain ``.md`` files so they are reviewable,
  diffable, and version-pinned by filename suffix.
- :mod:`.parser` — :func:`parse_answer` lifts the LLM's free-text
  output into :class:`Claim` rows with explicit citation lists.
  Tolerates the most common citation-format failure modes (missing
  brackets, multiple cites per sentence, citations on a separate
  line) without trying to "fix" the underlying claim.
- :mod:`.nli` — :class:`BaseNLIVerifier` Protocol +
  :class:`MockNLIVerifier` (token-overlap fallback for CI) +
  :class:`DeBERTaNLIVerifier` (Hugging Face transformers wrapper
  around ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``;
  loaded lazily so the unit-test import path doesn't pay the dep
  cost).

Suppression policy (ADR-017): every claim is verified against every
chunk it cites. The claim survives iff at least one of its citations
yields ``P(entailment) >= entail_threshold`` (default 0.5). Otherwise
the claim is dropped — never re-prompted, never silently re-written.
The dropped-claim list is kept on the :class:`GeneratedAnswer` so the
UI can surface "the model attempted this claim but the verifier could
not support it"; the trade-off discussion lives in ADR-017 §3.
"""
