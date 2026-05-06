"""Chunker registry.

Three pluggable strategies ship in Phase 3.1; Phase 3.2's 50-Q
retrieval eval picks the winner.

- ``"token"`` — :class:`~.token_window.TokenWindowChunker`
  (tiktoken cl100k_base, 512 tokens with 64-token stride). Robust
  baseline.
- ``"semantic"`` — :class:`~.semantic.SemanticChunker` (regex
  sentence splitter; groups whole sentences up to a token target).
  No LLM dependency, no spaCy dependency.
- ``"hybrid"`` — :class:`~.hybrid.HybridChunker` (heading-aware
  section detection, then token-window fallback within each section).

The CLI ``--strategy {all,token,semantic,hybrid}`` flag indexes
into :data:`NAME_TO_CHUNKER`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from .base import Chunk, Chunker, chunk_id_for, count_tokens, load_chunks, save_chunks
from .hybrid import HybridChunker
from .semantic import SemanticChunker
from .token_window import TokenWindowChunker

# Registry value is a zero-arg factory. The ``# type: ignore`` lines
# document a known mypy interaction with frozen-dataclass chunkers and
# a Protocol that declares an instance-level ``name: str``: at runtime
# every concrete chunker satisfies Chunker structurally (the test
# ``test_registered_factories_construct_expected_classes`` enforces
# this). Mypy refuses to verify it because it cannot prove that a
# dataclass with a defaulted ``name: str = "token"`` field exposes the
# attribute as a Protocol-required instance attribute. The runtime
# behaviour is correct.
NAME_TO_CHUNKER: Final[dict[str, Callable[[], Chunker]]] = {
    "token": TokenWindowChunker,  # type: ignore[dict-item]
    "semantic": SemanticChunker,  # type: ignore[dict-item]
    "hybrid": HybridChunker,  # type: ignore[dict-item]
}

__all__ = [
    "NAME_TO_CHUNKER",
    "Chunk",
    "Chunker",
    "HybridChunker",
    "SemanticChunker",
    "TokenWindowChunker",
    "chunk_id_for",
    "count_tokens",
    "load_chunks",
    "save_chunks",
]
