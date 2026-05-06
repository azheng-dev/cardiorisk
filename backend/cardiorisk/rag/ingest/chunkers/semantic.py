"""Sentence-aware chunker: regex sentence splitter + token target.

No spaCy dependency; the regex catches the cases that matter for
RACGP / NVDPA prose (period / question mark / exclamation followed
by whitespace and an upper-case start). Phase 3.2's eval will tell
us whether the splitter's misses on edge cases (medical
abbreviations like "e.g.", "i.e.", "vs.") are material; if they are,
the upgrade path is to swap this regex for spaCy's :func:`sents`
without touching the orchestrator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ..parse import ParsedDoc
from .base import Chunk, count_tokens, make_chunk

DEFAULT_TARGET_TOKENS: Final[int] = 512
DEFAULT_OVERLAP_SENTENCES: Final[int] = 1

# Sentence boundary: terminal punctuation + whitespace + uppercase
# starting letter. Lookbehind and lookahead so the split discards no
# characters. Standard English-prose heuristic; deliberately simple.
_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass(frozen=True)
class SemanticChunker:
    """Sentence-aware chunker.

    Splits the document text into sentences, then greedily groups
    sentences into chunks targeting :attr:`target_tokens` cl100k_base
    tokens. Always cuts on sentence boundaries; the last chunk may
    be smaller. ``overlap_sentences`` sentences from the end of one
    chunk are repeated at the start of the next.
    """

    name: str = "semantic"
    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError(f"target_tokens must be > 0, got {self.target_tokens}")
        if self.overlap_sentences < 0:
            raise ValueError(f"overlap_sentences must be >= 0, got {self.overlap_sentences}")

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        text = doc.full_text()
        if not text.strip():
            return []
        sentences = _split_sentences_with_offsets(text)
        if not sentences:
            return []
        return _group_sentences(
            doc=doc,
            sentences=sentences,
            strategy=self.name,
            target_tokens=self.target_tokens,
            overlap_sentences=self.overlap_sentences,
        )


def _split_sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Return ``(char_start, char_end, sentence_text)`` for each sentence.

    The first sentence starts at offset 0; subsequent sentences start
    immediately after the whitespace consumed by the boundary regex.
    Sentence text is stripped of leading whitespace but preserves
    internal whitespace.
    """
    matches = list(_SENTENCE_BOUNDARY_RE.finditer(text))
    if not matches:
        return [(0, len(text), text.strip())] if text.strip() else []
    sentences: list[tuple[int, int, str]] = []
    cursor = 0
    for m in matches:
        end = m.start()
        sentence_text = text[cursor:end].strip()
        if sentence_text:
            sentences.append((cursor, end, sentence_text))
        cursor = m.end()
    tail = text[cursor:].strip()
    if tail:
        sentences.append((cursor, len(text), tail))
    return sentences


def _group_sentences(
    *,
    doc: ParsedDoc,
    sentences: list[tuple[int, int, str]],
    strategy: str,
    target_tokens: int,
    overlap_sentences: int,
) -> list[Chunk]:
    """Greedy sentence grouping with token-budget + sentence-overlap."""
    chunks: list[Chunk] = []
    i = 0
    n = len(sentences)
    while i < n:
        group_start_idx = i
        running_tokens = 0
        end_idx = i
        while end_idx < n:
            _, _, s_text = sentences[end_idx]
            s_tokens = count_tokens(s_text)
            if running_tokens > 0 and running_tokens + s_tokens > target_tokens:
                break
            running_tokens += s_tokens
            end_idx += 1
        # Always advance at least one sentence even if it busts the budget,
        # otherwise we'd loop forever on pathologically long sentences.
        if end_idx == group_start_idx:
            end_idx = group_start_idx + 1
        char_start = sentences[group_start_idx][0]
        char_end = sentences[end_idx - 1][1]
        chunk_text = doc.full_text()[char_start:char_end]
        chunks.append(
            make_chunk(
                doc=doc,
                strategy=strategy,
                char_start=char_start,
                char_end=char_end,
                text=chunk_text,
            )
        )
        if end_idx >= n:
            break
        # Step forward past the consumed sentences, keeping
        # overlap_sentences for the next chunk's start.
        i = max(end_idx - overlap_sentences, group_start_idx + 1)
    return chunks
