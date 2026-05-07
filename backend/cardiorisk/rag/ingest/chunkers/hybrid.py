"""Heading-aware hybrid chunker.

Pass 1: detect section boundaries by lines that look like headings.
Two heuristics, OR'd together:

- **Markdown-style:** lines starting with one or more ``#`` characters
  followed by whitespace. Always a heading. Used by the markdown
  fixture; also occasionally produced by pdfplumber for ATS-style
  PDFs.
- **PDF-style:** short (<= 80 chars), no terminal punctuation, and
  one of:
    * begins with a numbered prefix (``1.``, ``1.2``, ``1.2.3``,
      ``A.``, ``A.1``); OR
    * is mostly upper-case (>=70% of letters).

Pass 2: within each section, fall back to the token-window chunker
so no individual chunk exceeds the token target. The hybrid chunker
emits the section path on each chunk so the citation layer (Phase
3.3) can render hierarchical breadcrumbs.

The heuristic is *intentionally* simple. It will misfire on some
legitimate paragraphs and will miss some legitimate headings; ADR-015
documents the trade-off and the upgrade path (docling, marker, or a
small classifier on hand-labelled headings) if the Phase 3.2 eval
shows hybrid losing to either of its siblings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ..parse import ParsedDoc
from .base import Chunk, count_tokens, make_chunk
from .token_window import DEFAULT_STRIDE_TOKENS, DEFAULT_WINDOW_TOKENS, TokenWindowChunker

DEFAULT_MAX_HEADING_CHARS: Final[int] = 80
DEFAULT_UPPER_RATIO_THRESHOLD: Final[float] = 0.70

_MD_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\d+(?:\.\d+){0,4}\.?|[A-Z]\.(?:\d+\.?)?)\s+\S"
)
_TERMINAL_PUNCT: Final[frozenset[str]] = frozenset(".!?:;,")


@dataclass(frozen=True)
class HybridChunker:
    """Heading-aware chunker with token-window fallback within sections."""

    name: str = "hybrid"
    window_tokens: int = DEFAULT_WINDOW_TOKENS
    stride_tokens: int = DEFAULT_STRIDE_TOKENS
    max_heading_chars: int = DEFAULT_MAX_HEADING_CHARS
    upper_ratio_threshold: float = DEFAULT_UPPER_RATIO_THRESHOLD

    def __post_init__(self) -> None:
        if not 0 <= self.upper_ratio_threshold <= 1:
            raise ValueError(
                f"upper_ratio_threshold must be in [0,1], got {self.upper_ratio_threshold}"
            )

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        text = doc.full_text()
        if not text.strip():
            return []
        sections = _split_sections(
            text,
            max_heading_chars=self.max_heading_chars,
            upper_ratio_threshold=self.upper_ratio_threshold,
        )
        token_window = TokenWindowChunker(
            window_tokens=self.window_tokens, stride_tokens=self.stride_tokens
        )
        chunks: list[Chunk] = []
        for section_start, section_end, section_path in sections:
            section_text = text[section_start:section_end]
            if not section_text.strip():
                continue
            section_token_count = count_tokens(section_text)
            if section_token_count <= self.window_tokens:
                chunks.append(
                    make_chunk(
                        doc=doc,
                        strategy=self.name,
                        char_start=section_start,
                        char_end=section_end,
                        text=section_text,
                        section_path=section_path,
                    )
                )
                continue
            # Section is too long: fall back to the token-window chunker on
            # this slice. Build a synthetic single-page sub-doc so the
            # token chunker's offsets are local to the section, then
            # translate them back into doc-level offsets.
            from ..parse import _build_parsed_doc

            sub_doc = _build_parsed_doc(doc.doc_id, [section_text])
            for sub_chunk in token_window.chunk(sub_doc):
                abs_start = section_start + sub_chunk.char_start
                abs_end = section_start + sub_chunk.char_end
                chunks.append(
                    make_chunk(
                        doc=doc,
                        strategy=self.name,
                        char_start=abs_start,
                        char_end=abs_end,
                        text=text[abs_start:abs_end],
                        section_path=section_path,
                    )
                )
        return chunks


def _is_pdf_heading_line(
    stripped: str, *, max_heading_chars: int, upper_ratio_threshold: float
) -> bool:
    if not stripped or len(stripped) > max_heading_chars:
        return False
    if stripped[-1] in _TERMINAL_PUNCT:
        return False
    if _NUMBERED_HEADING_RE.match(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= upper_ratio_threshold


def _detect_heading(
    line: str, *, max_heading_chars: int, upper_ratio_threshold: float
) -> tuple[int, str] | None:
    """Return ``(level, title)`` if ``line`` is a heading, else ``None``.

    Markdown-style headings carry their explicit ``#`` count as level;
    PDF-style headings are level 1 by default (the heuristic doesn't
    try to infer hierarchy from text alone).
    """
    md = _MD_HEADING_RE.match(line)
    if md:
        return (len(md.group(1)), md.group(2).strip())
    stripped = line.strip()
    if _is_pdf_heading_line(
        stripped,
        max_heading_chars=max_heading_chars,
        upper_ratio_threshold=upper_ratio_threshold,
    ):
        return (1, stripped)
    return None


def _split_sections(
    text: str, *, max_heading_chars: int, upper_ratio_threshold: float
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Split ``text`` into ``(start, end, section_path)`` triples.

    The section path is the running stack of headings in scope at
    ``start``. Sections cover the text from one heading line up to
    (but not including) the next heading line; preamble text before
    the first heading is one section with empty section path.
    """
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    cursor = 0
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    section_start = 0
    for line in text.splitlines(keepends=True):
        line_no_eol = line.rstrip("\r\n")
        heading = _detect_heading(
            line_no_eol,
            max_heading_chars=max_heading_chars,
            upper_ratio_threshold=upper_ratio_threshold,
        )
        if heading is not None:
            # Close the previous section right before this heading.
            section_end = cursor
            if section_end > section_start:
                section_path = tuple(title for _, title in heading_stack)
                sections.append((section_start, section_end, section_path))
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            section_start = cursor
        cursor += len(line)
    if cursor > section_start:
        section_path = tuple(title for _, title in heading_stack)
        sections.append((section_start, cursor, section_path))
    return sections
