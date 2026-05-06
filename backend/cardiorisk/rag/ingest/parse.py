"""PDF (and fixture markdown) parser to a uniform :class:`ParsedDoc`.

Two entry points:

- :func:`parse_pdf` — wraps :mod:`pdfplumber`. One :class:`ParsedPage`
  per PDF page, carrying ``page_no`` (1-indexed), ``text`` (the
  page's ``extract_text()`` output, ``""`` for empty pages), and
  ``char_offset`` (this page's start position in the whole-doc
  concatenation produced by :meth:`ParsedDoc.full_text`).
- :func:`parse_markdown_fixture` — used by ``--use-fixture`` mode.
  Splits a markdown file on the line-exact marker ``<!-- page break
  -->`` to simulate pagination. PDF-free, network-free, 0 deps.
  Emits the same :class:`ParsedDoc` schema so downstream chunkers
  cannot tell the difference.

The dispatch helper :func:`parse_doc_for_source` picks the right
backend by file extension (``.pdf`` -> pdfplumber, ``.md`` ->
markdown fixture). This lets ``build_corpus.py --use-fixture`` and
the production path share one driver.

Persistence: :func:`save_parsed_doc` writes a one-page-per-line JSONL
under ``data/external/corpus/parsed/<doc_id>.jsonl`` (gitignored).
:func:`load_parsed_doc` is the inverse. The manifest records each
parsed file's path + sha256 so the chunker stage can find them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pdfplumber

PAGE_BREAK_MARKER: Final[str] = "<!-- page break -->"


@dataclass(frozen=True)
class ParsedPage:
    """One page of a parsed document.

    Attributes:
        page_no: 1-indexed page number. Stable across re-parses; used
            by the eval-set ``expected_page_range`` check.
        text: The page's text content. ``""`` for an empty page (PDFs
            sometimes have figure-only pages where ``extract_text``
            returns ``None``); the parser coerces that to the empty
            string so chunkers don't have to defend against ``None``.
        char_offset: This page's start position in the whole-doc
            concatenation produced by :meth:`ParsedDoc.full_text`.
            Pages are joined by ``"\\n\\n"`` (two newlines), so
            ``char_offset`` of page ``k`` is
            ``sum(len(p.text) + 2 for p in pages[:k-1])``. Lets a
            chunk's ``(char_start, char_end)`` span be mapped back to
            the originating page in O(log n).
    """

    page_no: int
    text: str
    char_offset: int


@dataclass(frozen=True)
class ParsedDoc:
    """A parsed document: ``doc_id`` + an ordered list of pages.

    Pages are always sorted by ``page_no`` ascending; the parser
    enforces this invariant when constructing.
    """

    doc_id: str
    pages: tuple[ParsedPage, ...]

    def full_text(self) -> str:
        """Return the whole-document text, pages joined by ``\\n\\n``."""
        return "\n\n".join(p.text for p in self.pages)

    def page_for_offset(self, offset: int) -> int:
        """Return the 1-indexed page number containing ``offset``.

        Used by chunkers to back-map a chunk span to its source page.
        Falls back to the last page if ``offset`` is past the end
        (which happens when a chunker produces a span that includes
        the trailing ``\\n\\n`` join between pages).
        """
        if offset < 0:
            raise ValueError(f"offset {offset} must be non-negative")
        last_page_no = self.pages[-1].page_no if self.pages else 1
        for prev, nxt in zip(self.pages, self.pages[1:], strict=False):
            if prev.char_offset <= offset < nxt.char_offset:
                return prev.page_no
        return last_page_no


class ParseError(RuntimeError):
    """Raised when a document cannot be parsed."""


def _build_parsed_doc(doc_id: str, page_texts: list[str]) -> ParsedDoc:
    """Construct a :class:`ParsedDoc` from raw per-page text strings.

    Computes monotonically-increasing ``char_offset`` values that
    match the join semantics of :meth:`ParsedDoc.full_text`.
    """
    pages: list[ParsedPage] = []
    offset = 0
    for idx, text in enumerate(page_texts, start=1):
        pages.append(ParsedPage(page_no=idx, text=text, char_offset=offset))
        offset += len(text) + len("\n\n")
    return ParsedDoc(doc_id=doc_id, pages=tuple(pages))


def parse_pdf(path: Path, *, doc_id: str) -> ParsedDoc:
    """Parse a PDF using pdfplumber. One :class:`ParsedPage` per page.

    pdfplumber is MIT-licensed (per ADR-015's parser-licensing note).
    Empty / figure-only pages where ``extract_text`` returns ``None``
    are coerced to ``""`` so downstream chunkers see a uniform schema.

    Raises :class:`ParseError` if the file cannot be opened.
    """
    if not path.exists():
        raise ParseError(f"PDF not found: {path}")
    try:
        with pdfplumber.open(path) as pdf:
            page_texts = [(page.extract_text() or "") for page in pdf.pages]
    except Exception as exc:
        raise ParseError(f"pdfplumber failed on {path}: {exc}") from exc
    if not page_texts:
        raise ParseError(f"pdfplumber returned 0 pages for {path}")
    return _build_parsed_doc(doc_id, page_texts)


def parse_markdown_fixture(path: Path, *, doc_id: str) -> ParsedDoc:
    """Parse a markdown fixture file into a :class:`ParsedDoc`.

    Pages are separated by lines that match the marker
    ``<!-- page break -->`` exactly (no surrounding whitespace, no
    extra characters). A document with no marker is one page. The
    marker line is consumed (it does not appear in any page's text).
    """
    if not path.exists():
        raise ParseError(f"markdown fixture not found: {path}")
    raw = path.read_text(encoding="utf-8")
    pages_raw: list[list[str]] = [[]]
    for line in raw.splitlines():
        if line.strip() == PAGE_BREAK_MARKER:
            pages_raw.append([])
            continue
        pages_raw[-1].append(line)
    page_texts = ["\n".join(lines).strip("\n") for lines in pages_raw]
    return _build_parsed_doc(doc_id, page_texts)


def parse_doc_for_source(path: Path, *, doc_id: str) -> ParsedDoc:
    """Dispatch to the right parser based on file extension.

    ``.pdf`` -> :func:`parse_pdf`; ``.md`` -> :func:`parse_markdown_fixture`.
    Raises :class:`ParseError` on any other extension.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, doc_id=doc_id)
    if suffix == ".md":
        return parse_markdown_fixture(path, doc_id=doc_id)
    raise ParseError(f"unsupported extension {suffix!r} for {path}; expected .pdf or .md")


def save_parsed_doc(doc: ParsedDoc, out_path: Path) -> str:
    """Write a parsed doc to a one-page-per-line JSONL and return sha256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(asdict(p), sort_keys=True) for p in doc.pages)
    payload_bytes = (payload + ("\n" if payload else "")).encode("utf-8")
    out_path.write_bytes(payload_bytes)
    return hashlib.sha256(payload_bytes).hexdigest()


def load_parsed_doc(in_path: Path, *, doc_id: str) -> ParsedDoc:
    """Inverse of :func:`save_parsed_doc`."""
    if not in_path.exists():
        raise ParseError(f"parsed JSONL not found: {in_path}")
    pages: list[ParsedPage] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pages.append(
            ParsedPage(
                page_no=int(record["page_no"]),
                text=str(record["text"]),
                char_offset=int(record["char_offset"]),
            )
        )
    pages.sort(key=lambda p: p.page_no)
    return ParsedDoc(doc_id=doc_id, pages=tuple(pages))
