"""Chunker Protocol + :class:`Chunk` dataclass + shared helpers.

Every chunker emits the same :class:`Chunk` schema, so the manifest
records a uniform ``chunks/<strategy>.jsonl`` per strategy and Phase
3.2's retrieval layer can swap strategies without changing the
ingest contract.

Determinism: :func:`chunk_id_for` derives the chunk id from
``(doc_id, strategy, char_start, char_end)`` via SHA-256 → 16 hex
chars. Same input → same id; the test suite asserts this. This also
makes chunk ids stable across re-parses, which lets Phase 3.2's
retrieval eval pin "expected chunk id" in the eval set without
re-running ingestion every time the harness changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol

import tiktoken

from ..parse import ParsedDoc

TIKTOKEN_ENCODING_NAME: Final[str] = "cl100k_base"

# Module-level cached encoding. tiktoken loads the BPE merges from
# disk on the first call; subsequent .get_encoding() calls re-use
# the same in-process object, but caching one local reference makes
# the per-call cost transparent.
_ENC: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)
    return _ENC


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for ``text``."""
    return len(_enc().encode(text))


@dataclass(frozen=True)
class Chunk:
    """One retrievable chunk produced by a :class:`Chunker`.

    Attributes:
        chunk_id: Deterministic 16-hex-char SHA-256 prefix of
            ``f"{doc_id}|{strategy}|{char_start}|{char_end}"``.
        doc_id: Source document id (matches
            :attr:`~cardiorisk.rag.ingest.parse.ParsedDoc.doc_id`).
        strategy: ``"token"``, ``"semantic"``, or ``"hybrid"``.
        char_start: Inclusive start position in
            :meth:`ParsedDoc.full_text`.
        char_end: Exclusive end position in
            :meth:`ParsedDoc.full_text`. ``char_end > char_start``.
        page_start: 1-indexed page number of the first character.
        page_end: 1-indexed page number of the last character.
        text: The chunk's text. Equal to ``full_text[char_start:char_end]``.
        n_tokens: cl100k_base token count of ``text``.
        section_path: Optional path of containing headings (e.g.
            ``["Cardiovascular disease prevention", "Risk assessment"]``).
            Empty for the token-window chunker; populated by the
            heading-aware hybrid chunker; populated by the semantic
            chunker only when a sentence happens to land inside one.
    """

    chunk_id: str
    doc_id: str
    strategy: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    text: str
    n_tokens: int
    section_path: tuple[str, ...] = ()


def chunk_id_for(*, doc_id: str, strategy: str, char_start: int, char_end: int) -> str:
    """Return the deterministic 16-hex-char chunk id."""
    digest = hashlib.sha256(f"{doc_id}|{strategy}|{char_start}|{char_end}".encode()).hexdigest()
    return digest[:16]


def make_chunk(
    *,
    doc: ParsedDoc,
    strategy: str,
    char_start: int,
    char_end: int,
    text: str,
    section_path: tuple[str, ...] = (),
) -> Chunk:
    """Build a :class:`Chunk` and resolve its page range from ``doc``."""
    if char_end <= char_start:
        raise ValueError(f"empty chunk span: char_start={char_start}, char_end={char_end}")
    page_start = doc.page_for_offset(char_start)
    # ``char_end`` is exclusive; the last character is at char_end - 1.
    page_end = doc.page_for_offset(max(char_start, char_end - 1))
    return Chunk(
        chunk_id=chunk_id_for(
            doc_id=doc.doc_id,
            strategy=strategy,
            char_start=char_start,
            char_end=char_end,
        ),
        doc_id=doc.doc_id,
        strategy=strategy,
        char_start=char_start,
        char_end=char_end,
        page_start=page_start,
        page_end=page_end,
        text=text,
        n_tokens=count_tokens(text),
        section_path=section_path,
    )


class Chunker(Protocol):
    """A pluggable chunking strategy."""

    name: str

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:  # pragma: no cover - Protocol
        ...


def save_chunks(chunks: list[Chunk], out_path: Path) -> str:
    """Write ``chunks`` to a one-per-line JSONL and return the sha256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(asdict(c), sort_keys=True) for c in chunks)
    payload_bytes = (payload + ("\n" if payload else "")).encode("utf-8")
    out_path.write_bytes(payload_bytes)
    return hashlib.sha256(payload_bytes).hexdigest()


def load_chunks(in_path: Path) -> list[Chunk]:
    """Inverse of :func:`save_chunks`."""
    chunks: list[Chunk] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunks.append(
            Chunk(
                chunk_id=record["chunk_id"],
                doc_id=record["doc_id"],
                strategy=record["strategy"],
                char_start=int(record["char_start"]),
                char_end=int(record["char_end"]),
                page_start=int(record["page_start"]),
                page_end=int(record["page_end"]),
                text=record["text"],
                n_tokens=int(record["n_tokens"]),
                section_path=tuple(record.get("section_path", ())),
            )
        )
    return chunks
