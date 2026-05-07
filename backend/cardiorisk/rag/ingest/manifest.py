"""Corpus manifest: the one artefact downstream phases consume.

Phase 3.2 retrieval, Phase 3.3 generation, and the Phase 6 eval
harness all open ``data/external/corpus/manifest.json`` and follow
its references rather than re-running the parser. This decouples
ingest cost from retrieval cost and lets the chunking choice change
without re-fetching PDFs.

Schema (frozen for the duration of Phase 3.x; bump
:data:`MANIFEST_SCHEMA_VERSION` if you break it):

.. code-block:: json

    {
      "schema_version": 1,
      "built_at": "2026-05-06T21:30:00Z",
      "sources": [
        {"doc_id": "...", "title": "...", "publisher": "...",
         "url": "...", "raw_path": "...", "raw_sha256": "..." }
      ],
      "parsed_docs": [
        {"doc_id": "...", "n_pages": 12, "n_chars": 32145,
         "parsed_path": "data/external/corpus/parsed/<doc_id>.jsonl",
         "parsed_sha256": "..."}
      ],
      "chunks_by_strategy": {
        "token":    {"n_chunks": 87, "chunks_path": "...", "chunks_sha256": "..."},
        "semantic": { ... },
        "hybrid":   { ... }
      }
    }

Path semantics: every path is stored *relative to the repo root*, so
the manifest is portable across machines (the
``REPO_ROOT``-relative-isation happens at write time and the inverse
at read time uses :data:`cardiorisk.data.paths.REPO_ROOT`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from cardiorisk.data.paths import REPO_ROOT

from .sources import CorpusSource

MANIFEST_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class SourceEntry:
    """One row of ``manifest["sources"]``."""

    doc_id: str
    title: str
    publisher: str
    url: str
    raw_path: str
    raw_sha256: str


@dataclass(frozen=True)
class ParsedDocEntry:
    """One row of ``manifest["parsed_docs"]``."""

    doc_id: str
    n_pages: int
    n_chars: int
    parsed_path: str
    parsed_sha256: str


@dataclass(frozen=True)
class ChunkStrategyEntry:
    """One row of ``manifest["chunks_by_strategy"]``."""

    n_chunks: int
    chunks_path: str
    chunks_sha256: str


@dataclass(frozen=True)
class Manifest:
    """In-memory view of a written manifest.

    Attributes:
        schema_version: bumps when the on-disk format changes.
        built_at: ISO-8601 UTC timestamp of the build.
        sources: One entry per fetched source PDF (or markdown
            fixture file in ``--use-fixture`` mode).
        parsed_docs: One entry per parsed JSONL.
        chunks_by_strategy: Per-strategy aggregate; missing keys
            mean that strategy was skipped (``--strategy <one>``).
    """

    schema_version: int
    built_at: str
    sources: tuple[SourceEntry, ...]
    parsed_docs: tuple[ParsedDocEntry, ...]
    chunks_by_strategy: dict[str, ChunkStrategyEntry] = field(default_factory=dict)


def _to_repo_relative(path: Path) -> str:
    """Return ``path`` relative to ``REPO_ROOT`` as a forward-slashed string.

    Raises :class:`ValueError` if ``path`` is not inside the repo
    (e.g. a tempdir during a unit test). In that case the absolute
    path is returned so the manifest is still readable.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _from_repo_relative(s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def build_manifest(
    *,
    sources: list[tuple[CorpusSource, Path, str]],
    parsed_docs: list[tuple[str, Path, str, int, int]],
    chunks_by_strategy: dict[str, tuple[Path, str, int]],
) -> Manifest:
    """Assemble a :class:`Manifest` from the orchestrator's outputs.

    Args:
        sources: ``[(source, raw_path, raw_sha256), ...]``.
        parsed_docs: ``[(doc_id, parsed_path, parsed_sha256, n_pages, n_chars), ...]``.
        chunks_by_strategy: ``{strategy_name: (chunks_path, chunks_sha256, n_chunks)}``.
    """
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        built_at=datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        sources=tuple(
            SourceEntry(
                doc_id=src.doc_id,
                title=src.title,
                publisher=src.publisher,
                url=src.url,
                raw_path=_to_repo_relative(raw_path),
                raw_sha256=raw_sha256,
            )
            for src, raw_path, raw_sha256 in sources
        ),
        parsed_docs=tuple(
            ParsedDocEntry(
                doc_id=doc_id,
                n_pages=n_pages,
                n_chars=n_chars,
                parsed_path=_to_repo_relative(parsed_path),
                parsed_sha256=parsed_sha256,
            )
            for doc_id, parsed_path, parsed_sha256, n_pages, n_chars in parsed_docs
        ),
        chunks_by_strategy={
            strategy: ChunkStrategyEntry(
                n_chunks=n_chunks,
                chunks_path=_to_repo_relative(chunks_path),
                chunks_sha256=chunks_sha256,
            )
            for strategy, (chunks_path, chunks_sha256, n_chunks) in chunks_by_strategy.items()
        },
    )


def save_manifest(manifest: Manifest, out_path: Path) -> None:
    """Write ``manifest`` to ``out_path`` as deterministic, sorted JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": manifest.schema_version,
        "built_at": manifest.built_at,
        "sources": [asdict(s) for s in manifest.sources],
        "parsed_docs": [asdict(d) for d in manifest.parsed_docs],
        "chunks_by_strategy": {
            name: asdict(entry) for name, entry in sorted(manifest.chunks_by_strategy.items())
        },
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(in_path: Path) -> Manifest:
    """Inverse of :func:`save_manifest`."""
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return Manifest(
        schema_version=int(payload["schema_version"]),
        built_at=str(payload["built_at"]),
        sources=tuple(SourceEntry(**s) for s in payload.get("sources", [])),
        parsed_docs=tuple(ParsedDocEntry(**d) for d in payload.get("parsed_docs", [])),
        chunks_by_strategy={
            name: ChunkStrategyEntry(**entry)
            for name, entry in payload.get("chunks_by_strategy", {}).items()
        },
    )


def resolve_chunks_path(manifest: Manifest, strategy: str) -> Path:
    """Return the absolute path to the chunks JSONL for ``strategy``.

    Raises :class:`KeyError` if the manifest doesn't carry that
    strategy.
    """
    if strategy not in manifest.chunks_by_strategy:
        known = ", ".join(sorted(manifest.chunks_by_strategy)) or "<none>"
        raise KeyError(f"strategy {strategy!r} not in manifest; known: {known}")
    return _from_repo_relative(manifest.chunks_by_strategy[strategy].chunks_path)


def resolve_parsed_path(manifest: Manifest, doc_id: str) -> Path:
    """Return the absolute path to the parsed JSONL for ``doc_id``."""
    for entry in manifest.parsed_docs:
        if entry.doc_id == doc_id:
            return _from_repo_relative(entry.parsed_path)
    known = ", ".join(sorted(e.doc_id for e in manifest.parsed_docs))
    raise KeyError(f"doc_id {doc_id!r} not in manifest; known: {known}")
