"""Phase 3.1 corpus-ingestion pipeline (binding decision: ADR-015).

Fetches the Australian CVD-risk guideline corpus (RACGP Red Book +
NVDPA absolute-CVD-risk materials), parses each PDF to per-page text
with character offsets, and emits chunks under three pluggable
strategies. The chunking *winner* is **not** picked here — Phase 3.2
runs the 50-Q retrieval eval that decides between them.

Module map:

- :mod:`.sources` — :class:`CorpusSource` dataclass + the
  ``CORPUS_SOURCES`` tuple (publisher, title, URL, sha256 lockfile
  filename, ``doc_id``).
- :mod:`.fetch` — idempotent PDF fetcher; mirrors
  :mod:`cardiorisk.data.fetch` (sha256-pin, atomic write,
  ``FetchError`` on mismatch). ``--use-fixture`` short-circuits the
  network entirely.
- :mod:`.parse` — :class:`ParsedDoc` (per-page text + char offsets)
  produced by :func:`parse_pdf` (pdfplumber) or
  :func:`parse_markdown_fixture` (CI / unit-test path).
- :mod:`.chunkers` — :class:`Chunker` Protocol + :class:`Chunk`
  dataclass + a ``NAME_TO_CHUNKER`` registry mapping
  ``{"token", "semantic", "hybrid"}`` to concrete strategies.
- :mod:`.manifest` — :class:`Manifest` dataclass + build / save /
  load helpers; the manifest is the one artefact that downstream
  retrieval (Phase 3.2) consumes.

Storage (per ADR-015):

- ``data/external/corpus/raw/`` — downloaded PDFs (gitignored).
- ``data/external/corpus/parsed/`` — per-doc JSONL of pages
  (gitignored, derived).
- ``data/external/corpus/chunks/`` — per-strategy JSONL of chunks
  (gitignored, derived).
- ``data/external/corpus/manifest.json`` — gitignored, derived.
- ``data/checksums/corpus_<doc_id>.sha256`` — committed; pins each
  fetched PDF.
"""

from .chunkers import NAME_TO_CHUNKER, Chunk, Chunker
from .manifest import Manifest, build_manifest, load_manifest, save_manifest
from .parse import ParsedDoc, ParsedPage, parse_markdown_fixture, parse_pdf
from .sources import CORPUS_SOURCES, CorpusSource

__all__ = [
    "CORPUS_SOURCES",
    "NAME_TO_CHUNKER",
    "Chunk",
    "Chunker",
    "CorpusSource",
    "Manifest",
    "ParsedDoc",
    "ParsedPage",
    "build_manifest",
    "load_manifest",
    "parse_markdown_fixture",
    "parse_pdf",
    "save_manifest",
]
