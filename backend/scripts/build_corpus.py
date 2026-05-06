"""CLI: parse the fetched corpus and emit chunks + manifest.

Reads source PDFs (or markdown fixtures in ``--use-fixture`` mode)
from ``data/external/corpus/raw/``, parses each into a per-page
JSONL under ``data/external/corpus/parsed/<doc_id>.jsonl``, runs the
selected chunking strategies, and writes
``data/external/corpus/manifest.json`` referencing every artefact by
repo-relative path + sha256.

Modes:
    --use-fixture          read ``backend/tests/fixtures/corpus_mini/
                           sources.json`` instead of the real
                           CORPUS_SOURCES list; CI mode.
    --strategy {all,token,semantic,hybrid}
                           run only the named chunker (default: all).

The CLI is intentionally idempotent: rerunning with the same inputs
produces byte-identical chunk JSONLs (chunk ids are deterministic
hashes; the manifest's ``built_at`` is the only field that changes
between runs).

Usage:
    uv run python backend/scripts/build_corpus.py
    uv run python backend/scripts/build_corpus.py --use-fixture
    uv run python backend/scripts/build_corpus.py --use-fixture --strategy token
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from cardiorisk.data.paths import (
    CORPUS_CHUNKS,
    CORPUS_MANIFEST,
    CORPUS_PARSED,
    CORPUS_RAW,
    FIXTURE_CORPUS_DIR,
    REPO_ROOT,
)
from cardiorisk.rag.ingest.chunkers import NAME_TO_CHUNKER, save_chunks
from cardiorisk.rag.ingest.manifest import build_manifest, save_manifest
from cardiorisk.rag.ingest.parse import (
    ParseError,
    parse_doc_for_source,
    save_parsed_doc,
)
from cardiorisk.rag.ingest.sources import CORPUS_SOURCES, CorpusSource


def _display_path(p: Path) -> str:
    """Render ``p`` as repo-relative if possible, else absolute."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


@dataclass(frozen=True)
class _ResolvedSource:
    """A :class:`CorpusSource` plus the on-disk path to its raw bytes."""

    source: CorpusSource
    raw_path: Path


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_real_sources(raw_dir: Path) -> list[_ResolvedSource]:
    """Locate each :data:`CORPUS_SOURCES` entry's PDF in ``raw_dir``.

    Raises :class:`FileNotFoundError` listing every missing source so
    the maintainer can run ``fetch_corpus.py`` and re-try.
    """
    missing: list[str] = []
    resolved: list[_ResolvedSource] = []
    for source in CORPUS_SOURCES:
        raw_path = raw_dir / source.out_filename
        if not raw_path.exists():
            missing.append(f"  - {source.doc_id}: expected {_display_path(raw_path)}")
            continue
        resolved.append(_ResolvedSource(source=source, raw_path=raw_path))
    if missing:
        raise FileNotFoundError(
            "Missing raw corpus files; run backend/scripts/fetch_corpus.py first.\n"
            + "\n".join(missing)
        )
    return resolved


def _resolve_fixture_sources() -> list[_ResolvedSource]:
    """Read the fixture's ``sources.json`` and resolve each markdown file."""
    sources_json = FIXTURE_CORPUS_DIR / "sources.json"
    if not sources_json.exists():
        raise FileNotFoundError(
            f"fixture sources.json not found at {sources_json}; "
            "did backend/tests/fixtures/corpus_mini/ get deleted?"
        )
    payload = json.loads(sources_json.read_text(encoding="utf-8"))
    resolved: list[_ResolvedSource] = []
    for entry in payload["sources"]:
        md_path = FIXTURE_CORPUS_DIR / entry["filename"]
        if not md_path.exists():
            raise FileNotFoundError(f"fixture file not found: {md_path}")
        resolved.append(
            _ResolvedSource(
                source=CorpusSource(
                    doc_id=entry["doc_id"],
                    title=entry["title"],
                    publisher=entry["publisher"],
                    url=entry["url"],
                    out_filename=entry["filename"],
                    checksum_filename=f"corpus_fixture_{entry['doc_id']}.sha256",
                ),
                raw_path=md_path,
            )
        )
    return resolved


def _select_strategies(name: str) -> list[str]:
    if name == "all":
        return sorted(NAME_TO_CHUNKER)
    if name not in NAME_TO_CHUNKER:
        known = ", ".join([*sorted(NAME_TO_CHUNKER), "all"])
        raise ValueError(f"unknown strategy {name!r}; known: {known}")
    return [name]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help=(
            "read the markdown fixture under "
            "backend/tests/fixtures/corpus_mini/ instead of the real "
            "PDF corpus (CI mode)"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=[*sorted(NAME_TO_CHUNKER), "all"],
        default="all",
        help="run only this chunker (default: all)",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=CORPUS_PARSED,
        help=f"output dir for parsed JSONLs (default: {CORPUS_PARSED.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CORPUS_CHUNKS,
        help=f"output dir for chunks JSONLs (default: {CORPUS_CHUNKS.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CORPUS_MANIFEST,
        help=f"output path for manifest.json (default: {CORPUS_MANIFEST.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    try:
        if args.use_fixture:
            resolved = _resolve_fixture_sources()
        else:
            resolved = _resolve_real_sources(CORPUS_RAW)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        strategies = _select_strategies(args.strategy)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sources_for_manifest: list[tuple[CorpusSource, Path, str]] = []
    parsed_for_manifest: list[tuple[str, Path, str, int, int]] = []
    parsed_docs = []  # in-memory; chunkers consume these

    print(f"Parsing {len(resolved)} source(s)...")
    for r in resolved:
        try:
            doc = parse_doc_for_source(r.raw_path, doc_id=r.source.doc_id)
        except ParseError as exc:
            print(f"ERROR parsing {r.source.doc_id}: {exc}", file=sys.stderr)
            return 1
        parsed_path = args.parsed_dir / f"{r.source.doc_id}.jsonl"
        parsed_sha = save_parsed_doc(doc, parsed_path)
        n_chars = sum(len(p.text) for p in doc.pages)
        sources_for_manifest.append((r.source, r.raw_path, _sha256_of(r.raw_path)))
        parsed_for_manifest.append(
            (r.source.doc_id, parsed_path, parsed_sha, len(doc.pages), n_chars)
        )
        parsed_docs.append(doc)
        print(f"  [parsed ] {r.source.doc_id:<48} pages={len(doc.pages):>3}  chars={n_chars:>7}")

    chunks_by_strategy: dict[str, tuple[Path, str, int]] = {}
    for strategy in strategies:
        chunker = NAME_TO_CHUNKER[strategy]()
        all_chunks = []
        for doc in parsed_docs:
            all_chunks.extend(chunker.chunk(doc))
        chunks_path = args.chunks_dir / f"{strategy}.jsonl"
        chunks_sha = save_chunks(all_chunks, chunks_path)
        chunks_by_strategy[strategy] = (chunks_path, chunks_sha, len(all_chunks))
        print(
            f"  [chunked] strategy={strategy:<10} chunks={len(all_chunks):>5}  "
            f"-> {_display_path(chunks_path)}"
        )

    manifest = build_manifest(
        sources=sources_for_manifest,
        parsed_docs=parsed_for_manifest,
        chunks_by_strategy=chunks_by_strategy,
    )
    save_manifest(manifest, args.manifest_path)
    print(f"  [manifest] -> {_display_path(args.manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
