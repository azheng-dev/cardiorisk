"""CLI: fetch the Phase 3.1 guideline corpus PDFs.

Thin wrapper around :mod:`cardiorisk.rag.ingest.fetch`. Downloads the
RACGP Red Book + NVDPA absolute-CVD-risk PDFs listed in
:data:`cardiorisk.rag.ingest.sources.CORPUS_SOURCES`, verifies their
pinned SHA-256 lockfiles under ``data/checksums/``, and writes them
to ``data/external/corpus/raw/``.

Modes:
    --use-fixture     skip network entirely; copy the markdown
                      fixture into raw_dir as ``*.md`` files (CI
                      smoke-test mode; PDF parsing is then bypassed
                      by ``build_corpus.py --use-fixture``).
    --source <id>     fetch only one ``doc_id`` (repeatable). Default
                      fetches all sources.
    --force           re-download even if checksums match.

Usage:
    uv run python backend/scripts/fetch_corpus.py
    uv run python backend/scripts/fetch_corpus.py --use-fixture
    uv run python backend/scripts/fetch_corpus.py --source racgp_redbook_cvd
"""

from __future__ import annotations

# OpenMP-guard preamble (kept identical to compute_drift.py /
# compute_explanations.py): pdfplumber + tiktoken don't need it, but
# keeping every script's header invariant means a future agent can
# copy-paste a new script without thinking about which deps deadlock
# on macOS.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import requests

from cardiorisk.data.paths import CORPUS_RAW, FIXTURE_CORPUS_DIR, REPO_ROOT
from cardiorisk.rag.ingest.fetch import FetchError, fetch_all
from cardiorisk.rag.ingest.sources import CORPUS_SOURCES, CorpusSource, get_source


def _format_action(items: Iterable[tuple[str, Path, str, str]]) -> str:
    return "\n".join(
        f"  [{action:<13}] {name:<48} -> {path.relative_to(REPO_ROOT)}  sha256={digest[:12]}..."
        for name, path, digest, action in items
    )


def _copy_fixture_to_raw(*, raw_dir: Path = CORPUS_RAW) -> list[tuple[str, Path]]:
    """Copy the markdown fixture documents listed in ``sources.json``.

    Returns ``(doc_id, dest_path)`` for each copied file. We read the
    fixture's ``sources.json`` rather than globbing ``*.md`` so the
    fixture's own ``README.md`` (and any future docs) is not picked
    up by accident.
    """
    if not FIXTURE_CORPUS_DIR.exists():
        raise FetchError(
            f"fixture corpus not found at {FIXTURE_CORPUS_DIR}. "
            "Did backend/tests/fixtures/corpus_mini/ get deleted?"
        )
    sources_json = FIXTURE_CORPUS_DIR / "sources.json"
    if not sources_json.exists():
        raise FetchError(f"fixture sources.json not found at {sources_json}")
    payload = json.loads(sources_json.read_text(encoding="utf-8"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, Path]] = []
    for entry in payload["sources"]:
        md_path = FIXTURE_CORPUS_DIR / entry["filename"]
        if not md_path.exists():
            raise FetchError(f"fixture file not found: {md_path}")
        dest = raw_dir / md_path.name
        shutil.copyfile(md_path, dest)
        copied.append((entry["doc_id"], dest))
    if not copied:
        raise FetchError(f"sources.json under {FIXTURE_CORPUS_DIR} has zero entries")
    return copied


def _resolve_sources(names: list[str] | None) -> tuple[CorpusSource, ...]:
    """Return the subset of CORPUS_SOURCES selected by ``--source``."""
    if not names:
        return CORPUS_SOURCES
    return tuple(get_source(name) for name in names)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="skip network; copy the markdown fixture to raw_dir (CI mode)",
    )
    parser.add_argument(
        "--source",
        action="append",
        metavar="DOC_ID",
        help=(
            "fetch only this doc_id (repeatable); default fetches all "
            f"{len(CORPUS_SOURCES)} sources"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if checksums match",
    )
    args = parser.parse_args()

    if args.use_fixture:
        if args.source:
            print(
                "ERROR: --source and --use-fixture are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        try:
            copied = _copy_fixture_to_raw()
        except FetchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("Fixture corpus (markdown):")
        for doc_id, dest in copied:
            print(f"  [fixture     ] {doc_id:<48} -> {dest.relative_to(REPO_ROOT)}")
        return 0

    try:
        sources = _resolve_sources(args.source)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        results = fetch_all(force=args.force, sources=sources)
    except (FetchError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Corpus PDFs:")
    print(
        _format_action(
            (source.doc_id, path, digest, action) for source, path, digest, action in results
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
