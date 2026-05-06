"""Phase 3.1: tests for the corpus manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from cardiorisk.rag.ingest.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    build_manifest,
    load_manifest,
    resolve_chunks_path,
    resolve_parsed_path,
    save_manifest,
)
from cardiorisk.rag.ingest.sources import CorpusSource


@pytest.fixture
def fake_source() -> CorpusSource:
    return CorpusSource(
        doc_id="manifest_test_doc",
        title="Manifest test",
        publisher="RACGP",
        url="https://example.invalid/x.pdf",
        out_filename="manifest_test_doc.pdf",
        checksum_filename="corpus_manifest_test_doc.sha256",
    )


@pytest.fixture
def sample_manifest(tmp_path: Path, fake_source: CorpusSource) -> Manifest:
    raw = tmp_path / "x.pdf"
    raw.write_bytes(b"raw bytes")
    parsed = tmp_path / "x.jsonl"
    parsed.write_text("{}", encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}", encoding="utf-8")
    return build_manifest(
        sources=[(fake_source, raw, "deadbeef")],
        parsed_docs=[(fake_source.doc_id, parsed, "cafefade", 3, 1234)],
        chunks_by_strategy={
            "token": (chunks, "11" * 32, 7),
            "semantic": (chunks, "22" * 32, 5),
        },
    )


def test_manifest_schema_version_is_one() -> None:
    assert MANIFEST_SCHEMA_VERSION == 1


def test_build_manifest_records_source_metadata(
    sample_manifest: Manifest, fake_source: CorpusSource
) -> None:
    assert len(sample_manifest.sources) == 1
    s = sample_manifest.sources[0]
    assert s.doc_id == fake_source.doc_id
    assert s.publisher == fake_source.publisher
    assert s.url == fake_source.url
    assert s.raw_sha256 == "deadbeef"


def test_build_manifest_records_parsed_doc_metadata(sample_manifest: Manifest) -> None:
    assert len(sample_manifest.parsed_docs) == 1
    d = sample_manifest.parsed_docs[0]
    assert d.n_pages == 3
    assert d.n_chars == 1234


def test_build_manifest_records_chunks_per_strategy(sample_manifest: Manifest) -> None:
    assert set(sample_manifest.chunks_by_strategy) == {"token", "semantic"}
    assert sample_manifest.chunks_by_strategy["token"].n_chunks == 7
    assert sample_manifest.chunks_by_strategy["semantic"].n_chunks == 5


def test_save_then_load_round_trip(sample_manifest: Manifest, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    save_manifest(sample_manifest, out)
    loaded = load_manifest(out)
    assert loaded == sample_manifest


def test_save_writes_human_readable_indented_json(
    sample_manifest: Manifest, tmp_path: Path
) -> None:
    out = tmp_path / "manifest.json"
    save_manifest(sample_manifest, out)
    text = out.read_text(encoding="utf-8")
    assert "\n  " in text
    assert text.endswith("\n")


def test_built_at_is_iso8601_z(sample_manifest: Manifest) -> None:
    assert sample_manifest.built_at.endswith("Z")
    assert "T" in sample_manifest.built_at


def test_resolve_chunks_path_returns_existing_file(
    sample_manifest: Manifest, tmp_path: Path
) -> None:
    p = resolve_chunks_path(sample_manifest, "token")
    assert p.exists()


def test_resolve_chunks_path_unknown_strategy_raises(sample_manifest: Manifest) -> None:
    with pytest.raises(KeyError):
        resolve_chunks_path(sample_manifest, "definitely_not_a_strategy")


def test_resolve_parsed_path_unknown_doc_raises(sample_manifest: Manifest) -> None:
    with pytest.raises(KeyError):
        resolve_parsed_path(sample_manifest, "no_such_doc")


def test_paths_are_repo_relative_when_inside_repo(
    fake_source: CorpusSource, tmp_path: Path
) -> None:
    """Paths inside REPO_ROOT should be persisted as repo-relative.

    Paths outside REPO_ROOT (a tempdir on macOS lives under /private/var)
    fall back to absolute, which the test exercises implicitly via the
    sample_manifest fixture using tmp_path.
    """
    # Use a path that's actually inside the repo.
    from cardiorisk.data.paths import REPO_ROOT

    raw = REPO_ROOT / "backend" / "tests" / "fixtures" / "hfp_mini.csv"
    parsed = REPO_ROOT / "backend" / "tests" / "fixtures" / "hfp_mini.csv"
    chunks = REPO_ROOT / "backend" / "tests" / "fixtures" / "hfp_mini.csv"
    m = build_manifest(
        sources=[(fake_source, raw, "ab")],
        parsed_docs=[(fake_source.doc_id, parsed, "cd", 1, 1)],
        chunks_by_strategy={"token": (chunks, "ef", 1)},
    )
    assert not m.sources[0].raw_path.startswith("/")
    assert m.sources[0].raw_path == "backend/tests/fixtures/hfp_mini.csv"


def test_chunks_by_strategy_default_empty() -> None:
    m = Manifest(
        schema_version=1,
        built_at="2026-01-01T00:00:00Z",
        sources=(),
        parsed_docs=(),
    )
    assert m.chunks_by_strategy == {}


def test_load_manifest_handles_empty_chunks_by_strategy(tmp_path: Path) -> None:
    m = Manifest(
        schema_version=1,
        built_at="2026-01-01T00:00:00Z",
        sources=(),
        parsed_docs=(),
    )
    out = tmp_path / "x.json"
    save_manifest(m, out)
    loaded = load_manifest(out)
    assert loaded.chunks_by_strategy == {}
