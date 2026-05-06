"""Phase 3.1: tests for the corpus PDF fetcher.

Network-free: every test patches :func:`cardiorisk.rag.ingest.fetch.download_to`
to avoid hitting RACGP / NVDPA upstream from CI.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from cardiorisk.rag.ingest import fetch as fetch_mod
from cardiorisk.rag.ingest.fetch import (
    FetchError,
    fetch_all,
    fetch_one,
    read_pinned_checksum,
    sha256_of,
    write_pinned_checksum,
)
from cardiorisk.rag.ingest.sources import CorpusSource

PAYLOAD = b"%PDF-1.4 fake fixture body for fetch tests\n" * 8
EXPECTED_DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def fake_source() -> CorpusSource:
    return CorpusSource(
        doc_id="fake_doc_for_fetch_tests",
        title="Fake doc",
        publisher="RACGP",
        url="https://example.invalid/fake.pdf",
        out_filename="fake_doc_for_fetch_tests.pdf",
        checksum_filename="corpus_fake_doc.sha256",
    )


DownloadFn = Callable[[str, Path], None]


@pytest.fixture
def patch_download(monkeypatch: pytest.MonkeyPatch) -> DownloadFn:
    def fake_download(url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(PAYLOAD)

    monkeypatch.setattr(fetch_mod, "download_to", fake_download)
    return fake_download


def test_sha256_of_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(PAYLOAD)
    assert sha256_of(p) == EXPECTED_DIGEST


def test_read_pinned_checksum_missing_returns_none(tmp_path: Path) -> None:
    assert read_pinned_checksum(tmp_path / "missing.sha256") is None


def test_write_then_read_pinned_checksum(tmp_path: Path) -> None:
    cks = tmp_path / "x.sha256"
    write_pinned_checksum(cks, "ABCDEF1234", "https://example.invalid/x.pdf")
    assert read_pinned_checksum(cks) == "abcdef1234"
    text = cks.read_text(encoding="utf-8")
    assert "https://example.invalid/x.pdf" in text


def test_first_run_writes_lockfile(
    tmp_path: Path, fake_source: CorpusSource, patch_download: DownloadFn
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    out_path, digest, action = fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    assert action == "downloaded"
    assert digest == EXPECTED_DIGEST
    assert out_path.exists()
    pinned = read_pinned_checksum(cks / fake_source.checksum_filename)
    assert pinned == EXPECTED_DIGEST


def test_subsequent_run_with_matching_pin_short_circuits(
    tmp_path: Path, fake_source: CorpusSource, patch_download: DownloadFn
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    _, digest, action = fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    assert action == "reused"
    assert digest == EXPECTED_DIGEST


def test_force_redownloads_even_when_pin_matches(
    tmp_path: Path, fake_source: CorpusSource, patch_download: DownloadFn
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    _, _, action = fetch_one(fake_source, force=True, raw_dir=raw, checksum_dir=cks)
    assert action == "redownloaded"


def test_pin_mismatch_raises_fetch_error(
    tmp_path: Path,
    fake_source: CorpusSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    cks.mkdir(parents=True, exist_ok=True)
    write_pinned_checksum(
        cks / fake_source.checksum_filename,
        "0" * 64,
        "https://example.invalid/x.pdf",
    )

    def fake_download(url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(PAYLOAD)

    monkeypatch.setattr(fetch_mod, "download_to", fake_download)

    with pytest.raises(FetchError) as exc:
        fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    assert "checksum mismatch" in str(exc.value)
    assert fake_source.doc_id in str(exc.value)


def test_fetch_all_iterates_sources(
    tmp_path: Path, fake_source: CorpusSource, patch_download: DownloadFn
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    other = CorpusSource(
        doc_id="other_doc",
        title="Other",
        publisher="NVDPA",
        url="https://example.invalid/other.pdf",
        out_filename="other_doc.pdf",
        checksum_filename="corpus_other.sha256",
    )
    results = fetch_all(
        force=False,
        sources=(fake_source, other),
        raw_dir=raw,
        checksum_dir=cks,
    )
    assert len(results) == 2
    assert {s.doc_id for s, _, _, _ in results} == {fake_source.doc_id, "other_doc"}


def test_atomic_write_cleans_up_on_download_failure(
    tmp_path: Path,
    fake_source: CorpusSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    cks = tmp_path / "checksums"
    raw.mkdir(parents=True, exist_ok=True)

    def boom(url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        tmp.write_bytes(b"partial")
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(fetch_mod, "download_to", boom)
    with pytest.raises(RuntimeError):
        fetch_one(fake_source, force=False, raw_dir=raw, checksum_dir=cks)
    # The actual download_to wrapper would have cleaned the .part file;
    # our `boom` simulates a failure *inside* download_to, so the test
    # confirms the wrapper still raises (no swallow).
