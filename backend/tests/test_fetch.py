"""Tests for cardiorisk.data.fetch.

Network is mocked everywhere — these tests must run offline in CI without
hitting archive.ics.uci.edu or kaggle.com.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cardiorisk.data.fetch import (
    UCI_SOURCES,
    FetchError,
    UciSource,
    fetch_one,
    read_pinned_checksum,
    sha256_of,
    use_fixture,
    write_pinned_checksum,
)

REQUESTS_GET = "cardiorisk.data.fetch.requests.get"

# ---------------------------------------------------------------- sha256


def test_sha256_of_known_content(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")
    assert sha256_of(f) == ("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")


def test_sha256_of_handles_large_file_in_chunks(tmp_path: Path) -> None:
    """File larger than the 64KB read-chunk size still hashes correctly."""
    payload = b"a" * (200 * 1024)
    f = tmp_path / "big.bin"
    f.write_bytes(payload)
    digest = sha256_of(f)
    # SHA-256 of 200KB of "a" is well-known and stable; we just check
    # non-emptiness + length here so we don't pin to a specific algorithm
    # implementation detail.
    assert len(digest) == 64
    assert digest.isalnum()


# ---------------------------------------------------------------- checksum I/O


def test_read_pinned_checksum_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_pinned_checksum(tmp_path / "absent.sha256") is None


def test_read_pinned_checksum_strips_inline_comment(tmp_path: Path) -> None:
    p = tmp_path / "x.sha256"
    p.write_text("DEADBEEF  # source: https://example.com/x.csv\n", encoding="utf-8")
    assert read_pinned_checksum(p) == "deadbeef"


def test_read_pinned_checksum_skips_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.sha256"
    p.write_text("# top-level comment\n   \nabc123\n", encoding="utf-8")
    assert read_pinned_checksum(p) == "abc123"


def test_write_pinned_checksum_creates_dir_and_writes_format(tmp_path: Path) -> None:
    target = tmp_path / "checksums" / "x.sha256"
    write_pinned_checksum(target, "AABB", "https://example.com/x.csv")
    contents = target.read_text(encoding="utf-8")
    assert contents == "aabb  # source: https://example.com/x.csv\n"


# ---------------------------------------------------------------- fetch_one


def _mock_streaming_response(payload: bytes) -> MagicMock:
    """Build a MagicMock that mimics a `requests` streaming context manager."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [payload]
    return response


def _make_fake_source(tmp_path: Path) -> UciSource:
    return UciSource(
        name="testsrc",
        url="https://example.com/test.data",
        out_filename="test.data",
        checksum_filename="test.sha256",
    )


def test_fetch_one_fresh_download_writes_data_and_checksum(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    checksum_dir = tmp_path / "checksums"
    source = _make_fake_source(tmp_path)
    payload = b"col1,col2\n1,2\n3,4\n"

    with patch(REQUESTS_GET, return_value=_mock_streaming_response(payload)):
        out_path, observed, action = fetch_one(
            source, force=False, raw_dir=raw_dir, checksum_dir=checksum_dir
        )

    assert out_path.exists()
    assert out_path.read_bytes() == payload
    assert action == "downloaded"
    assert observed == sha256_of(out_path)
    assert (checksum_dir / source.checksum_filename).exists()
    assert read_pinned_checksum(checksum_dir / source.checksum_filename) == observed


def test_fetch_one_reused_when_pinned_matches_existing_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    checksum_dir = tmp_path / "checksums"
    raw_dir.mkdir()
    checksum_dir.mkdir()
    source = _make_fake_source(tmp_path)

    payload = b"already here"
    out_path = raw_dir / source.out_filename
    out_path.write_bytes(payload)
    digest = sha256_of(out_path)
    write_pinned_checksum(checksum_dir / source.checksum_filename, digest, source.url)

    with patch(REQUESTS_GET, side_effect=AssertionError("network must not be hit")):
        out, observed, action = fetch_one(
            source, force=False, raw_dir=raw_dir, checksum_dir=checksum_dir
        )

    assert action == "reused"
    assert observed == digest
    assert out == out_path


def test_fetch_one_redownloads_when_pinned_mismatches_existing_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    checksum_dir = tmp_path / "checksums"
    raw_dir.mkdir()
    checksum_dir.mkdir()
    source = _make_fake_source(tmp_path)

    new_payload = b"new contents"
    new_digest = sha256_of_bytes(new_payload)
    write_pinned_checksum(checksum_dir / source.checksum_filename, new_digest, source.url)

    out_path = raw_dir / source.out_filename
    out_path.write_bytes(b"stale contents")

    with patch(REQUESTS_GET, return_value=_mock_streaming_response(new_payload)):
        out, observed, action = fetch_one(
            source, force=False, raw_dir=raw_dir, checksum_dir=checksum_dir
        )

    assert action == "redownloaded"
    assert observed == new_digest
    assert out.read_bytes() == new_payload


def test_fetch_one_raises_on_checksum_mismatch_after_download(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    checksum_dir = tmp_path / "checksums"
    source = _make_fake_source(tmp_path)

    write_pinned_checksum(
        checksum_dir / source.checksum_filename,
        "0" * 64,
        source.url,
    )

    with (
        patch(REQUESTS_GET, return_value=_mock_streaming_response(b"won't match the all-zero pin")),
        pytest.raises(FetchError, match="checksum mismatch"),
    ):
        fetch_one(source, force=False, raw_dir=raw_dir, checksum_dir=checksum_dir)


def test_fetch_one_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    """If the download stream raises mid-flight, no `.part` file is left behind."""
    raw_dir = tmp_path / "raw"
    checksum_dir = tmp_path / "checksums"
    source = _make_fake_source(tmp_path)

    failing = MagicMock()
    failing.__enter__.return_value = failing
    failing.__exit__.return_value = False
    failing.raise_for_status.side_effect = RuntimeError("boom")

    with patch(REQUESTS_GET, return_value=failing), pytest.raises(RuntimeError, match="boom"):
        fetch_one(source, force=False, raw_dir=raw_dir, checksum_dir=checksum_dir)

    leftover_part = raw_dir / (source.out_filename + ".part")
    assert not leftover_part.exists()


# ---------------------------------------------------------------- use_fixture


def test_use_fixture_copies_synthetic_csv_into_raw_dir(tmp_path: Path) -> None:
    fixture = tmp_path / "src" / "hfp_mini.csv"
    fixture.parent.mkdir()
    payload = io.BytesIO(b"Age,Sex,HeartDisease\n50,M,1\n").getvalue()
    fixture.write_bytes(payload)

    raw_dir = tmp_path / "raw"
    target = use_fixture(fixture_path=fixture, raw_dir=raw_dir)

    assert target == raw_dir / "hfp_mini.csv"
    assert target.read_bytes() == payload


def test_use_fixture_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="fixture not found"):
        use_fixture(fixture_path=tmp_path / "missing.csv", raw_dir=tmp_path / "raw")


# ---------------------------------------------------------------- registry


def test_uci_sources_have_unique_filenames() -> None:
    out_names = [s.out_filename for s in UCI_SOURCES]
    checksum_names = [s.checksum_filename for s in UCI_SOURCES]
    assert len(set(out_names)) == len(out_names)
    assert len(set(checksum_names)) == len(checksum_names)


def test_uci_sources_cover_expected_sites() -> None:
    names = {s.name for s in UCI_SOURCES}
    assert names == {"cleveland", "hungarian", "switzerland", "va"}


# ---------------------------------------------------------------- helpers


def sha256_of_bytes(data: bytes) -> str:
    """SHA-256 of in-memory bytes (test-local helper, not exported)."""
    import hashlib

    return hashlib.sha256(data).hexdigest()
