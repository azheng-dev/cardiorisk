"""Phase 3.1: end-to-end CLI smoke for build_corpus.py.

Network-free: uses --use-fixture so the markdown corpus_mini fixture
exercises every code path (parse + 3 chunkers + manifest write).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cardiorisk.data.paths import REPO_ROOT
from cardiorisk.rag.ingest.chunkers import load_chunks
from cardiorisk.rag.ingest.manifest import load_manifest

BUILD_SCRIPT = REPO_ROOT / "backend" / "scripts" / "build_corpus.py"
FETCH_SCRIPT = REPO_ROOT / "backend" / "scripts" / "fetch_corpus.py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - args are test-controlled paths
        [sys.executable, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=None,
    )


def _build_in(tmp_path: Path, *, strategy: str = "all") -> dict[str, Any]:
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    manifest = tmp_path / "manifest.json"
    proc = _run(
        [
            str(BUILD_SCRIPT),
            "--use-fixture",
            "--strategy",
            strategy,
            "--parsed-dir",
            str(parsed_dir),
            "--chunks-dir",
            str(chunks_dir),
            "--manifest-path",
            str(manifest),
        ],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert manifest.exists()
    payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    return payload


def test_smoke_strategy_all(tmp_path: Path) -> None:
    payload = _build_in(tmp_path, strategy="all")
    assert payload["schema_version"] == 1
    assert {s["doc_id"] for s in payload["sources"]} == {
        "fixture_racgp_cvd",
        "fixture_nvdpa_quickref",
    }
    assert set(payload["chunks_by_strategy"]) == {"token", "semantic", "hybrid"}


def test_smoke_strategy_only_token(tmp_path: Path) -> None:
    payload = _build_in(tmp_path, strategy="token")
    assert set(payload["chunks_by_strategy"]) == {"token"}


def test_smoke_writes_chunks_jsonl_per_strategy(tmp_path: Path) -> None:
    payload = _build_in(tmp_path, strategy="all")
    for strategy in ("token", "semantic", "hybrid"):
        entry = payload["chunks_by_strategy"][strategy]
        chunks_path = (
            REPO_ROOT / entry["chunks_path"]
            if not Path(entry["chunks_path"]).is_absolute()
            else Path(entry["chunks_path"])
        )
        # The orchestrator wrote into tmp_path so manifest paths are absolute.
        assert chunks_path.exists(), chunks_path
        chunks = load_chunks(chunks_path)
        assert chunks, strategy


def test_smoke_writes_parsed_jsonl_per_doc(tmp_path: Path) -> None:
    payload = _build_in(tmp_path, strategy="all")
    for entry in payload["parsed_docs"]:
        parsed_path = (
            REPO_ROOT / entry["parsed_path"]
            if not Path(entry["parsed_path"]).is_absolute()
            else Path(entry["parsed_path"])
        )
        assert parsed_path.exists()
        assert entry["n_pages"] >= 1
        assert entry["n_chars"] > 0


def test_smoke_manifest_loadable_via_load_manifest(tmp_path: Path) -> None:
    """End-to-end: manifest written by CLI loads back into a Manifest object."""
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest.json"
    proc = _run(
        [
            str(BUILD_SCRIPT),
            "--use-fixture",
            "--parsed-dir",
            str(parsed_dir),
            "--chunks-dir",
            str(chunks_dir),
            "--manifest-path",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = load_manifest(manifest_path)
    assert manifest.schema_version == 1
    assert len(manifest.sources) == 2
    assert len(manifest.parsed_docs) == 2


def test_smoke_idempotent_chunk_ids(tmp_path: Path) -> None:
    """Running the build twice produces byte-identical chunk JSONLs."""
    payload_a = _build_in(tmp_path / "a", strategy="all")
    payload_b = _build_in(tmp_path / "b", strategy="all")
    for strategy in payload_a["chunks_by_strategy"]:
        sha_a = payload_a["chunks_by_strategy"][strategy]["chunks_sha256"]
        sha_b = payload_b["chunks_by_strategy"][strategy]["chunks_sha256"]
        assert sha_a == sha_b, strategy


def test_unknown_strategy_returns_nonzero(tmp_path: Path) -> None:
    proc = _run(
        [
            str(BUILD_SCRIPT),
            "--use-fixture",
            "--strategy",
            "definitely_not_a_strategy",
        ],
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_returns_zero(flag: str) -> None:
    proc = _run([str(BUILD_SCRIPT), flag], cwd=REPO_ROOT)
    assert proc.returncode == 0
    assert "build_corpus.py" in proc.stdout or "build_corpus.py" in proc.stderr


def test_fetch_corpus_use_fixture_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_corpus.py --use-fixture copies markdown into raw_dir."""
    proc = _run([str(FETCH_SCRIPT), "--use-fixture"], cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    # The default --use-fixture path uses CORPUS_RAW; verify the call
    # didn't crash. We don't introspect raw_dir contents because other
    # tests share the location; the `subprocess` exit code is enough.


def test_fetch_corpus_help_returns_zero() -> None:
    proc = _run([str(FETCH_SCRIPT), "--help"], cwd=REPO_ROOT)
    assert proc.returncode == 0
