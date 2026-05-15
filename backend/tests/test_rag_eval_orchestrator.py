"""End-to-end orchestrator smoke against the markdown fixture.

Builds the corpus (token+semantic+hybrid), then runs the
orchestrator with the dep-free MockEmbedder + MockReranker so the
test stays under a second and doesn't pull big model weights.

Verifies: per_cell.json + aggregate.json schema, figures get written,
and the winning-cell pick is deterministic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cardiorisk.data.paths import REPO_ROOT
from cardiorisk.rag.eval_retrieval.orchestrator import (
    OrchestratorConfig,
    run,
)


def _build_corpus_into(tmp_path: Path) -> Path:
    """Run build_corpus.py --use-fixture into a tmp dir; return manifest path."""
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend" / "scripts" / "build_corpus.py"),
        "--use-fixture",
        "--strategy",
        "all",
        "--parsed-dir",
        str(parsed_dir),
        "--chunks-dir",
        str(chunks_dir),
        "--manifest-path",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603 - test-controlled args
    return manifest_path


@pytest.fixture()
def fixture_manifest(tmp_path: Path) -> Path:
    return _build_corpus_into(tmp_path)


@pytest.fixture()
def smoke_orchestrator_config(fixture_manifest: Path, tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        strategies=("token", "semantic", "hybrid"),
        rerank_conditions=(False, True),
        embedder_name="mock",
        reranker_name="mock",
        use_fixture=True,
        top_k=5,
        per_leg_k=20,
        smoke=True,
        n_resamples=200,
        manifest_path=fixture_manifest,
        embed_cache_dir=tmp_path / "embed_cache",
        index_dir=tmp_path / "index",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
    )


def test_run_writes_per_cell_and_aggregate_json(
    smoke_orchestrator_config: OrchestratorConfig,
) -> None:
    aggregate = run(smoke_orchestrator_config)
    per_cell_path = smoke_orchestrator_config.reports_dir / "per_cell.json"
    aggregate_path = smoke_orchestrator_config.reports_dir / "aggregate.json"
    assert per_cell_path.exists()
    assert aggregate_path.exists()
    saved = json.loads(aggregate_path.read_text())
    assert saved["n_cells"] == 6  # 3 chunkers x 2 rerank conditions
    assert "winning_cell" in saved
    assert aggregate["winning_cell"]["chunker"] in {"token", "semantic", "hybrid"}


def test_run_writes_three_figures(
    smoke_orchestrator_config: OrchestratorConfig,
) -> None:
    run(smoke_orchestrator_config)
    fig_dir = smoke_orchestrator_config.figures_dir
    assert (fig_dir / "hit_at_5_by_cell.png").exists()
    assert (fig_dir / "mrr_by_cell.png").exists()
    assert (fig_dir / "per_tag_winning_cell.png").exists()


def test_per_cell_json_schema(
    smoke_orchestrator_config: OrchestratorConfig,
) -> None:
    run(smoke_orchestrator_config)
    payload = json.loads((smoke_orchestrator_config.reports_dir / "per_cell.json").read_text())
    assert "config" in payload
    assert "cells" in payload
    for cell in payload["cells"]:
        assert {
            "label",
            "chunker",
            "with_rerank",
            "embedder",
            "reranker",
            "n_questions",
            "hit_at_1",
            "hit_at_5",
            "mrr",
            "ci_hit_at_5",
            "per_tag",
        } <= cell.keys()


def test_aggregate_json_schema(
    smoke_orchestrator_config: OrchestratorConfig,
) -> None:
    run(smoke_orchestrator_config)
    payload = json.loads((smoke_orchestrator_config.reports_dir / "aggregate.json").read_text())
    assert "winning_cell" in payload
    assert "per_chunker_max_hit_at_5" in payload
    assert "rerank_lift" in payload
    for _chunker, lift in payload["rerank_lift"].items():
        assert lift is None or isinstance(lift, (int, float))


def test_run_is_deterministic(
    smoke_orchestrator_config: OrchestratorConfig,
) -> None:
    a = run(smoke_orchestrator_config)
    b = run(smoke_orchestrator_config)
    # Same winner; same hit@5 point (CI bounds may shift if RNG state
    # advances differently between runs — we only assert the
    # deterministic point estimate).
    assert a["winning_cell"]["chunker"] == b["winning_cell"]["chunker"]
    assert a["winning_cell"]["hit_at_5"] == b["winning_cell"]["hit_at_5"]


def test_smoke_cli_runs_via_subprocess(tmp_path: Path) -> None:
    """End-to-end CLI smoke; uses the mock embedder so it's fast and dep-free."""
    manifest_path = _build_corpus_into(tmp_path)
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"

    # We run eval_retrieval.py directly with the mock embedder + mock
    # reranker by overriding the smoke config via CLI flags.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend" / "scripts" / "eval_retrieval.py"),
        "--use-fixture",
        "--rerank",
        "off",
        "--embedder",
        "mock",
        "--reranker",
        "mock",
        "--strategies",
        "hybrid",
        "--n-resamples",
        "200",
        "--manifest-path",
        str(manifest_path),
        "--reports-dir",
        str(reports_dir),
        "--figures-dir",
        str(figures_dir),
    ]
    proc = subprocess.run(  # noqa: S603 - test-controlled args
        cmd, check=True, capture_output=True, text=True
    )
    assert (reports_dir / "per_cell.json").exists()
    assert (reports_dir / "aggregate.json").exists()
    assert proc.returncode == 0
