"""End-to-end orchestrator smoke for the generation eval.

Builds the markdown fixture corpus, then runs the orchestrator with
the dep-free mock LLM + mock NLI + mock embedder.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cardiorisk.data.paths import REPO_ROOT
from cardiorisk.rag.eval_generation.orchestrator import (
    OrchestratorConfig,
    run,
)


def _build_corpus_into(tmp_path: Path) -> Path:
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
def smoke_config(fixture_manifest: Path, tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        strategy="hybrid",
        embedder_name="mock",
        reranker_name="mock",
        with_rerank=False,
        llm_name="mock",
        nli_name="mock",
        prompt_template="citation_required.v1.md",
        use_fixture=True,
        top_k=5,
        entail_threshold=0.5,
        smoke=True,
        n_resamples=200,
        manifest_path=fixture_manifest,
        embed_cache_dir=tmp_path / "embed_cache",
        index_dir=tmp_path / "index",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
    )


def test_run_writes_per_case_and_aggregate(smoke_config: OrchestratorConfig) -> None:
    run(smoke_config)
    per_case_path = smoke_config.reports_dir / "per_case.json"
    aggregate_path = smoke_config.reports_dir / "aggregate.json"
    assert per_case_path.exists()
    assert aggregate_path.exists()
    payload = json.loads(aggregate_path.read_text())
    assert payload["n_cases"] > 0
    assert payload["n_refusal"] == 6
    assert "per_tag" in payload


def test_run_writes_two_figures(smoke_config: OrchestratorConfig) -> None:
    run(smoke_config)
    fig_dir = smoke_config.figures_dir
    assert (fig_dir / "citation_precision_by_tag.png").exists()
    assert (fig_dir / "hallucination_rate_by_tag.png").exists()


def test_per_case_payload_has_expected_keys(smoke_config: OrchestratorConfig) -> None:
    run(smoke_config)
    payload = json.loads((smoke_config.reports_dir / "per_case.json").read_text())
    assert "config" in payload
    assert "cases" in payload
    for case in payload["cases"]:
        assert {
            "id",
            "tags",
            "should_refuse",
            "refused",
            "verified_text",
            "raw_llm_text",
            "verified_claims",
            "suppressed_claims",
            "retrieved_chunk_ids",
            "metrics",
        } <= case.keys()


def test_aggregate_records_config(smoke_config: OrchestratorConfig) -> None:
    run(smoke_config)
    payload = json.loads((smoke_config.reports_dir / "aggregate.json").read_text())
    cfg = payload["config"]
    assert cfg["llm"] == "mock"
    assert cfg["nli"] == "mock"
    assert cfg["use_fixture"] is True


def test_smoke_is_deterministic(smoke_config: OrchestratorConfig) -> None:
    a = run(smoke_config)
    b = run(smoke_config)
    assert a["citation_precision"] == b["citation_precision"]
    assert a["keyword_recall"] == b["keyword_recall"]
    assert a["refusal_accuracy"] == b["refusal_accuracy"]


def test_cli_smoke_runs_via_subprocess(tmp_path: Path) -> None:
    manifest_path = _build_corpus_into(tmp_path)
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend" / "scripts" / "eval_generation.py"),
        "--smoke",
        "--llm",
        "mock",
        "--nli",
        "mock",
        "--embedder",
        "mock",
        "--n-resamples",
        "100",
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
    assert (reports_dir / "per_case.json").exists()
    assert (reports_dir / "aggregate.json").exists()
    assert proc.returncode == 0
