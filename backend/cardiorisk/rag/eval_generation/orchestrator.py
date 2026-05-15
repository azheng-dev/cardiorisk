"""End-to-end Phase 3.3 generation-eval driver.

Builds the retrieval pipeline (Phase 3.2 winning cell: token chunker,
no rerank — overridable from CLI), the LLM client, and the NLI
verifier; runs every case in ``eval/generation/cases.jsonl``; writes:

- ``reports/v1/generation/per_case.json``
- ``reports/v1/generation/aggregate.json``
- ``reports/v1/figures/generation/citation_precision_by_tag.png``
- ``reports/v1/figures/generation/hallucination_rate_by_tag.png``
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cardiorisk.data.paths import (
    CORPUS_EMBED_CACHE,
    CORPUS_INDEX,
    CORPUS_MANIFEST,
    REPORTS_V1_GENERATION,
    REPORTS_V1_GENERATION_FIGURES,
)

from ..eval_retrieval.orchestrator import _build_indices_for_strategy
from ..generation.generator import CitationGenerator, GeneratedAnswer
from ..generation.llm import BaseLLMClient, get_llm_client
from ..generation.nli import (
    DEFAULT_ENTAILMENT_THRESHOLD,
    BaseNLIVerifier,
    get_nli_verifier,
)
from ..generation.prompts import DEFAULT_PROMPT
from ..ingest.chunkers import Chunk, load_chunks
from ..ingest.manifest import load_manifest, resolve_chunks_path
from ..retrieval.embed import EmbedCache, get_embedder
from ..retrieval.pipeline import RetrievalPipeline
from ..retrieval.rerank import BaseReranker, get_reranker
from .figures import render_citation_precision_by_tag, render_hallucination_rate_by_tag
from .loader import EvalCase, load_cases
from .scorer import CaseResult, EvalReport, aggregate_scores, score_case

#: Default Phase 3.2 winning cell (token chunker, no rerank).
DEFAULT_STRATEGY: Final[str] = "token"
DEFAULT_TOP_K: Final[int] = 5

_LOG_EVERY: Final[int] = 5


@dataclass(frozen=True)
class OrchestratorConfig:
    """Knobs the CLI maps to."""

    strategy: str
    embedder_name: str
    reranker_name: str
    with_rerank: bool
    llm_name: str
    nli_name: str
    prompt_template: str
    use_fixture: bool
    top_k: int
    entail_threshold: float
    smoke: bool
    n_resamples: int
    manifest_path: Path
    embed_cache_dir: Path
    index_dir: Path
    reports_dir: Path
    figures_dir: Path


def default_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        strategy=DEFAULT_STRATEGY,
        embedder_name="bge-m3",
        reranker_name="bge-reranker-v2-m3",
        with_rerank=False,
        llm_name="mock",
        nli_name="deberta",
        prompt_template=DEFAULT_PROMPT,
        use_fixture=False,
        top_k=DEFAULT_TOP_K,
        entail_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
        smoke=False,
        n_resamples=2000,
        manifest_path=CORPUS_MANIFEST,
        embed_cache_dir=CORPUS_EMBED_CACHE,
        index_dir=CORPUS_INDEX,
        reports_dir=REPORTS_V1_GENERATION,
        figures_dir=REPORTS_V1_GENERATION_FIGURES,
    )


def smoke_config() -> OrchestratorConfig:
    base = default_config()
    return OrchestratorConfig(
        strategy=DEFAULT_STRATEGY,
        embedder_name="minilm",
        reranker_name="mock",
        with_rerank=False,
        llm_name="mock",
        nli_name="mock",
        prompt_template=DEFAULT_PROMPT,
        use_fixture=True,
        top_k=DEFAULT_TOP_K,
        entail_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
        smoke=True,
        n_resamples=500,
        manifest_path=base.manifest_path,
        embed_cache_dir=base.embed_cache_dir,
        index_dir=base.index_dir,
        reports_dir=base.reports_dir / "smoke",
        figures_dir=base.figures_dir / "smoke",
    )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def _ci_dict(ci: Any) -> dict[str, Any]:
    return {
        "point": ci.point,
        "lower": ci.lower,
        "upper": ci.upper,
        "n_resamples": ci.n_resamples,
        "alpha": ci.alpha,
    }


def _report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "n_cases": report.n_cases,
        "n_positive": report.n_positive,
        "n_refusal": report.n_refusal,
        "citation_precision": report.citation_precision,
        "keyword_recall": report.keyword_recall,
        "hallucination_rate": report.hallucination_rate,
        "refusal_accuracy": report.refusal_accuracy,
        "ci_keyword_recall": _ci_dict(report.ci_keyword_recall),
        "ci_hallucination_rate": _ci_dict(report.ci_hallucination_rate),
        "ci_refusal_accuracy": _ci_dict(report.ci_refusal_accuracy),
        "per_tag": report.per_tag,
    }


def _per_case_payload(
    case: EvalCase, answer: GeneratedAnswer, result: CaseResult
) -> dict[str, Any]:
    return {
        "id": case.id,
        "tags": list(case.tags),
        "should_refuse": case.should_refuse,
        "refused": result.refused,
        "verified_text": answer.verified_text,
        "raw_llm_text": answer.raw_llm_text,
        "verified_claims": [
            {
                "text": c.text,
                "headline_chunk_id": c.headline_chunk_id,
                "headline_score": c.headline_score,
                "supporting_chunk_ids": list(c.supporting_chunk_ids),
                "supporting_scores": list(c.supporting_scores),
            }
            for c in answer.verified_claims
        ],
        "suppressed_claims": [
            {
                "text": s.text,
                "cited_chunk_ids": list(s.cited_chunk_ids),
                "best_entailment": s.best_entailment,
                "reason": s.reason,
            }
            for s in answer.suppressed_claims
        ],
        "retrieved_chunk_ids": [r.chunk.chunk_id for r in answer.retrieved],
        "metrics": {
            "keyword_recall": result.keyword_recall,
            "citation_precision": result.citation_precision,
            "hallucination": result.hallucination,
        },
    }


def _build_pipeline(
    *,
    config: OrchestratorConfig,
    log: Callable[[str], None],
) -> tuple[RetrievalPipeline, BaseLLMClient, BaseNLIVerifier]:
    if not config.manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {config.manifest_path}; run "
            "backend/scripts/build_corpus.py first"
        )
    manifest = load_manifest(config.manifest_path)
    if config.strategy not in manifest.chunks_by_strategy:
        raise RuntimeError(
            f"strategy {config.strategy!r} not in manifest ({sorted(manifest.chunks_by_strategy)})"
        )
    chunks_path = resolve_chunks_path(manifest, config.strategy)
    chunks: list[Chunk] = load_chunks(chunks_path)
    log(f"  [chunks  ] strategy={config.strategy} n={len(chunks)}")

    embedder = get_embedder(config.embedder_name)
    embed_cache = EmbedCache(config.embed_cache_dir, embedder)
    log(f"  [embedder] {embedder.name} (dim={embedder.dim})")

    reranker: BaseReranker | None = None
    if config.with_rerank:
        reranker = get_reranker(config.reranker_name)
        log(f"  [reranker] {reranker.name}")

    vec_idx, bm25_idx = _build_indices_for_strategy(
        strategy=config.strategy,
        chunks=chunks,
        embedder=embedder,
        embed_cache=embed_cache,
        index_root=config.index_dir,
    )
    log(f"  [indexed ] vec={len(vec_idx)} bm25={len(bm25_idx)}")
    pipeline = RetrievalPipeline(
        embedder=embedder,
        embed_cache=embed_cache,
        vector_index=vec_idx,
        bm25_index=bm25_idx,
        chunks_by_id={c.chunk_id: c for c in chunks},
        reranker=reranker,
    )

    llm_client = get_llm_client(config.llm_name)
    log(f"  [llm     ] {llm_client.name}")
    nli_verifier = get_nli_verifier(config.nli_name)
    log(f"  [nli     ] {nli_verifier.name}")

    return pipeline, llm_client, nli_verifier


def run(config: OrchestratorConfig, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the full generation eval over ``cases.jsonl``."""
    pipeline, llm_client, nli_verifier = _build_pipeline(config=config, log=log)
    cases = load_cases(
        skip_full_corpus=config.use_fixture,
        skip_fixture=not config.use_fixture,
    )
    log(
        f"Loaded {len(cases)} eval cases "
        f"({'fixture+refusal' if config.use_fixture else 'real-corpus+refusal'})"
    )

    generator = CitationGenerator(
        retrieval_pipeline=pipeline,
        llm_client=llm_client,
        nli_verifier=nli_verifier,
        prompt_template=config.prompt_template,
        prompt_top_k=config.top_k,
        with_rerank=config.with_rerank,
        entail_threshold=config.entail_threshold,
    )

    per_case_records: list[dict[str, Any]] = []
    case_results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        answer = generator.generate(case.question)
        result = score_case(case, answer)
        per_case_records.append(_per_case_payload(case, answer, result))
        case_results.append(result)
        if i == 1 or i % _LOG_EVERY == 0 or i == len(cases):
            log(
                f"  [case    ] {i:>3d}/{len(cases)} {case.id} "
                f"refused={result.refused} "
                f"recall={result.keyword_recall:.2f} "
                f"halluc={int(result.hallucination)}"
            )

    report = aggregate_scores(case_results, n_resamples=config.n_resamples)
    log(
        f"  [aggregate] cit_prec={report.citation_precision:.3f} "
        f"recall={report.keyword_recall:.3f} "
        f"halluc={report.hallucination_rate:.3f} "
        f"refusal_acc={report.refusal_accuracy:.3f}"
    )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    per_case_payload = {
        "config": _config_dict(config),
        "cases": per_case_records,
    }
    aggregate_payload = {
        "config": _config_dict(config),
        **_report_to_dict(report),
    }
    (config.reports_dir / "per_case.json").write_text(
        json.dumps(_to_json_safe(per_case_payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (config.reports_dir / "aggregate.json").write_text(
        json.dumps(_to_json_safe(aggregate_payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log(f"  [report   ] -> {config.reports_dir}/per_case.json + aggregate.json")

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    render_citation_precision_by_tag(
        report, out_path=config.figures_dir / "citation_precision_by_tag.png"
    )
    render_hallucination_rate_by_tag(
        report, out_path=config.figures_dir / "hallucination_rate_by_tag.png"
    )
    log(f"  [figures  ] -> {config.figures_dir}/")

    return aggregate_payload


def _config_dict(config: OrchestratorConfig) -> dict[str, Any]:
    return {
        "strategy": config.strategy,
        "embedder": config.embedder_name,
        "reranker": config.reranker_name,
        "with_rerank": config.with_rerank,
        "llm": config.llm_name,
        "nli": config.nli_name,
        "prompt_template": config.prompt_template,
        "use_fixture": config.use_fixture,
        "top_k": config.top_k,
        "entail_threshold": config.entail_threshold,
        "smoke": config.smoke,
        "n_resamples": config.n_resamples,
    }
