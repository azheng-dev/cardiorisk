"""End-to-end Phase 3.2 retrieval-eval driver.

Reads ``data/external/corpus/manifest.json`` (built by Phase 3.1's
``backend/scripts/build_corpus.py``), loads the per-strategy chunks,
loads or builds the vector + BM25 indices, runs the
``{chunker x {with_rerank, no_rerank}}`` eval matrix, and writes:

- ``reports/v1/retrieval/per_cell.json``
- ``reports/v1/retrieval/aggregate.json``
- ``reports/v1/figures/retrieval/{hit_at_5_by_cell,mrr_by_cell,per_tag_winning_cell}.png``

The orchestrator is idempotent over the embedding cache; rerunning
with the same chunks + same embedder produces byte-identical
embeddings (and therefore byte-identical hit@k metrics, modulo
hnswlib's randomised graph construction which uses our pinned RNG
seed via ``np.random.default_rng``).
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
    REPORTS_V1_RETRIEVAL,
    REPORTS_V1_RETRIEVAL_FIGURES,
)

from ..ingest.chunkers import Chunk, load_chunks
from ..ingest.manifest import load_manifest, resolve_chunks_path
from ..retrieval.bm25 import BM25Index
from ..retrieval.embed import BaseEmbedder, EmbedCache, get_embedder
from ..retrieval.index import HNSWIndex
from ..retrieval.pipeline import RetrievalPipeline
from ..retrieval.rerank import BaseReranker, get_reranker
from .figures import (
    CellResult,
    render_hit_at_5_by_cell,
    render_mrr_by_cell,
    render_per_tag_for_winning_cell,
)
from .loader import EvalQuestion, load_questions
from .scorer import (
    EvalReport,
    QuestionResult,
    aggregate_scores,
    score_question,
)

DEFAULT_STRATEGIES: Final[tuple[str, ...]] = ("token", "semantic", "hybrid")
DEFAULT_PER_LEG_K: Final[int] = 50
DEFAULT_TOP_K_FINAL: Final[int] = 5

# Smoke configuration: 1 chunker x {no-rerank} only, fixture Qs only,
# MiniLM embedder. ~60 s on ubuntu-latest.
SMOKE_STRATEGIES: Final[tuple[str, ...]] = ("hybrid",)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Knobs the CLI maps to."""

    strategies: tuple[str, ...]
    rerank_conditions: tuple[bool, ...]
    embedder_name: str
    reranker_name: str
    use_fixture: bool
    top_k: int
    per_leg_k: int
    smoke: bool
    n_resamples: int
    manifest_path: Path
    embed_cache_dir: Path
    index_dir: Path
    reports_dir: Path
    figures_dir: Path


def default_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        strategies=DEFAULT_STRATEGIES,
        rerank_conditions=(False, True),
        embedder_name="bge-m3",
        reranker_name="bge-reranker-v2-m3",
        use_fixture=False,
        top_k=DEFAULT_TOP_K_FINAL,
        per_leg_k=DEFAULT_PER_LEG_K,
        smoke=False,
        n_resamples=2000,
        manifest_path=CORPUS_MANIFEST,
        embed_cache_dir=CORPUS_EMBED_CACHE,
        index_dir=CORPUS_INDEX,
        reports_dir=REPORTS_V1_RETRIEVAL,
        figures_dir=REPORTS_V1_RETRIEVAL_FIGURES,
    )


def smoke_config() -> OrchestratorConfig:
    base = default_config()
    return OrchestratorConfig(
        strategies=SMOKE_STRATEGIES,
        rerank_conditions=(False,),
        embedder_name="minilm",
        reranker_name="mock",
        use_fixture=True,
        top_k=DEFAULT_TOP_K_FINAL,
        per_leg_k=DEFAULT_PER_LEG_K,
        smoke=True,
        n_resamples=500,
        manifest_path=base.manifest_path,
        embed_cache_dir=base.embed_cache_dir,
        index_dir=base.index_dir,
        reports_dir=base.reports_dir / "smoke",
        figures_dir=base.figures_dir / "smoke",
    )


def _to_json_safe(value: Any) -> Any:
    """Recursively replace NaN / inf with JSON ``null`` (mirrors Phase 2.x)."""
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
        "n_questions": report.n_questions,
        "hit_at_1": report.hit_at_1,
        "hit_at_5": report.hit_at_5,
        "mrr": report.mrr,
        "ci_hit_at_1": _ci_dict(report.ci_hit_at_1),
        "ci_hit_at_5": _ci_dict(report.ci_hit_at_5),
        "ci_mrr": _ci_dict(report.ci_mrr),
        "per_tag": report.per_tag,
    }


def _build_indices_for_strategy(
    *,
    strategy: str,
    chunks: list[Chunk],
    embedder: BaseEmbedder,
    embed_cache: EmbedCache,
    index_root: Path,
) -> tuple[HNSWIndex, BM25Index]:
    """Build (or load) vector + BM25 indices for one chunker strategy.

    Indices are persisted under ``index_root/<embedder_name>/<strategy>/``
    so different embedders don't collide. We always re-build (cheap at
    Phase-3.1 corpus size; ensures the chunker contents and the index
    stay in sync without a manual cache-bust).
    """
    out_dir = index_root / embedder.name / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    if not chunks:
        return HNSWIndex(dim=embedder.dim), BM25Index()
    vectors = embed_cache.encode(chunk_ids, texts)
    vec_idx = HNSWIndex(dim=embedder.dim)
    vec_idx.build(chunk_ids=chunk_ids, vectors=vectors)
    vec_idx.save(out_dir)
    bm25_idx = BM25Index()
    bm25_idx.build(chunk_ids=chunk_ids, texts=texts)
    bm25_idx.save(out_dir)
    return vec_idx, bm25_idx


def _run_cell(
    *,
    chunker_name: str,
    with_rerank: bool,
    pipeline: RetrievalPipeline,
    questions: list[EvalQuestion],
    top_k: int,
    n_resamples: int,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[EvalReport, list[QuestionResult]]:
    """Score one ``{chunker x rerank}`` cell.

    Prints a per-question heartbeat every ``_LOG_EVERY`` questions so a
    long-running rerank cell (cross-encoder over long real-corpus
    passages can be 5-10 s per question) doesn't look hung.
    """
    per_q: list[QuestionResult] = []
    for i, q in enumerate(questions, start=1):
        retrieved = pipeline.retrieve(q.question, top_k=top_k, with_rerank=with_rerank)
        per_q.append(score_question(q, retrieved, top_k=top_k))
        if i == 1 or i % _LOG_EVERY == 0 or i == len(questions):
            log(f"             [{i:>3d}/{len(questions)}] {q.id} done")
    report = aggregate_scores(per_q, n_resamples=n_resamples)
    return report, per_q


_LOG_EVERY: Final[int] = 5


def _winning_cell(cells: list[CellResult]) -> CellResult:
    """Pick the cell with the highest hit@5 point estimate.

    Ties broken first by MRR, then by lower complexity (no-rerank
    over with-rerank), then alphabetically by chunker.
    """
    return max(
        cells,
        key=lambda c: (
            c.report.hit_at_5,
            c.report.mrr,
            -int(c.with_rerank),
            -ord(c.chunker[0]),
        ),
    )


def _materialise_chunks_by_id(
    strategy_chunks: dict[str, list[Chunk]],
) -> dict[str, dict[str, Chunk]]:
    """Per-strategy ``{chunk_id: Chunk}`` lookup for the pipeline."""
    return {
        strategy: {c.chunk_id: c for c in chunks} for strategy, chunks in strategy_chunks.items()
    }


def run(config: OrchestratorConfig, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the full eval matrix.

    Args:
        config: Knob set; built by :func:`default_config` /
            :func:`smoke_config` or by the CLI.
        log: ``print``-shaped callable used by the CLI for status.

    Returns:
        The aggregate-report dict that was written to
        ``aggregate.json``.
    """
    if not config.manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {config.manifest_path}; run "
            "backend/scripts/build_corpus.py first"
        )

    manifest = load_manifest(config.manifest_path)
    log(
        f"Loaded manifest: {len(manifest.parsed_docs)} parsed_docs, "
        f"{len(manifest.chunks_by_strategy)} strategies"
    )

    # In ``use_fixture=True`` mode the orchestrator runs against the
    # markdown fixture corpus; the 10 ``requires_full_corpus`` Qs would
    # never hit, so drop them. In ``use_fixture=False`` mode the
    # orchestrator runs against the real RACGP / NVDPA corpus; the 40
    # fixture Qs reference ``fixture_*`` doc_ids that don't exist in the
    # real corpus, so drop them too. Without this filter the
    # full-corpus headline hit@5 caps at the real-corpus Q fraction
    # (10/50 = 0.20) regardless of how good retrieval actually is.
    questions = load_questions(
        skip_full_corpus=config.use_fixture,
        skip_fixture=not config.use_fixture,
    )
    log(
        f"Loaded {len(questions)} eval questions "
        f"({'fixture-only' if config.use_fixture else 'real-corpus-only'})"
    )

    embedder = get_embedder(config.embedder_name)
    embed_cache = EmbedCache(config.embed_cache_dir, embedder)
    log(f"Embedder: {embedder.name} (dim={embedder.dim})")

    reranker: BaseReranker | None = None
    if any(config.rerank_conditions):
        reranker = get_reranker(config.reranker_name)
        log(f"Reranker: {reranker.name}")

    strategy_chunks: dict[str, list[Chunk]] = {}
    for strategy in config.strategies:
        if strategy not in manifest.chunks_by_strategy:
            log(f"  WARN: strategy {strategy!r} not in manifest; skipping")
            continue
        chunks_path = resolve_chunks_path(manifest, strategy)
        chunks = load_chunks(chunks_path)
        strategy_chunks[strategy] = chunks
        log(f"  [chunks  ] strategy={strategy:<10} n={len(chunks)}")

    chunks_by_id_by_strategy = _materialise_chunks_by_id(strategy_chunks)

    indices_by_strategy: dict[str, tuple[HNSWIndex, BM25Index]] = {}
    for strategy, chunks in strategy_chunks.items():
        vec_idx, bm25_idx = _build_indices_for_strategy(
            strategy=strategy,
            chunks=chunks,
            embedder=embedder,
            embed_cache=embed_cache,
            index_root=config.index_dir,
        )
        indices_by_strategy[strategy] = (vec_idx, bm25_idx)
        log(f"  [indexed ] strategy={strategy:<10} vec={len(vec_idx)} bm25={len(bm25_idx)}")

    cells: list[CellResult] = []
    per_q_by_cell: dict[str, list[QuestionResult]] = {}
    for strategy in strategy_chunks:
        vec_idx, bm25_idx = indices_by_strategy[strategy]
        for with_rerank in config.rerank_conditions:
            pipeline = RetrievalPipeline(
                embedder=embedder,
                embed_cache=embed_cache,
                vector_index=vec_idx,
                bm25_index=bm25_idx,
                chunks_by_id=chunks_by_id_by_strategy[strategy],
                reranker=reranker if with_rerank else None,
                per_leg_k=config.per_leg_k,
            )
            label = f"{strategy}_{'rerank' if with_rerank else 'norerank'}"
            log(f"  [cell    ] {label}: scoring {len(questions)} questions...")
            report, per_q = _run_cell(
                chunker_name=strategy,
                with_rerank=with_rerank,
                pipeline=pipeline,
                questions=questions,
                top_k=config.top_k,
                n_resamples=config.n_resamples,
                log=log,
            )
            cells.append(
                CellResult(
                    cell_label=label, chunker=strategy, with_rerank=with_rerank, report=report
                )
            )
            per_q_by_cell[label] = per_q
            log(
                f"             hit@1={report.hit_at_1:.3f} "
                f"hit@5={report.hit_at_5:.3f} MRR={report.mrr:.3f} "
                f"(95% CI hit@5: [{report.ci_hit_at_5.lower:.3f}, {report.ci_hit_at_5.upper:.3f}])"
            )

    if not cells:
        raise RuntimeError("no eval cells produced; check manifest + strategies")

    per_cell_payload = {
        "config": _config_dict(config),
        "cells": [
            {
                "label": c.cell_label,
                "chunker": c.chunker,
                "with_rerank": c.with_rerank,
                "embedder": embedder.name,
                "reranker": reranker.name if (reranker and c.with_rerank) else None,
                **_report_to_dict(c.report),
            }
            for c in cells
        ],
    }

    winner = _winning_cell(cells)
    aggregate_payload = {
        "config": _config_dict(config),
        "n_cells": len(cells),
        "winning_cell": {
            "label": winner.cell_label,
            "chunker": winner.chunker,
            "with_rerank": winner.with_rerank,
            "hit_at_1": winner.report.hit_at_1,
            "hit_at_5": winner.report.hit_at_5,
            "mrr": winner.report.mrr,
            "ci_hit_at_5": _ci_dict(winner.report.ci_hit_at_5),
        },
        "per_chunker_max_hit_at_5": _per_chunker_max(cells),
        "rerank_lift": _rerank_lift(cells),
    }

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "per_cell.json").write_text(
        json.dumps(_to_json_safe(per_cell_payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (config.reports_dir / "aggregate.json").write_text(
        json.dumps(_to_json_safe(aggregate_payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log(f"  [report  ] -> {config.reports_dir}/per_cell.json + aggregate.json")

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    render_hit_at_5_by_cell(cells, out_path=config.figures_dir / "hit_at_5_by_cell.png")
    render_mrr_by_cell(cells, out_path=config.figures_dir / "mrr_by_cell.png")
    render_per_tag_for_winning_cell(
        winner, out_path=config.figures_dir / "per_tag_winning_cell.png"
    )
    log(f"  [figures ] -> {config.figures_dir}/")

    return aggregate_payload


def _config_dict(config: OrchestratorConfig) -> dict[str, Any]:
    return {
        "strategies": list(config.strategies),
        "rerank_conditions": list(config.rerank_conditions),
        "embedder": config.embedder_name,
        "reranker": config.reranker_name,
        "use_fixture": config.use_fixture,
        "top_k": config.top_k,
        "per_leg_k": config.per_leg_k,
        "smoke": config.smoke,
        "n_resamples": config.n_resamples,
    }


def _per_chunker_max(cells: list[CellResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cells:
        out[c.chunker] = max(out.get(c.chunker, float("-inf")), c.report.hit_at_5)
    return out


def _rerank_lift(cells: list[CellResult]) -> dict[str, float | None]:
    """Per-chunker hit@5 difference: with_rerank - no_rerank.

    Returns ``None`` for chunkers where one of the two conditions
    is missing (e.g. smoke runs which only do no-rerank).
    """
    by_pair: dict[str, dict[bool, float]] = {}
    for c in cells:
        by_pair.setdefault(c.chunker, {})[c.with_rerank] = c.report.hit_at_5
    out: dict[str, float | None] = {}
    for chunker, conds in by_pair.items():
        if True in conds and False in conds:
            out[chunker] = conds[True] - conds[False]
        else:
            out[chunker] = None
    return out
