"""CLI: build vector + BM25 indices from the Phase-3.1 manifest.

For each chunker strategy in the manifest, encodes all chunks with
the chosen embedder and writes a vector + BM25 index pair under
``data/external/corpus/index/<embedder_name>/<strategy>/``.

The eval CLI (``eval_retrieval.py``) re-runs the index build by
default (cheap at Phase-3.1 corpus size), but a maintainer who wants
to rebuild indices once and then sweep the eval matrix multiple
times can call this CLI directly.

Usage::

    uv run python backend/scripts/build_index.py
    uv run python backend/scripts/build_index.py --use-fixture --embedder minilm
    uv run python backend/scripts/build_index.py --strategy hybrid --embedder bge-m3
"""

from __future__ import annotations

import os

# OpenMP-guard preamble (mirrors compute_explanations.py / compute_drift.py).
# Must run BEFORE any torch / sentence-transformers / FlagEmbedding import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass

import argparse
import sys
from pathlib import Path

from cardiorisk.data.paths import (
    CORPUS_EMBED_CACHE,
    CORPUS_INDEX,
    CORPUS_MANIFEST,
    REPO_ROOT,
)
from cardiorisk.rag.eval_retrieval.orchestrator import (
    DEFAULT_STRATEGIES,
    _build_indices_for_strategy,
)
from cardiorisk.rag.ingest.chunkers import load_chunks
from cardiorisk.rag.ingest.manifest import load_manifest, resolve_chunks_path
from cardiorisk.rag.retrieval.embed import EmbedCache, get_embedder


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        choices=[*DEFAULT_STRATEGIES, "all"],
        default="all",
        help="build indices only for this chunker (default: all in manifest)",
    )
    parser.add_argument(
        "--embedder",
        choices=["bge-m3", "minilm", "mock"],
        default="bge-m3",
        help="embedder to use (default: bge-m3 for full local run)",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help=(
            "ignored here for parity with build_corpus.py; the manifest already "
            "reflects whatever build_corpus.py produced. Pass-through only."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CORPUS_MANIFEST,
        help=f"path to manifest.json (default: {CORPUS_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=CORPUS_INDEX,
        help=f"output dir for indices (default: {CORPUS_INDEX.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--embed-cache-dir",
        type=Path,
        default=CORPUS_EMBED_CACHE,
        help=f"embedding cache root (default: {CORPUS_EMBED_CACHE.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if not args.manifest_path.exists():
        print(
            f"ERROR: manifest not found at {args.manifest_path}; run "
            "backend/scripts/build_corpus.py first",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(args.manifest_path)
    if args.strategy == "all":
        strategies = sorted(manifest.chunks_by_strategy)
    else:
        if args.strategy not in manifest.chunks_by_strategy:
            print(
                f"ERROR: strategy {args.strategy!r} not in manifest; known: "
                f"{sorted(manifest.chunks_by_strategy)}",
                file=sys.stderr,
            )
            return 2
        strategies = [args.strategy]

    embedder = get_embedder(args.embedder)
    print(f"Embedder: {embedder.name} (dim={embedder.dim})")
    cache = EmbedCache(args.embed_cache_dir, embedder)

    for strategy in strategies:
        chunks_path = resolve_chunks_path(manifest, strategy)
        chunks = load_chunks(chunks_path)
        vec_idx, bm25_idx = _build_indices_for_strategy(
            strategy=strategy,
            chunks=chunks,
            embedder=embedder,
            embed_cache=cache,
            index_root=args.index_dir,
        )
        print(
            f"  [indexed ] strategy={strategy:<10} "
            f"vec={len(vec_idx)} bm25={len(bm25_idx)} "
            f"-> {(args.index_dir / embedder.name / strategy).relative_to(REPO_ROOT)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
