"""CLI: run the Phase-3.2 retrieval eval matrix.

Reads ``data/external/corpus/manifest.json``, builds (or rebuilds)
the per-strategy indices, scores every cell in the
``{chunker x {with_rerank, no_rerank}}`` matrix, and writes
``reports/v1/retrieval/{per_cell,aggregate}.json`` + figures.

Modes::

    --smoke            1 chunker x {no-rerank}, MiniLM embedder,
                       fixture Qs only. ~60 s on ubuntu-latest.
    --use-fixture      skip questions with requires_full_corpus=true.
    --rerank both      run with-rerank AND no-rerank cells.
    --rerank on        only run with-rerank.
    --rerank off       only run no-rerank.
    --embedder X       one of {bge-m3, minilm, mock}.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    # The cross-encoder reranker (BGE-reranker-v2-m3, 568 M params) is the
    # wall-clock bottleneck at ``--rerank on``: with the default single-
    # thread guard the 50-Q matrix takes hours on an M4 Pro. Honour
    # ``CARDIORISK_TORCH_THREADS`` if set, otherwise fall back to the
    # safe single-threaded default that the rest of Phase 2.x relies on
    # to avoid the TabICL/XGBoost/PyTorch OpenMP deadlock on macOS.
    # This script never imports TabICL or XGBoost so lifting the cap
    # locally is safe.
    _torch_threads_env = os.environ.get("CARDIORISK_TORCH_THREADS")
    if _torch_threads_env:
        torch.set_num_threads(max(1, int(_torch_threads_env)))
    else:
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
    REPORTS_V1_RETRIEVAL,
    REPORTS_V1_RETRIEVAL_FIGURES,
)
from cardiorisk.rag.eval_retrieval.orchestrator import (
    DEFAULT_STRATEGIES,
    OrchestratorConfig,
    default_config,
    run,
    smoke_config,
)


def _rerank_conditions(arg: str) -> tuple[bool, ...]:
    if arg == "both":
        return (False, True)
    if arg == "on":
        return (True,)
    if arg == "off":
        return (False,)
    raise ValueError(f"unknown --rerank value {arg!r}; expected one of: both, on, off")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke mode: 1 chunker x {no-rerank}, MiniLM embedder, "
            "fixture Qs only, 500-resample bootstrap. CI default."
        ),
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="skip questions with requires_full_corpus=true",
    )
    parser.add_argument(
        "--rerank",
        choices=["both", "on", "off"],
        default="both",
        help="which rerank conditions to run (default: both)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="chunker strategies to evaluate (default: all three)",
    )
    parser.add_argument(
        "--embedder",
        choices=["bge-m3", "minilm", "mock"],
        default="bge-m3",
        help="embedder (default: bge-m3 for full local run)",
    )
    parser.add_argument(
        "--reranker",
        choices=["bge-reranker-v2-m3", "mock"],
        default="bge-reranker-v2-m3",
        help="reranker (default: bge-reranker-v2-m3 for full local run)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="final top-k after fusion + rerank (default: 5)",
    )
    parser.add_argument(
        "--per-leg-k",
        type=int,
        default=50,
        help="per-leg fan-out before RRF (default: 50)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="bootstrap resample count (default: 2000)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=CORPUS_MANIFEST,
        help=f"path to manifest.json (default: {CORPUS_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_V1_RETRIEVAL,
        help=f"reports output dir (default: {REPORTS_V1_RETRIEVAL.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_V1_RETRIEVAL_FIGURES,
        help=f"figures output dir (default: {REPORTS_V1_RETRIEVAL_FIGURES.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if args.smoke:
        config = smoke_config()
        # Override smoke defaults from CLI flags where given (preserve
        # smoke=True; allow embedder override for ad-hoc maintainer runs).
        config = OrchestratorConfig(
            strategies=config.strategies,
            rerank_conditions=config.rerank_conditions,
            embedder_name=args.embedder if args.embedder != "bge-m3" else config.embedder_name,
            reranker_name=config.reranker_name,
            use_fixture=True,
            top_k=config.top_k,
            per_leg_k=config.per_leg_k,
            smoke=True,
            n_resamples=args.n_resamples if args.n_resamples != 2000 else config.n_resamples,
            manifest_path=args.manifest_path,
            embed_cache_dir=CORPUS_EMBED_CACHE,
            index_dir=CORPUS_INDEX,
            reports_dir=config.reports_dir,
            figures_dir=config.figures_dir,
        )
    else:
        base = default_config()
        config = OrchestratorConfig(
            strategies=tuple(args.strategies),
            rerank_conditions=_rerank_conditions(args.rerank),
            embedder_name=args.embedder,
            reranker_name=args.reranker,
            use_fixture=args.use_fixture,
            top_k=args.top_k,
            per_leg_k=args.per_leg_k,
            smoke=False,
            n_resamples=args.n_resamples,
            manifest_path=args.manifest_path,
            embed_cache_dir=base.embed_cache_dir,
            index_dir=base.index_dir,
            reports_dir=args.reports_dir,
            figures_dir=args.figures_dir,
        )

    try:
        run(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
