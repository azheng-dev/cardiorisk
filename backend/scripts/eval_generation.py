"""CLI: run the Phase-3.3 citation-generation eval.

Modes::

    --smoke            MockLLM + MockNLI + fixture-only cases. ~30 s
                       on ubuntu-latest. CI default.
    --use-fixture      restrict to cases whose expected_doc_ids are
                       fixture; refusal cases are kept under both
                       modes (they don't depend on any corpus).
    --llm X            one of {mock, anthropic, openai}.
    --nli X            one of {mock, deberta}.
    --strategy X       chunker; default 'token' (Phase 3.2 winner).
    --with-rerank      enable cross-encoder rerank stage.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

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
    REPORTS_V1_GENERATION,
    REPORTS_V1_GENERATION_FIGURES,
)
from cardiorisk.rag.eval_generation.orchestrator import (
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    OrchestratorConfig,
    default_config,
    run,
    smoke_config,
)
from cardiorisk.rag.generation.nli import DEFAULT_ENTAILMENT_THRESHOLD


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke mode: MockLLM + MockNLI + MiniLM embedder, "
            "fixture-only cases, 500-resample bootstrap. CI default."
        ),
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="restrict positive cases to fixture doc_ids (refusal cases always kept)",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "anthropic", "openai"],
        default="mock",
        help="LLM client (default: mock)",
    )
    parser.add_argument(
        "--nli",
        choices=["mock", "deberta"],
        default="mock",
        help="NLI verifier (default: mock; use 'deberta' for the real headline)",
    )
    parser.add_argument(
        "--strategy",
        choices=["token", "semantic", "hybrid"],
        default=DEFAULT_STRATEGY,
        help=f"chunker (default: {DEFAULT_STRATEGY}, Phase 3.2 winner)",
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
        help="reranker (default: bge-reranker-v2-m3, only used with --with-rerank)",
    )
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="enable cross-encoder rerank stage (default: off; Phase 3.2 winner is no-rerank)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"prompt top-k passages (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--entail-threshold",
        type=float,
        default=DEFAULT_ENTAILMENT_THRESHOLD,
        help=f"NLI entailment threshold (default: {DEFAULT_ENTAILMENT_THRESHOLD})",
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
        default=REPORTS_V1_GENERATION,
        help=f"reports output dir (default: {REPORTS_V1_GENERATION.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_V1_GENERATION_FIGURES,
        help=f"figures output dir (default: {REPORTS_V1_GENERATION_FIGURES.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke = smoke_config()
        # The smoke defaults under reports/smoke/, but the CLI also
        # respects --reports-dir / --figures-dir so a maintainer (or a
        # test) can redirect output without losing the smoke knobs.
        reports_dir = (
            args.reports_dir if args.reports_dir != REPORTS_V1_GENERATION else smoke.reports_dir
        )
        figures_dir = (
            args.figures_dir
            if args.figures_dir != REPORTS_V1_GENERATION_FIGURES
            else smoke.figures_dir
        )
        config = OrchestratorConfig(
            strategy=smoke.strategy,
            embedder_name=args.embedder if args.embedder != "bge-m3" else smoke.embedder_name,
            reranker_name=smoke.reranker_name,
            with_rerank=False,
            llm_name=args.llm if args.llm != "mock" else smoke.llm_name,
            nli_name=args.nli if args.nli != "mock" else smoke.nli_name,
            prompt_template=smoke.prompt_template,
            use_fixture=True,
            top_k=smoke.top_k,
            entail_threshold=smoke.entail_threshold,
            smoke=True,
            n_resamples=args.n_resamples if args.n_resamples != 2000 else smoke.n_resamples,
            manifest_path=args.manifest_path,
            embed_cache_dir=CORPUS_EMBED_CACHE,
            index_dir=CORPUS_INDEX,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
        )
    else:
        base = default_config()
        config = OrchestratorConfig(
            strategy=args.strategy,
            embedder_name=args.embedder,
            reranker_name=args.reranker,
            with_rerank=args.with_rerank,
            llm_name=args.llm,
            nli_name=args.nli,
            prompt_template=base.prompt_template,
            use_fixture=args.use_fixture,
            top_k=args.top_k,
            entail_threshold=args.entail_threshold,
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
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
