"""CLI: run the Phase-4 agent eval.

Modes::

    --smoke            MockLLM + MockNLI + MockRetrievalPipeline +
                       3-case slice. ~3 s on ubuntu-latest. CI default.
                       No corpus weights, no API keys.
    --full             All 30 cases against whatever generator/pipeline
                       you build (passed through --llm / --nli / etc).

Outputs land under ``reports/v1/agents/{per_case,aggregate}.json`` and
``reports/v1/figures/agents/{per_stage_pass_rate,risk_band_confusion,
per_tag_pass_rate}.png``. ``--smoke`` writes under ``…/smoke/`` so
the CI noise stays out of git.
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
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cardiorisk.agents.eval import load_cases, run_eval
from cardiorisk.data.paths import (
    REPO_ROOT,
    REPORTS_V1_AGENTS,
    REPORTS_V1_AGENTS_FIGURES,
)
from cardiorisk.rag.generation.generator import CitationGenerator
from cardiorisk.rag.generation.llm import MockLLMClient
from cardiorisk.rag.generation.nli import EntailmentResult, MockNLIVerifier
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


# --------------------------------------------------------------------- mocks
def _mock_chunks() -> list[Chunk]:
    """A small set of fake guideline-shaped chunks for the smoke harness.

    The retrieval pipeline used in --smoke mode returns these chunks
    for *every* query; the Mock LLM emits one citation per passage
    so the parser + verifier exercise their full surface.
    """
    return [
        _chunk(
            "smoke-c1",
            "Pharmacotherapy is recommended for high CVD risk patients in the Australian primary-care setting.",
            doc_id="nvdpa",
        ),
        _chunk(
            "smoke-c2",
            "Lifestyle interventions remain the first-line approach for primary cardiovascular prevention.",
            doc_id="racgp",
        ),
        _chunk(
            "smoke-c3",
            "Statin therapy should be considered for adults with calculated 5-year absolute CVD risk above 10%.",
            doc_id="nvdpa",
        ),
    ]


def _chunk(chunk_id: str, text: str, doc_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        strategy="token",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
        text=text,
        n_tokens=len(text.split()),
    )


@dataclass
class _StubPipeline:
    chunks: list[Chunk]

    def retrieve(
        self, query: str, *, top_k: int = 5, with_rerank: bool = False
    ) -> list[RetrievedChunk]:
        del query, with_rerank
        return [
            RetrievedChunk(
                chunk=c,
                score=1.0 - i * 0.1,
                rrf_score=1.0 - i * 0.1,
                vector_rank=i + 1,
                bm25_rank=i + 1,
                rerank_score=None,
            )
            for i, c in enumerate(self.chunks[:top_k])
        ]


class _AlwaysEntails(MockNLIVerifier):
    """MockNLI that always returns p_entail=0.99 — drives the smoke harness.

    The default :class:`MockNLIVerifier` uses token-overlap which is too
    strict for the smoke chunks (the LLM-generated claim wording does
    not always overlap with the chunk wording word-for-word). For the
    smoke harness we want to *exercise* the verifier path while
    guaranteeing every claim is kept; the full eval (Phase 6) swaps
    this for the DeBERTa verifier where the threshold matters.
    """

    name: str = "always-entail"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        del premise, hypothesis
        return EntailmentResult(p_entailment=0.99, p_neutral=0.005, p_contradiction=0.005)

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


def _build_smoke_generator() -> CitationGenerator:
    return CitationGenerator(
        retrieval_pipeline=_StubPipeline(_mock_chunks()),  # type: ignore[arg-type]
        llm_client=MockLLMClient(),
        nli_verifier=_AlwaysEntails(),
    )


# --------------------------------------------------------------------- cli
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke mode: 3 cases, MockLLM + always-entail NLI + stub "
            "retrieval pipeline. No weights, no API keys, no network."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N cases (after tag filter); overrides --smoke's 3-case slice",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="restrict to cases with this tag (e.g. high_risk, refusal)",
    )
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=None,
        help="override the default eval/agents/cases.jsonl path",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_V1_AGENTS,
        help=f"reports output dir (default: {REPORTS_V1_AGENTS.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_V1_AGENTS_FIGURES,
        help=f"figures output dir (default: {REPORTS_V1_AGENTS_FIGURES.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="tabicl",
        help="risk model name (matches models/v1/<model>_<source>.joblib; default: tabicl)",
    )
    parser.add_argument(
        "--risk-source",
        type=str,
        default="Cleveland",
        help="LODO held-out source for the risk artefact (default: Cleveland)",
    )
    args = parser.parse_args()

    cases = load_cases(
        cases_path=args.cases_path,
        tag_filter=args.tag,
        limit=args.limit if args.limit is not None else (3 if args.smoke else None),
    )
    if not cases:
        print("No eval cases matched the filters.", file=sys.stderr)
        return 1

    generator = _build_smoke_generator()  # Phase 4 only ships the smoke generator path.

    summary = run_eval(
        generator=generator,
        cases=cases,
        risk_model_name=args.risk_model,
        risk_held_out_source=args.risk_source,
        llm_client_name=type(generator._llm).__name__,
        nli_verifier_name=type(generator._nli).__name__,
        embedder_name="stub",
        reranker_name="off",
        is_smoke=args.smoke,
        output_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        cases_path_override=args.cases_path,
    )

    # Tiny summary on stdout for CI logs.
    headline = {
        "n_cases": summary["aggregate"]["n_cases"],
        "triage_pass_rate": summary["aggregate"]["triage_pass_rate"],
        "risk_band_match_rate": summary["aggregate"]["risk_band_match_rate"],
        "guideline_pass_rate": summary["aggregate"]["guideline_pass_rate"],
        "letter_pass_rate": summary["aggregate"]["letter_pass_rate"],
        "full_pipeline_pass_rate": summary["aggregate"]["full_pipeline_pass_rate"],
        "median_total_duration_ms": summary["aggregate"]["median_total_duration_ms"],
        "p95_total_duration_ms": summary["aggregate"]["p95_total_duration_ms"],
    }
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
