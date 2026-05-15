"""End-to-end :class:`CitationGenerator` tests using mock components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cardiorisk.rag.generation.generator import (
    CitationGenerator,
    GeneratedAnswer,
)
from cardiorisk.rag.generation.llm import LLMMessage, MockLLMClient
from cardiorisk.rag.generation.nli import (
    BaseNLIVerifier,
    EntailmentResult,
    MockNLIVerifier,
)
from cardiorisk.rag.ingest.chunkers import Chunk
from cardiorisk.rag.retrieval.pipeline import RetrievedChunk


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
    """Stand-in for :class:`RetrievalPipeline` that returns canned chunks."""

    chunks: list[Chunk]

    def retrieve(
        self, query: str, *, top_k: int = 5, with_rerank: bool = False
    ) -> list[RetrievedChunk]:
        del query, with_rerank
        out: list[RetrievedChunk] = []
        for i, c in enumerate(self.chunks[:top_k]):
            out.append(
                RetrievedChunk(
                    chunk=c,
                    score=1.0 - i * 0.1,
                    rrf_score=1.0 - i * 0.1,
                    vector_rank=i + 1,
                    bm25_rank=i + 1,
                    rerank_score=None,
                )
            )
        return out


class _AlwaysEntailingVerifier:
    """NLI stub that says every pair entails. Used to isolate the parser."""

    name = "always-entail"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        del premise, hypothesis
        return EntailmentResult(p_entailment=0.99, p_neutral=0.005, p_contradiction=0.005)

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


class _NeverEntailingVerifier:
    """NLI stub that says no pair entails. Used to test suppression."""

    name = "never-entail"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        del premise, hypothesis
        return EntailmentResult(p_entailment=0.05, p_neutral=0.9, p_contradiction=0.05)

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


def _generator(*, chunks: list[Chunk], verifier: BaseNLIVerifier) -> CitationGenerator:
    return CitationGenerator(
        retrieval_pipeline=_StubPipeline(chunks=chunks),  # type: ignore[arg-type]
        llm_client=MockLLMClient(max_passages=2),
        nli_verifier=verifier,
    )


def test_generator_returns_verified_claims_when_nli_entails() -> None:
    chunks = [
        _chunk("a:1", "Adults aged 45 to 79 should be assessed every two years."),
        _chunk("b:1", "For diabetes, begin assessment at age 35."),
    ]
    gen = _generator(chunks=chunks, verifier=_AlwaysEntailingVerifier())
    out: GeneratedAnswer = gen.generate("From what age?")
    assert not out.is_refusal
    assert len(out.verified_claims) == 2
    assert out.verified_claims[0].headline_chunk_id in {"a:1", "b:1"}
    assert out.verified_text


def test_generator_suppresses_when_nli_never_entails() -> None:
    chunks = [_chunk("a:1", "This passage will be cited.")]
    gen = _generator(chunks=chunks, verifier=_NeverEntailingVerifier())
    out = gen.generate("Q?")
    assert out.verified_claims == ()
    assert out.suppressed_claims
    assert out.suppressed_claims[0].reason == "no_passage_entails"


def test_generator_refuses_when_no_passages_retrieved() -> None:
    gen = _generator(chunks=[], verifier=_AlwaysEntailingVerifier())
    out = gen.generate("Q?")
    assert out.is_refusal
    assert "supporting guidance" in out.verified_text.lower()


def test_generator_uses_mock_nli_end_to_end() -> None:
    chunks = [
        _chunk(
            "a:1",
            "Aspirin is not routinely recommended for primary prevention "
            "because the bleeding risk usually offsets the cardiovascular benefit.",
        )
    ]
    gen = _generator(chunks=chunks, verifier=MockNLIVerifier())
    out = gen.generate("Is aspirin routinely recommended for primary CVD prevention?")
    # MockLLM emits one sentence containing the passage's first sentence,
    # which by construction has high token overlap, so MockNLI entails it.
    assert out.verified_claims
    assert out.verified_claims[0].headline_chunk_id == "a:1"


def test_generator_records_phantom_citation_under_suppressed() -> None:
    chunks = [_chunk("a:1", "Real passage.")]

    class _PhantomLLM:
        name = "phantom"

        def generate(self, messages: Sequence[LLMMessage], **_: object) -> str:
            return "Bogus claim [does_not_exist]."

    gen = CitationGenerator(
        retrieval_pipeline=_StubPipeline(chunks=chunks),  # type: ignore[arg-type]
        llm_client=_PhantomLLM(),
        nli_verifier=_AlwaysEntailingVerifier(),
    )
    out = gen.generate("Q?")
    assert out.verified_claims == ()
    assert out.suppressed_claims
    assert out.suppressed_claims[0].reason == "phantom_citation"


def test_generator_records_uncited_claim_under_suppressed() -> None:
    chunks = [_chunk("a:1", "Real passage.")]

    class _UncitedLLM:
        name = "uncited"

        def generate(self, messages: Sequence[LLMMessage], **_: object) -> str:
            return "I am sure of this without a citation."

    gen = CitationGenerator(
        retrieval_pipeline=_StubPipeline(chunks=chunks),  # type: ignore[arg-type]
        llm_client=_UncitedLLM(),
        nli_verifier=_AlwaysEntailingVerifier(),
    )
    out = gen.generate("Q?")
    assert out.verified_claims == ()
    assert out.suppressed_claims
    assert out.suppressed_claims[0].reason == "no_citation"
