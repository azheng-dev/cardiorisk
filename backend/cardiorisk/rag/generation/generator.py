"""End-to-end citation-mandatory generator.

Pipeline (one ``CitationGenerator.generate(query)`` call):

1. Run the retrieval pipeline (Phase 3.2 hybrid) with the question as
   the query. Returns up to ``top_k`` :class:`RetrievedChunk` rows.
2. Pack the retrieved chunks into :class:`PromptPassage` rows and
   render the citation-required prompt template.
3. Send the prompt to the LLM client.
4. Parse the response with :func:`parse_answer` into
   :class:`Claim` rows.
5. For every parsed claim: run NLI against every cited chunk's text.
   The claim survives iff at least one citation yields
   ``P(entailment) >= entail_threshold``. Survivors keep their
   strongest-entailment citation as the headline cite; the rest are
   recorded as supporting cites.
6. Return a :class:`GeneratedAnswer` with the verified claims, the
   suppressed-claim audit trail, and the raw LLM text for debugging.

Suppression policy (ADR-017): claims that fail verification are
dropped, never silently re-written. The generator does NOT re-prompt
the LLM with "fix this claim" — re-prompting is a known hallucination
amplifier (the LLM doubles down on the unsupported assertion).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from ..retrieval.pipeline import RetrievalPipeline, RetrievedChunk
from .llm import BaseLLMClient, LLMMessage
from .nli import DEFAULT_ENTAILMENT_THRESHOLD, BaseNLIVerifier, EntailmentResult
from .parser import REFUSAL_SENTINEL, ParsedAnswer, parse_answer
from .prompts import DEFAULT_PROMPT, PromptPassage, render_citation_prompt

#: Default top-k passages the prompt sees. Smaller than the eval
#: ``top_k`` of 5 because the prompt budget caps quickly with the
#: BGE-M3 chunk size.
DEFAULT_PROMPT_TOP_K: Final[int] = 5
#: Default canonical refusal text. Mirrors the v1 prompt so the
#: ParsedAnswer.is_refusal check stays consistent.
REFUSAL_TEXT: Final[str] = "I do not have the supporting guidance for that question."


@dataclass(frozen=True)
class VerifiedClaim:
    """A claim that survived NLI verification."""

    text: str
    headline_chunk_id: str
    headline_score: float
    supporting_chunk_ids: tuple[str, ...]
    supporting_scores: tuple[float, ...]


@dataclass(frozen=True)
class SuppressedClaim:
    """A claim the verifier dropped, kept for auditability."""

    text: str
    cited_chunk_ids: tuple[str, ...]
    best_entailment: float
    reason: str  # 'no_citation' | 'phantom_citation' | 'no_passage_entails'


@dataclass(frozen=True)
class GeneratedAnswer:
    """Output of :class:`CitationGenerator.generate`."""

    query: str
    raw_llm_text: str
    is_refusal: bool
    verified_claims: tuple[VerifiedClaim, ...] = field(default_factory=tuple)
    suppressed_claims: tuple[SuppressedClaim, ...] = field(default_factory=tuple)
    retrieved: tuple[RetrievedChunk, ...] = field(default_factory=tuple)

    @property
    def verified_text(self) -> str:
        """Joined verified-claim text — what a UI surface should display."""
        if self.is_refusal and not self.verified_claims:
            return REFUSAL_TEXT
        if not self.verified_claims:
            return ""
        return " ".join(claim.text for claim in self.verified_claims)


class CitationGenerator:
    """Wires retrieval + LLM + NLI into one ``generate(query)`` call."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: BaseLLMClient,
        nli_verifier: BaseNLIVerifier,
        prompt_template: str = DEFAULT_PROMPT,
        prompt_top_k: int = DEFAULT_PROMPT_TOP_K,
        with_rerank: bool = False,
        entail_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
        max_tokens: int = 512,
    ) -> None:
        self._retrieval = retrieval_pipeline
        self._llm = llm_client
        self._nli = nli_verifier
        self._prompt_template = prompt_template
        self._prompt_top_k = prompt_top_k
        self._with_rerank = with_rerank
        self._entail_threshold = entail_threshold
        self._max_tokens = max_tokens

    def generate(self, query: str) -> GeneratedAnswer:
        """Run the full retrieve / prompt / parse / verify pipeline."""
        retrieved = tuple(
            self._retrieval.retrieve(
                query,
                top_k=self._prompt_top_k,
                with_rerank=self._with_rerank,
            )
        )
        passages = [
            PromptPassage(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                page_start=r.chunk.page_start,
                page_end=r.chunk.page_end,
                text=r.chunk.text,
            )
            for r in retrieved
        ]

        if not passages:
            return GeneratedAnswer(
                query=query,
                raw_llm_text=f"{REFUSAL_TEXT} {REFUSAL_SENTINEL}",
                is_refusal=True,
                retrieved=retrieved,
            )

        prompt = render_citation_prompt(
            question=query,
            passages=passages,
            template_name=self._prompt_template,
        )
        raw = self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        parsed = parse_answer(raw, passages)
        verified, suppressed = self._verify_claims(parsed, passages)
        # If the LLM refused AND nothing survived verification, surface
        # the refusal flag. If the LLM refused but a claim slipped
        # through with a real citation we treat it as a non-refusal
        # answer (the refusal sentinel was misplaced); the verifier
        # already kept the supported claim.
        is_refusal = parsed.is_refusal and not verified
        return GeneratedAnswer(
            query=query,
            raw_llm_text=raw,
            is_refusal=is_refusal,
            verified_claims=tuple(verified),
            suppressed_claims=tuple(suppressed),
            retrieved=retrieved,
        )

    def _verify_claims(
        self,
        parsed: ParsedAnswer,
        passages: Sequence[PromptPassage],
    ) -> tuple[list[VerifiedClaim], list[SuppressedClaim]]:
        passage_by_id = {p.chunk_id: p for p in passages}
        verified: list[VerifiedClaim] = []
        suppressed: list[SuppressedClaim] = []

        for claim in parsed.claims:
            if not claim.cited_chunk_ids:
                # Two failure modes share this branch — distinguish so
                # the audit trail says whether the LLM forgot to cite
                # at all (no_citation) or cited something that does
                # not match any retrieved passage (phantom_citation).
                reason = "phantom_citation" if claim.unresolved_tokens else "no_citation"
                suppressed.append(
                    SuppressedClaim(
                        text=claim.text,
                        cited_chunk_ids=claim.unresolved_tokens,
                        best_entailment=0.0,
                        reason=reason,
                    )
                )
                continue

            valid_cites = [c for c in claim.cited_chunk_ids if c in passage_by_id]
            if not valid_cites:
                # Defensive: parse_answer already filters phantoms out
                # of cited_chunk_ids, but a future parser refactor that
                # changes the contract should still suppress safely.
                suppressed.append(
                    SuppressedClaim(
                        text=claim.text,
                        cited_chunk_ids=claim.cited_chunk_ids,
                        best_entailment=0.0,
                        reason="phantom_citation",
                    )
                )
                continue

            pairs = [(passage_by_id[c].text, claim.text) for c in valid_cites]
            results = self._nli.entails_batch(pairs)
            scored: list[tuple[str, EntailmentResult]] = list(
                zip(valid_cites, results, strict=True)
            )
            scored.sort(key=lambda pair: pair[1].p_entailment, reverse=True)
            best_cid, best_res = scored[0]
            if best_res.p_entailment < self._entail_threshold:
                suppressed.append(
                    SuppressedClaim(
                        text=claim.text,
                        cited_chunk_ids=claim.cited_chunk_ids,
                        best_entailment=best_res.p_entailment,
                        reason="no_passage_entails",
                    )
                )
                continue

            supporting = [
                (cid, res.p_entailment)
                for cid, res in scored[1:]
                if res.p_entailment >= self._entail_threshold
            ]
            verified.append(
                VerifiedClaim(
                    text=claim.text,
                    headline_chunk_id=best_cid,
                    headline_score=best_res.p_entailment,
                    supporting_chunk_ids=tuple(c for c, _ in supporting),
                    supporting_scores=tuple(s for _, s in supporting),
                )
            )

        return verified, suppressed
