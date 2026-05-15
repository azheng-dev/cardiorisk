"""Lift LLM free-text into structured ``Claim`` rows.

Input shape (per the v1 prompt): each sentence ends with one or more
``[chunk_id]`` citations. Sentences without a citation are
hallucinations and the verifier drops them; sentences with a citation
go through NLI verification against every cited chunk.

Tolerance policy:

- The parser accepts ``[chunk_id]`` and the legacy ``[1]``-style
  numeric form (translated against the supplied passage list).
- Multiple citations in one sentence are split: ``... [a] [b].``
  becomes one Claim with two citations.
- Citations on a separate line immediately after a sentence are
  attached to that sentence.
- A bare ``[REFUSE]`` sentinel anywhere in the answer surfaces as
  :attr:`ParsedAnswer.is_refusal`.
- Anything else (markdown headers, bullets, etc.) is silently dropped
  rather than coerced into claims; the eval treats this as 'no
  cited claim' which the verifier counts as zero recall.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from .prompts import PromptPassage

#: Sentinel the v1 prompt instructs the LLM to emit when it cannot
#: answer the question from the supplied passages.
REFUSAL_SENTINEL: Final[str] = "[REFUSE]"

#: Refusal phrases the parser also accepts (case-insensitive). The
#: sentinel is mandatory in the v1 prompt; these are belt-and-braces
#: for LLMs that omit it.
_REFUSAL_PHRASES: Final[tuple[str, ...]] = (
    "i do not have the supporting guidance",
    "i don't have the supporting guidance",
    "i cannot answer this from the supplied",
    "no supporting guidance was found",
)

# Citation tokens: bracketed chunk_ids, plus a tolerant numeric form.
_CITE_RE = re.compile(r"\[(?P<body>[^\[\]]+?)\]")
_NUMERIC_CITE_RE = re.compile(r"^\s*(?P<n>\d+)\s*$")
# Sentence splitter: break on whitespace immediately followed by an
# uppercase letter, when the preceding character is either a sentence
# terminator (``.!?``) or a citation closing bracket (``]``). The
# closing-bracket case is what keeps "S1. [cite1] S2. [cite2]" from
# collapsing into a single sentence — the bracket-then-uppercase
# transition is the real sentence boundary in that pattern.
_SENT_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z])")


@dataclass(frozen=True)
class Claim:
    """One sentence + its cited chunk_ids.

    ``unresolved_tokens`` carries citation tokens that the LLM emitted
    but the parser could not map back to a known passage chunk_id.
    Downstream callers use the distinction between "no citation tokens
    at all" and "citation tokens present but none resolve" to
    classify the suppression reason in the audit trail.
    """

    text: str
    cited_chunk_ids: tuple[str, ...]
    unresolved_tokens: tuple[str, ...] = ()

    @property
    def has_citation(self) -> bool:
        return bool(self.cited_chunk_ids)

    @property
    def attempted_citation(self) -> bool:
        return bool(self.cited_chunk_ids) or bool(self.unresolved_tokens)


@dataclass(frozen=True)
class ParsedAnswer:
    """Structured form of an LLM response."""

    raw_text: str
    claims: tuple[Claim, ...] = ()
    is_refusal: bool = False
    unparseable_lines: tuple[str, ...] = field(default_factory=tuple)


def _resolve_citation_token(
    token: str,
    passage_chunk_ids: Sequence[str],
) -> str | None:
    """Map a citation token to a known chunk_id, or ``None``.

    Accepted forms:

    - Exact ``chunk_id`` match: ``token == "racgp_redbook_cvd:p012:c003"``.
    - ``chunk_id=racgp_redbook_cvd:p012:c003`` form (the prompt's own
      passage header).
    - Numeric ``"3"`` interpreted as 1-indexed into ``passage_chunk_ids``.
    """
    token = token.strip()
    if not token:
        return None
    if token == REFUSAL_SENTINEL.strip("[]"):
        return None
    if token.startswith("chunk_id="):
        token = token[len("chunk_id=") :]
    if token in passage_chunk_ids:
        return token
    numeric = _NUMERIC_CITE_RE.match(token)
    if numeric:
        idx = int(numeric.group("n")) - 1
        if 0 <= idx < len(passage_chunk_ids):
            return passage_chunk_ids[idx]
    return None


def _looks_like_refusal(text: str) -> bool:
    if REFUSAL_SENTINEL in text:
        return True
    lowered = text.lower()
    return any(p in lowered for p in _REFUSAL_PHRASES)


def parse_answer(
    raw_text: str,
    passages: Sequence[PromptPassage],
) -> ParsedAnswer:
    """Lift one LLM response into a :class:`ParsedAnswer`.

    Args:
        raw_text: Exactly the string the LLM returned.
        passages: The passages the prompt supplied. Used to resolve
            numeric citations and to validate that cited chunk_ids
            actually exist (claims with phantom citations are kept
            but flagged with ``cited_chunk_ids=()`` so the verifier
            drops them).

    Returns:
        A :class:`ParsedAnswer`. ``is_refusal`` is true iff the
        response contains the ``[REFUSE]`` sentinel or one of the
        accepted refusal phrases AND no claim survived parsing with
        a real citation.
    """
    raw = raw_text.strip()
    passage_chunk_ids = [p.chunk_id for p in passages]

    # Refusal short-circuit; we still parse claims so the test for
    # 'refused-but-also-cited' surfaces in unparseable_lines.
    refusal = _looks_like_refusal(raw)

    # Strip the refusal sentinel so it doesn't pollute the sentence
    # split.
    body = raw.replace(REFUSAL_SENTINEL, "").strip()

    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(body) if s.strip()]

    claims: list[Claim] = []
    unparseable: list[str] = []
    for sentence in sentences:
        cite_matches = list(_CITE_RE.finditer(sentence))
        if not cite_matches:
            # Bare sentence — likely the refusal phrase itself, or a
            # hallucination the LLM forgot to cite. Either way, it
            # produces a Claim with no citations so the verifier can
            # uniformly drop it.
            sentence_text = sentence.rstrip(".!?")
            if sentence_text and not _looks_like_refusal(sentence_text):
                claims.append(Claim(text=sentence_text + ".", cited_chunk_ids=()))
            elif sentence_text:
                unparseable.append(sentence_text)
            continue

        # Citation tokens found; collect resolved chunk_ids and strip
        # them out of the sentence text.
        resolved: list[str] = []
        unresolved: list[str] = []
        for match in cite_matches:
            for token in match.group("body").split(","):
                normalised = token.strip()
                if not normalised or normalised == "REFUSE":
                    continue
                cid = _resolve_citation_token(normalised, passage_chunk_ids)
                if cid is not None and cid not in resolved:
                    resolved.append(cid)
                elif cid is None:
                    unresolved.append(normalised)
                    unparseable.append(f"unresolved citation token: {normalised!r}")
        cleaned = _CITE_RE.sub("", sentence).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        if not cleaned.endswith((".", "!", "?")):
            cleaned = cleaned + "."
        claims.append(
            Claim(
                text=cleaned,
                cited_chunk_ids=tuple(resolved),
                unresolved_tokens=tuple(unresolved),
            )
        )

    return ParsedAnswer(
        raw_text=raw_text,
        claims=tuple(claims),
        is_refusal=refusal and not any(c.cited_chunk_ids for c in claims),
        unparseable_lines=tuple(unparseable),
    )
