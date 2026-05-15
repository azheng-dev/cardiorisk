"""Tests for the answer parser."""

from __future__ import annotations

from cardiorisk.rag.generation.parser import REFUSAL_SENTINEL, parse_answer
from cardiorisk.rag.generation.prompts import PromptPassage


def _passage(chunk_id: str) -> PromptPassage:
    return PromptPassage(
        chunk_id=chunk_id,
        doc_id="doc",
        page_start=1,
        page_end=1,
        text="x",
    )


def test_single_sentence_with_chunk_id_citation_parses() -> None:
    parsed = parse_answer(
        "Adults aged 45 to 79 should be assessed [fixture_racgp_cvd:p1:c1].",
        [_passage("fixture_racgp_cvd:p1:c1")],
    )
    assert len(parsed.claims) == 1
    claim = parsed.claims[0]
    assert claim.cited_chunk_ids == ("fixture_racgp_cvd:p1:c1",)
    assert "[" not in claim.text
    assert claim.text.endswith(".")
    assert not parsed.is_refusal


def test_two_sentences_each_get_their_citation() -> None:
    parsed = parse_answer(
        "Sentence one [a:1]. Sentence two [b:1].",
        [_passage("a:1"), _passage("b:1")],
    )
    assert len(parsed.claims) == 2
    assert parsed.claims[0].cited_chunk_ids == ("a:1",)
    assert parsed.claims[1].cited_chunk_ids == ("b:1",)


def test_multiple_citations_in_one_sentence_collected() -> None:
    parsed = parse_answer(
        "Big claim [a:1] [b:1].",
        [_passage("a:1"), _passage("b:1")],
    )
    assert len(parsed.claims) == 1
    assert set(parsed.claims[0].cited_chunk_ids) == {"a:1", "b:1"}


def test_numeric_citation_resolves_to_passage_index() -> None:
    parsed = parse_answer(
        "Numeric form [1].",
        [_passage("a:1"), _passage("b:1")],
    )
    assert parsed.claims[0].cited_chunk_ids == ("a:1",)


def test_phantom_citation_yields_empty_cited_chunk_ids() -> None:
    parsed = parse_answer(
        "Bogus claim [does_not_exist].",
        [_passage("a:1")],
    )
    assert parsed.claims[0].cited_chunk_ids == ()
    assert any("unresolved" in line for line in parsed.unparseable_lines)


def test_uncited_sentence_becomes_uncited_claim() -> None:
    parsed = parse_answer(
        "I am sure of this without a citation.",
        [_passage("a:1")],
    )
    assert len(parsed.claims) == 1
    assert parsed.claims[0].cited_chunk_ids == ()


def test_refusal_sentinel_sets_flag() -> None:
    parsed = parse_answer(
        f"I do not have the supporting guidance for that question. {REFUSAL_SENTINEL}",
        [_passage("a:1")],
    )
    assert parsed.is_refusal
    # The refusal phrase itself is dropped from claims (no citation
    # possible), so claims list is empty.
    assert all(not c.cited_chunk_ids for c in parsed.claims)


def test_refusal_with_real_cited_claim_does_not_set_refusal() -> None:
    parsed = parse_answer(
        f"This is supported [a:1]. {REFUSAL_SENTINEL}",
        [_passage("a:1")],
    )
    assert not parsed.is_refusal
    assert parsed.claims[0].cited_chunk_ids == ("a:1",)


def test_chunk_id_prefix_form_resolves() -> None:
    parsed = parse_answer(
        "Claim [chunk_id=a:1].",
        [_passage("a:1")],
    )
    assert parsed.claims[0].cited_chunk_ids == ("a:1",)
