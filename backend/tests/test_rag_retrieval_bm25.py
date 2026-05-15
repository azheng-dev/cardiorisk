"""Tests for the BM25 sparse-retrieval wrapper.

Exact-phrase Qs should rank the matching document first; the
tokeniser should preserve numeric tokens like '140' and '10%' that
are clinically meaningful.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cardiorisk.rag.retrieval.bm25 import STOPWORDS, BM25Index, tokenise


def test_tokenise_keeps_alphanum() -> None:
    toks = tokenise("Patients aged 45 to 79 years old.")
    assert "patients" in toks
    assert "45" in toks
    assert "79" in toks
    assert "years" in toks


def test_tokenise_drops_stopwords() -> None:
    toks = tokenise("The patient is at risk of CVD.")
    assert "the" not in toks
    assert "is" not in toks
    assert "at" not in toks
    # 'of' is a stopword too; we leave 'cvd' (length 3).
    assert "of" not in toks
    assert "patient" in toks
    assert "cvd" in toks


def test_tokenise_keeps_clinically_meaningful_negation() -> None:
    """Phase 3.2 semantics: 'not' inverts clinical meaning, so it
    must NOT be in the stopword list (per ADR-016 §3 footnote)."""
    assert "not" not in STOPWORDS
    toks = tokenise("aspirin is not routinely recommended")
    assert "not" in toks


def test_tokenise_keeps_percent_tokens() -> None:
    toks = tokenise("Patients with risk above 10% should be treated.")
    assert "10%" in toks


def test_tokenise_lowercases() -> None:
    toks = tokenise("Cardiovascular Disease")
    assert all(t.islower() or any(c.isdigit() for c in t) for t in toks)


def test_tokenise_drops_short_alpha_tokens() -> None:
    """Single-letter alphabetic tokens are dropped; single-digit numeric tokens are kept."""
    toks = tokenise("a 5% increase in i risk")
    assert "5%" in toks
    # Single 'a' and 'i' are dropped (alpha singletons + stopword).
    assert "a" not in toks
    assert "i" not in toks


def test_build_then_search_finds_keyword_match() -> None:
    idx = BM25Index()
    idx.build(
        chunk_ids=["c1", "c2", "c3"],
        texts=[
            "Aspirin is not routinely recommended for primary prevention.",
            "Statins should be offered to high-risk patients.",
            "Lifestyle interventions include physical activity.",
        ],
    )
    hits = idx.search("aspirin primary prevention", top_k=3)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0


def test_search_returns_no_hit_on_empty_query() -> None:
    idx = BM25Index()
    idx.build(chunk_ids=["c1"], texts=["something"])
    assert idx.search("", top_k=3) == []
    # Whitespace + stopwords-only query also returns nothing.
    assert idx.search("the of and to", top_k=3) == []


def test_save_load_round_trip(tmp_path: Path) -> None:
    idx = BM25Index()
    idx.build(
        chunk_ids=["a", "b", "c"],
        texts=[
            "aspirin therapy primary prevention",
            "statin therapy high risk",
            "lifestyle intervention smoking cessation",
        ],
    )
    idx.save(tmp_path)

    loaded = BM25Index.load(tmp_path)
    assert loaded.chunk_ids == ("a", "b", "c")
    hits = loaded.search("aspirin primary", top_k=3)
    assert hits[0].chunk_id == "a"
    assert hits[0].score >= hits[-1].score


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BM25Index.load(tmp_path)


def test_empty_corpus_search_returns_empty() -> None:
    idx = BM25Index()
    idx.build(chunk_ids=[], texts=[])
    assert idx.search("query", top_k=5) == []


def test_length_mismatch_raises() -> None:
    idx = BM25Index()
    with pytest.raises(ValueError):
        idx.build(chunk_ids=["a", "b"], texts=["only one"])


def test_handles_chunk_with_only_stopwords() -> None:
    """A chunk that tokenises to [] must not crash rank_bm25."""
    idx = BM25Index()
    idx.build(
        chunk_ids=["c1", "c2", "c3"],
        texts=[
            "the of and to",
            "aspirin therapy primary prevention",
            "lifestyle intervention smoking cessation",
        ],
    )
    hits = idx.search("aspirin primary", top_k=3)
    assert hits[0].chunk_id == "c2"


def test_score_descends_with_relevance() -> None:
    """Identical query terms; document with more matches should rank higher.

    BM25Okapi's IDF goes to zero when a term appears in exactly half
    the documents (`log((N-n+0.5)/(n+0.5)) = log(1) = 0`). We size the
    fixture so 'aspirin' is in 2/5 docs (IDF > 0).
    """
    idx = BM25Index()
    idx.build(
        chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        texts=[
            "aspirin aspirin aspirin therapy primary",
            "aspirin therapy mentioned briefly",
            "statin therapy high risk patients",
            "lifestyle intervention smoking cessation",
            "blood pressure medication monitoring",
        ],
    )
    hits = idx.search("aspirin", top_k=5)
    # c1 has 3x 'aspirin'; c2 has 1x; c3/c4/c5 have none.
    assert hits[0].chunk_id == "c1"
    # c2 should be the second-ranked aspirin-containing chunk.
    aspirin_positives = [h for h in hits if h.chunk_id in {"c1", "c2"}]
    assert aspirin_positives[0].chunk_id == "c1"
    assert aspirin_positives[1].chunk_id == "c2"
