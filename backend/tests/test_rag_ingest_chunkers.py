"""Phase 3.1: tests for the three chunking strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

from cardiorisk.data.paths import FIXTURE_CORPUS_DIR
from cardiorisk.rag.ingest.chunkers import (
    NAME_TO_CHUNKER,
    Chunk,
    Chunker,
    HybridChunker,
    SemanticChunker,
    TokenWindowChunker,
    chunk_id_for,
    count_tokens,
    load_chunks,
    save_chunks,
)
from cardiorisk.rag.ingest.parse import (
    ParsedDoc,
    parse_markdown_fixture,
)


@pytest.fixture
def racgp_doc() -> ParsedDoc:
    return parse_markdown_fixture(
        FIXTURE_CORPUS_DIR / "fixture_racgp_cvd.md",
        doc_id="fixture_racgp_cvd",
    )


@pytest.fixture
def nvdpa_doc() -> ParsedDoc:
    return parse_markdown_fixture(
        FIXTURE_CORPUS_DIR / "fixture_nvdpa_quickref.md",
        doc_id="fixture_nvdpa_quickref",
    )


# -----------------------------------------------------------------
# Registry
# -----------------------------------------------------------------


def test_registry_has_all_three_strategies() -> None:
    assert set(NAME_TO_CHUNKER) == {"token", "semantic", "hybrid"}


def test_registered_factories_construct_expected_classes() -> None:
    assert isinstance(NAME_TO_CHUNKER["token"](), TokenWindowChunker)
    assert isinstance(NAME_TO_CHUNKER["semantic"](), SemanticChunker)
    assert isinstance(NAME_TO_CHUNKER["hybrid"](), HybridChunker)


# -----------------------------------------------------------------
# count_tokens + chunk_id_for
# -----------------------------------------------------------------


def test_count_tokens_smoke() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello") >= 1
    long_text = " ".join(["lorem"] * 200)
    assert count_tokens(long_text) >= 100


def test_chunk_id_is_deterministic() -> None:
    a = chunk_id_for(doc_id="d", strategy="token", char_start=0, char_end=100)
    b = chunk_id_for(doc_id="d", strategy="token", char_start=0, char_end=100)
    assert a == b
    assert len(a) == 16


def test_chunk_id_changes_with_inputs() -> None:
    a = chunk_id_for(doc_id="d", strategy="token", char_start=0, char_end=100)
    b = chunk_id_for(doc_id="d", strategy="token", char_start=0, char_end=101)
    c = chunk_id_for(doc_id="e", strategy="token", char_start=0, char_end=100)
    d = chunk_id_for(doc_id="d", strategy="semantic", char_start=0, char_end=100)
    assert len({a, b, c, d}) == 4


# -----------------------------------------------------------------
# Token-window chunker
# -----------------------------------------------------------------


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunkers_produce_chunk_dataclasses(
    racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]
) -> None:
    chunker = ChunkerCls()
    chunks = chunker.chunk(racgp_doc)
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_id == racgp_doc.doc_id
        assert c.strategy == chunker.name


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunk_spans_in_bounds(racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]) -> None:
    chunker = ChunkerCls()
    full = racgp_doc.full_text()
    for c in chunker.chunk(racgp_doc):
        assert 0 <= c.char_start < c.char_end <= len(full)


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunk_text_matches_span(racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]) -> None:
    chunker = ChunkerCls()
    full = racgp_doc.full_text()
    for c in chunker.chunk(racgp_doc):
        assert c.text == full[c.char_start : c.char_end]


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunk_pages_within_doc_pages(racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]) -> None:
    chunker = ChunkerCls()
    page_nos = {p.page_no for p in racgp_doc.pages}
    for c in chunker.chunk(racgp_doc):
        assert c.page_start in page_nos
        assert c.page_end in page_nos
        assert c.page_start <= c.page_end


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunkers_are_deterministic(racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]) -> None:
    a = ChunkerCls().chunk(racgp_doc)
    b = ChunkerCls().chunk(racgp_doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunk_ids_unique_within_strategy(racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]) -> None:
    chunks = ChunkerCls().chunk(racgp_doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunkers_handle_empty_doc(ChunkerCls: type[Chunker]) -> None:
    empty = ParsedDoc(doc_id="empty", pages=())
    assert ChunkerCls().chunk(empty) == []


@pytest.mark.parametrize("ChunkerCls", [TokenWindowChunker, SemanticChunker, HybridChunker])
def test_chunks_cover_at_least_95pct_of_text(
    racgp_doc: ParsedDoc, ChunkerCls: type[Chunker]
) -> None:
    """Chunks may overlap; their union must cover ≥95% of doc chars."""
    chunks = ChunkerCls().chunk(racgp_doc)
    assert chunks
    full_len = len(racgp_doc.full_text())
    covered = [False] * full_len
    for c in chunks:
        for i in range(c.char_start, c.char_end):
            covered[i] = True
    coverage = sum(covered) / max(full_len, 1)
    assert coverage >= 0.95, f"coverage={coverage:.3f}"


def test_token_window_n_tokens_within_target() -> None:
    text = "A simple sentence. " * 400  # roughly 1600+ tokens worth
    doc = ParsedDoc.__new__(ParsedDoc)
    object.__setattr__(doc, "doc_id", "x")
    from cardiorisk.rag.ingest.parse import ParsedPage

    object.__setattr__(doc, "pages", (ParsedPage(page_no=1, text=text, char_offset=0),))
    chunks = TokenWindowChunker(window_tokens=128, stride_tokens=64).chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.n_tokens <= 128


def test_token_window_invalid_params_raises() -> None:
    with pytest.raises(ValueError):
        TokenWindowChunker(window_tokens=0)
    with pytest.raises(ValueError):
        TokenWindowChunker(stride_tokens=0)
    with pytest.raises(ValueError):
        TokenWindowChunker(window_tokens=10, stride_tokens=20)


# -----------------------------------------------------------------
# Semantic chunker specifics
# -----------------------------------------------------------------


def test_semantic_chunks_end_on_sentence_boundary(racgp_doc: ParsedDoc) -> None:
    chunks = SemanticChunker().chunk(racgp_doc)
    assert chunks
    for c in chunks:
        last = c.text.rstrip()
        if last:
            assert last[-1] in ".!?:", c.text[-40:]


def test_semantic_invalid_params_raises() -> None:
    with pytest.raises(ValueError):
        SemanticChunker(target_tokens=0)
    with pytest.raises(ValueError):
        SemanticChunker(overlap_sentences=-1)


def test_semantic_one_sentence_doc(tmp_path: Path) -> None:
    text = "Just one sentence with no boundary."
    doc = parse_markdown_fixture(_write_md(tmp_path, text), doc_id="x")
    chunks = SemanticChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == text


# -----------------------------------------------------------------
# Hybrid chunker specifics
# -----------------------------------------------------------------


def test_hybrid_attaches_section_path(racgp_doc: ParsedDoc) -> None:
    chunks = HybridChunker().chunk(racgp_doc)
    assert chunks
    has_path = [c for c in chunks if c.section_path]
    assert has_path, "expected at least one chunk to carry a section_path"


def test_hybrid_section_paths_cite_real_headings(racgp_doc: ParsedDoc) -> None:
    chunks = HybridChunker().chunk(racgp_doc)
    full_text = racgp_doc.full_text()
    for c in chunks:
        for heading in c.section_path:
            assert heading in full_text, heading


def test_hybrid_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        HybridChunker(upper_ratio_threshold=2.0)


# -----------------------------------------------------------------
# Save / load round-trip
# -----------------------------------------------------------------


def test_save_then_load_chunks_round_trip(racgp_doc: ParsedDoc, tmp_path: Path) -> None:
    chunks = TokenWindowChunker().chunk(racgp_doc)
    out = tmp_path / "chunks.jsonl"
    sha = save_chunks(chunks, out)
    assert len(sha) == 64
    loaded = load_chunks(out)
    assert loaded == chunks


def test_save_chunks_is_byte_stable(racgp_doc: ParsedDoc, tmp_path: Path) -> None:
    """Same input -> identical bytes (so the manifest sha256 is stable)."""
    chunks = TokenWindowChunker().chunk(racgp_doc)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    sha_a = save_chunks(chunks, out_a)
    sha_b = save_chunks(chunks, out_b)
    assert sha_a == sha_b


# -----------------------------------------------------------------
# Helper
# -----------------------------------------------------------------


def _write_md(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "md.md"
    p.write_text(text, encoding="utf-8")
    return p
