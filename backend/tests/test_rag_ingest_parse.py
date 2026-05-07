"""Phase 3.1: tests for the PDF/markdown parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from cardiorisk.data.paths import FIXTURE_CORPUS_DIR
from cardiorisk.rag.ingest.parse import (
    ParsedDoc,
    ParsedPage,
    ParseError,
    load_parsed_doc,
    parse_doc_for_source,
    parse_markdown_fixture,
    save_parsed_doc,
)


@pytest.fixture
def racgp_fixture_path() -> Path:
    return FIXTURE_CORPUS_DIR / "fixture_racgp_cvd.md"


@pytest.fixture
def nvdpa_fixture_path() -> Path:
    return FIXTURE_CORPUS_DIR / "fixture_nvdpa_quickref.md"


def test_parse_markdown_fixture_returns_parsed_doc(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="fixture_racgp_cvd")
    assert isinstance(doc, ParsedDoc)
    assert doc.doc_id == "fixture_racgp_cvd"
    assert len(doc.pages) >= 1


def test_fixture_with_page_break_marker_yields_two_pages(
    racgp_fixture_path: Path,
) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    assert len(doc.pages) == 2


def test_page_numbers_are_monotonic(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    nums = [p.page_no for p in doc.pages]
    assert nums == sorted(nums)
    assert nums[0] == 1


def test_char_offsets_match_full_text(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    full = doc.full_text()
    for p in doc.pages:
        assert full[p.char_offset : p.char_offset + len(p.text)] == p.text


def test_full_text_concatenates_with_double_newline(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    expected = "\n\n".join(p.text for p in doc.pages)
    assert doc.full_text() == expected


def test_page_for_offset_round_trip(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    for p in doc.pages:
        if p.text:
            assert doc.page_for_offset(p.char_offset) == p.page_no
            mid = p.char_offset + max(0, len(p.text) // 2)
            assert doc.page_for_offset(mid) == p.page_no


def test_page_for_offset_negative_raises() -> None:
    doc = ParsedDoc(doc_id="x", pages=(ParsedPage(page_no=1, text="hi", char_offset=0),))
    with pytest.raises(ValueError):
        doc.page_for_offset(-1)


def test_save_then_load_round_trip(racgp_fixture_path: Path, tmp_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="rt")
    out = tmp_path / "rt.jsonl"
    sha = save_parsed_doc(doc, out)
    assert len(sha) == 64
    loaded = load_parsed_doc(out, doc_id="rt")
    assert loaded == doc


def test_parse_markdown_fixture_missing_file_raises() -> None:
    with pytest.raises(ParseError):
        parse_markdown_fixture(Path("/nonexistent/missing.md"), doc_id="x")


def test_parse_doc_for_source_dispatches_on_extension(racgp_fixture_path: Path) -> None:
    doc = parse_doc_for_source(racgp_fixture_path, doc_id="dispatch_test")
    assert doc.doc_id == "dispatch_test"
    assert len(doc.pages) >= 1


def test_parse_doc_for_source_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "what.docx"
    p.write_text("not supported")
    with pytest.raises(ParseError):
        parse_doc_for_source(p, doc_id="x")


def test_empty_page_handled_gracefully(tmp_path: Path) -> None:
    p = tmp_path / "two_pages_one_empty.md"
    p.write_text("First page text\n\n<!-- page break -->\n\n", encoding="utf-8")
    doc = parse_markdown_fixture(p, doc_id="x")
    assert len(doc.pages) == 2
    assert doc.pages[0].text.startswith("First page")
    assert doc.pages[1].text == ""


def test_single_page_when_no_marker(tmp_path: Path) -> None:
    p = tmp_path / "single.md"
    p.write_text("Just one page of text. No break marker.\n", encoding="utf-8")
    doc = parse_markdown_fixture(p, doc_id="x")
    assert len(doc.pages) == 1
    assert doc.pages[0].page_no == 1


def test_load_parsed_doc_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        load_parsed_doc(tmp_path / "missing.jsonl", doc_id="x")


def test_save_parsed_doc_writes_one_line_per_page(racgp_fixture_path: Path, tmp_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    out = tmp_path / "out.jsonl"
    save_parsed_doc(doc, out)
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(doc.pages)


def test_page_for_offset_past_end_returns_last_page(racgp_fixture_path: Path) -> None:
    doc = parse_markdown_fixture(racgp_fixture_path, doc_id="x")
    last = doc.pages[-1]
    assert doc.page_for_offset(10**9) == last.page_no
