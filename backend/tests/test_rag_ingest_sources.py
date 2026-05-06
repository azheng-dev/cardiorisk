"""Phase 3.1: tests for the CORPUS_SOURCES manifest of guideline PDFs."""

from __future__ import annotations

import re

import pytest

from cardiorisk.rag.ingest.sources import (
    CORPUS_SOURCES,
    CorpusSource,
    get_source,
)

_HTTP_URL_RE = re.compile(r"^https?://[^\s]+$")


def test_sources_non_empty() -> None:
    assert len(CORPUS_SOURCES) >= 1


def test_every_source_is_a_corpus_source() -> None:
    for s in CORPUS_SOURCES:
        assert isinstance(s, CorpusSource)


def test_doc_ids_are_unique() -> None:
    ids = [s.doc_id for s in CORPUS_SOURCES]
    assert len(ids) == len(set(ids))


def test_doc_ids_are_snake_case_no_extension() -> None:
    for s in CORPUS_SOURCES:
        assert re.match(r"^[a-z0-9_]+$", s.doc_id), s.doc_id
        assert "." not in s.doc_id


def test_titles_are_non_empty() -> None:
    for s in CORPUS_SOURCES:
        assert s.title.strip()


def test_publishers_are_known() -> None:
    known = {"RACGP", "NVDPA"}
    for s in CORPUS_SOURCES:
        assert s.publisher in known, s.publisher


def test_urls_are_http_or_https() -> None:
    for s in CORPUS_SOURCES:
        assert _HTTP_URL_RE.match(s.url), s.url


def test_out_filenames_are_unique_and_have_pdf_extension() -> None:
    filenames = [s.out_filename for s in CORPUS_SOURCES]
    assert len(filenames) == len(set(filenames))
    for fn in filenames:
        assert fn.endswith(".pdf"), fn


def test_checksum_filenames_are_unique_and_end_in_sha256() -> None:
    cks = [s.checksum_filename for s in CORPUS_SOURCES]
    assert len(cks) == len(set(cks))
    for ck in cks:
        assert ck.endswith(".sha256"), ck
        assert ck.startswith("corpus_"), ck


def test_get_source_round_trip() -> None:
    for s in CORPUS_SOURCES:
        assert get_source(s.doc_id) is s


def test_get_source_unknown_raises_keyerror_with_known_list() -> None:
    with pytest.raises(KeyError) as exc:
        get_source("definitely_not_a_doc")
    msg = str(exc.value)
    for s in CORPUS_SOURCES:
        assert s.doc_id in msg


def test_corpus_source_is_frozen() -> None:
    s = CORPUS_SOURCES[0]
    with pytest.raises((AttributeError, Exception)):
        s.title = "tampered"  # type: ignore[misc]


def test_phase_3_1_corpus_scope_is_racgp_and_nvdpa_only() -> None:
    """Per the user's Phase 3.1 corpus-scope decision (AGENTS §2)."""
    publishers = {s.publisher for s in CORPUS_SOURCES}
    assert publishers <= {"RACGP", "NVDPA"}


def test_at_least_one_racgp_and_one_nvdpa_source() -> None:
    publishers = [s.publisher for s in CORPUS_SOURCES]
    assert "RACGP" in publishers
    assert "NVDPA" in publishers
