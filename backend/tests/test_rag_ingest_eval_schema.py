"""Phase 3.1: validate the retrieval eval-set against its JSON Schema.

Phase 3.2 grows ``eval/retrieval/questions.jsonl`` from 10 seeds to
the 50-Q target. This test catches schema violations the moment a
new question is added.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from cardiorisk.data.paths import FIXTURE_CORPUS_DIR, REPO_ROOT
from cardiorisk.rag.ingest.parse import parse_markdown_fixture

EVAL_DIR = REPO_ROOT / "eval" / "retrieval"
SCHEMA_PATH = EVAL_DIR / "schema.json"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"


def _load_questions() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_schema() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return payload


def _load_fixture_doc_ids() -> set[str]:
    sj = json.loads((FIXTURE_CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))
    return {entry["doc_id"] for entry in sj["sources"]}


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists()


def test_questions_file_exists_and_non_empty() -> None:
    assert QUESTIONS_PATH.exists()
    assert _load_questions()


def test_schema_is_valid_json_schema() -> None:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_every_question_validates_against_schema() -> None:
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    for q in _load_questions():
        errors = sorted(validator.iter_errors(q), key=lambda e: e.path)
        assert not errors, f"{q['id']}: {[e.message for e in errors]}"


def test_question_ids_are_unique() -> None:
    ids = [q["id"] for q in _load_questions()]
    assert len(ids) == len(set(ids))


def test_phase_3_1_seed_count_at_least_10() -> None:
    qs = _load_questions()
    seed_qs = [q for q in qs if q.get("source_phase") == "3.1"]
    assert len(seed_qs) >= 10


def test_fixture_questions_reference_known_fixture_doc_ids() -> None:
    fixture_ids = _load_fixture_doc_ids()
    for q in _load_questions():
        if q.get("requires_full_corpus"):
            continue
        assert q["expected_doc_id"] in fixture_ids, q["id"]


def test_full_corpus_questions_reference_real_corpus_doc_ids() -> None:
    from cardiorisk.rag.ingest.sources import CORPUS_SOURCES

    real_ids = {s.doc_id for s in CORPUS_SOURCES}
    for q in _load_questions():
        if q.get("requires_full_corpus"):
            assert q["expected_doc_id"] in real_ids, q["id"]


def test_fixture_question_keywords_appear_in_fixture_text() -> None:
    """For non-full-corpus seeds, every keyword must occur in the fixture.

    Whitespace is collapsed in both the haystack and the keyword so a
    line-wrap inside the fixture (or a multi-line keyword inside a
    real PDF chunk) doesn't cause a false negative. This matches how
    Phase 3.2's retrieval scorer should behave.
    """
    import re

    def _collapse(s: str) -> str:
        return re.sub(r"\s+", " ", s).lower()

    fixture_text_by_id: dict[str, str] = {}
    for entry in json.loads((FIXTURE_CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))[
        "sources"
    ]:
        doc = parse_markdown_fixture(FIXTURE_CORPUS_DIR / entry["filename"], doc_id=entry["doc_id"])
        fixture_text_by_id[entry["doc_id"]] = _collapse(doc.full_text())
    for q in _load_questions():
        if q.get("requires_full_corpus"):
            continue
        full = fixture_text_by_id[q["expected_doc_id"]]
        for kw in q["expected_span_keywords"]:
            assert _collapse(kw) in full, (q["id"], kw)


def test_fixture_question_pages_within_fixture_page_count() -> None:
    page_counts: dict[str, int] = {}
    for entry in json.loads((FIXTURE_CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))[
        "sources"
    ]:
        doc = parse_markdown_fixture(FIXTURE_CORPUS_DIR / entry["filename"], doc_id=entry["doc_id"])
        page_counts[entry["doc_id"]] = len(doc.pages)
    for q in _load_questions():
        if q.get("requires_full_corpus"):
            continue
        n_pages = page_counts[q["expected_doc_id"]]
        start, end = q["expected_page_range"]
        assert 1 <= start <= end <= n_pages, q["id"]


def test_id_format_is_qNNN() -> None:
    import re

    pat = re.compile(r"^q[0-9]{3}$")
    for q in _load_questions():
        assert pat.match(q["id"]), q["id"]


@pytest.mark.parametrize("field", ["question", "rationale", "expected_doc_id"])
def test_required_string_fields_non_empty(field: str) -> None:
    for q in _load_questions():
        assert q[field].strip(), (q["id"], field)
