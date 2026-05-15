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
    """For non-full-corpus, non-negative-case seeds, every keyword must occur in the fixture.

    Whitespace is collapsed in both the haystack and the keyword so a
    line-wrap inside the fixture (or a multi-line keyword inside a
    real PDF chunk) doesn't cause a false negative. This matches how
    Phase 3.2's retrieval scorer should behave.

    Negative-case (``expected_no_hit: true``) Qs are skipped — by
    construction their keywords should NOT appear in the corpus.
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
        if q.get("expected_no_hit"):
            continue
        full = fixture_text_by_id[q["expected_doc_id"]]
        for kw in q["expected_span_keywords"]:
            assert _collapse(kw) in full, (q["id"], kw)


def test_negative_case_keywords_do_not_appear_in_fixture_text() -> None:
    """Inverse invariant: ``expected_no_hit: true`` Qs must have keywords absent.

    If a negative-case Q's keyword does in fact appear in the fixture,
    the negative case is mis-authored — a retriever returning that
    chunk would be correct, not hallucinating. We catch that mistake
    in CI.
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
    full_corpus_blob = " ".join(fixture_text_by_id.values())
    for q in _load_questions():
        if not q.get("expected_no_hit"):
            continue
        if q.get("requires_full_corpus"):
            continue
        for kw in q["expected_span_keywords"]:
            assert _collapse(kw) not in full_corpus_blob, (q["id"], kw)


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
        if q.get("expected_no_hit"):
            # Negative-case Qs use sentinel page ranges; the scorer ignores them.
            continue
        n_pages = page_counts[q["expected_doc_id"]]
        start, end = q["expected_page_range"]
        assert 1 <= start <= end <= n_pages, q["id"]


def test_phase_3_2_target_count_at_least_50() -> None:
    """Phase 3.2 promised the eval set grows from 10 to 50 Qs."""
    assert len(_load_questions()) >= 50


def test_phase_3_2_distribution_targets() -> None:
    """Per ADR-016 / docs/research/13: ~6 Qs per tag + 5 negative-cases."""
    qs = _load_questions()
    by_tag: dict[str, int] = {}
    for q in qs:
        for tag in q.get("tags", []):
            by_tag[tag] = by_tag.get(tag, 0) + 1
    # Every closed-set tag has at least 4 Qs (eval-set is small but
    # stratified). Negative cases at least 5 per the plan.
    for tag in (
        "risk_assessment",
        "pharmacotherapy",
        "lifestyle",
        "communication",
        "reclassifiers",
        "follow_up",
    ):
        assert by_tag.get(tag, 0) >= 4, (tag, by_tag.get(tag, 0))
    assert by_tag.get("negative_case", 0) >= 5, by_tag.get("negative_case", 0)


def test_real_corpus_count_at_least_5() -> None:
    """At least 5 real-corpus Qs per Phase-3.2 plan (so locally a maintainer
    can reproduce the real-corpus retrieval numbers with confidence)."""
    qs = _load_questions()
    n_real = sum(1 for q in qs if q.get("requires_full_corpus"))
    assert n_real >= 5, n_real


def test_id_format_is_qNNN() -> None:
    import re

    pat = re.compile(r"^q[0-9]{3}$")
    for q in _load_questions():
        assert pat.match(q["id"]), q["id"]


@pytest.mark.parametrize("field", ["question", "rationale", "expected_doc_id"])
def test_required_string_fields_non_empty(field: str) -> None:
    for q in _load_questions():
        assert q[field].strip(), (q["id"], field)
