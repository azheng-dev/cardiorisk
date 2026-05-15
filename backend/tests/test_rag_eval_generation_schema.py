"""Phase 3.3: validate the generation eval-set against its JSON Schema."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
import pytest

from cardiorisk.data.paths import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval" / "generation"
SCHEMA_PATH = EVAL_DIR / "schema.json"
CASES_PATH = EVAL_DIR / "cases.jsonl"


def _load_cases() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_schema() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return payload


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists()


def test_cases_file_exists_and_non_empty() -> None:
    assert CASES_PATH.exists()
    assert _load_cases()


def test_schema_is_valid_json_schema() -> None:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_every_case_validates_against_schema() -> None:
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    for case in _load_cases():
        errors = sorted(validator.iter_errors(case), key=lambda e: e.path)
        assert not errors, f"{case['id']}: {[e.message for e in errors]}"


def test_case_ids_are_unique() -> None:
    ids = [c["id"] for c in _load_cases()]
    assert len(ids) == len(set(ids))


def test_id_format_is_gNNN() -> None:
    pat = re.compile(r"^g[0-9]{3}$")
    for c in _load_cases():
        assert pat.match(c["id"]), c["id"]


def test_phase_3_3_target_count_at_least_30() -> None:
    """Phase 3.3 ships 30 cases at sign-off; later phases may grow the
    set. Lock-in a floor rather than an exact count so adding a real-
    corpus case (Phase 3.3 amendment) does not spuriously break CI."""
    assert len(_load_cases()) >= 30


def test_phase_3_3_real_corpus_cases_target_known_doc_ids() -> None:
    """Real-corpus cases must reference the live RACGP / NVDPA doc_ids."""
    from cardiorisk.rag.ingest.sources import CORPUS_SOURCES

    real_ids = {s.doc_id for s in CORPUS_SOURCES}
    for c in _load_cases():
        if c.get("requires_full_corpus") and c.get("expected_doc_ids"):
            for did in c["expected_doc_ids"]:
                assert did in real_ids, (c["id"], did)


def test_refusal_cases_have_no_expected_doc_or_keywords() -> None:
    for c in _load_cases():
        if c.get("should_refuse"):
            assert c.get("expected_doc_ids") == [], c["id"]
            assert c.get("expected_keywords") == [], c["id"]


def test_positive_cases_have_at_least_one_expected_doc_and_keyword() -> None:
    for c in _load_cases():
        if c.get("should_refuse"):
            continue
        assert c.get("expected_doc_ids"), c["id"]
        assert c.get("expected_keywords"), c["id"]


def test_phase_3_3_distribution_includes_all_tags() -> None:
    by_tag: dict[str, int] = {}
    for c in _load_cases():
        for tag in c.get("tags", []):
            by_tag[tag] = by_tag.get(tag, 0) + 1
    for tag in (
        "risk_assessment",
        "pharmacotherapy",
        "lifestyle",
        "communication",
        "reclassifiers",
        "follow_up",
    ):
        assert by_tag.get(tag, 0) >= 1, (tag, by_tag.get(tag, 0))
    assert by_tag.get("refusal", 0) >= 5, by_tag.get("refusal", 0)


@pytest.mark.parametrize("field", ["question", "rationale"])
def test_required_string_fields_non_empty(field: str) -> None:
    for c in _load_cases():
        assert c[field].strip(), (c["id"], field)
