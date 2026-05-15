"""Tests for the generation eval-set loader."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from cardiorisk.rag.eval_generation.loader import EvalCase, load_cases


def test_default_cases_load_and_validate() -> None:
    cases = load_cases()
    assert len(cases) >= 30
    assert all(isinstance(c, EvalCase) for c in cases)
    refusal = [c for c in cases if c.should_refuse]
    assert len(refusal) == 6
    for c in refusal:
        assert c.expected_doc_ids == ()
        assert c.expected_keywords == ()


def test_skip_full_corpus_drops_real_corpus_cases() -> None:
    cases = load_cases(skip_full_corpus=True)
    assert all(not c.requires_full_corpus for c in cases)


def test_skip_fixture_keeps_refusal_cases() -> None:
    cases = load_cases(skip_fixture=True)
    refusal_count = sum(1 for c in cases if c.should_refuse)
    assert refusal_count == 6, "refusal cases must survive skip_fixture (no doc dependency)"


def test_invalid_row_raises_jsonschema_error(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string", "pattern": "^g[0-9]{3}$"}},
            }
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({"id": "BAD-ID"}) + "\n")
    with pytest.raises(jsonschema.ValidationError):
        load_cases(cases_path=cases_path, schema_path=schema_path)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "question",
                    "expected_doc_ids",
                    "expected_keywords",
                    "rationale",
                    "source_phase",
                    "tags",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^g[0-9]{3}$"},
                    "question": {"type": "string"},
                    "expected_doc_ids": {"type": "array", "items": {"type": "string"}},
                    "expected_keywords": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                    "source_phase": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            }
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "g001",
                "question": "Q?",
                "expected_doc_ids": [],
                "expected_keywords": [],
                "rationale": "rationale text",
                "source_phase": "3.3",
                "tags": ["risk_assessment"],
            }
        )
        + "\n\n"
    )
    cases = load_cases(cases_path=cases_path, schema_path=schema_path)
    assert len(cases) == 1
