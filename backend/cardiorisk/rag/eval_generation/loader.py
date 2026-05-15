"""Load + JSON-Schema-validate ``eval/generation/cases.jsonl``.

Mirrors :mod:`cardiorisk.rag.eval_retrieval.loader` to keep the two
eval surfaces ergonomically identical for callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from cardiorisk.data.paths import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval" / "generation"
SCHEMA_PATH = EVAL_DIR / "schema.json"
CASES_PATH = EVAL_DIR / "cases.jsonl"


@dataclass(frozen=True)
class EvalCase:
    """One row of the generation eval set."""

    id: str
    question: str
    expected_doc_ids: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    rationale: str
    source_phase: str
    requires_full_corpus: bool
    should_refuse: bool
    tags: tuple[str, ...]


def _coerce(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        id=str(row["id"]),
        question=str(row["question"]),
        expected_doc_ids=tuple(str(d) for d in row.get("expected_doc_ids", [])),
        expected_keywords=tuple(str(k) for k in row.get("expected_keywords", [])),
        rationale=str(row["rationale"]),
        source_phase=str(row["source_phase"]),
        requires_full_corpus=bool(row.get("requires_full_corpus", False)),
        should_refuse=bool(row.get("should_refuse", False)),
        tags=tuple(str(t) for t in row["tags"]),
    )


def _is_fixture_case(case: EvalCase) -> bool:
    """Treat a case as fixture-only when every expected doc_id is fixture."""
    if not case.expected_doc_ids:
        return False
    return all(d.startswith("fixture_") for d in case.expected_doc_ids)


def load_cases(
    *,
    cases_path: Path = CASES_PATH,
    schema_path: Path = SCHEMA_PATH,
    skip_full_corpus: bool = False,
    skip_fixture: bool = False,
) -> list[EvalCase]:
    """Load + validate ``cases.jsonl``.

    Args:
        cases_path: Override for testing.
        schema_path: Override for testing.
        skip_full_corpus: If true, drop rows with
            ``requires_full_corpus: true`` (used by the CI smoke).
        skip_fixture: If true, drop rows whose expected doc_ids are
            all fixture rows. Refusal cases (which have no
            expected_doc_ids) are kept under both flags because they
            don't depend on any corpus.

    Returns:
        A list of :class:`EvalCase`.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    out: list[EvalCase] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validator.validate(row)
        case = _coerce(row)
        if skip_full_corpus and case.requires_full_corpus:
            continue
        if skip_fixture and _is_fixture_case(case):
            continue
        out.append(case)
    return out
