"""Load + JSON-Schema-validate ``eval/retrieval/questions.jsonl``.

The schema lives at ``eval/retrieval/schema.json`` and is enforced
in CI by :mod:`tests.test_rag_ingest_eval_schema`. The loader
re-validates at runtime so a developer running the eval locally
gets a clear failure if their working copy of ``questions.jsonl``
is malformed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from cardiorisk.data.paths import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval" / "retrieval"
SCHEMA_PATH = EVAL_DIR / "schema.json"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"


@dataclass(frozen=True)
class EvalQuestion:
    """One row of the retrieval eval set.

    ``expected_no_hit=True`` rows have inverted scoring: a "hit"
    means **no** top-k chunk contained all keywords. ``doc_id`` and
    ``page_range`` are sentinel fields the scorer ignores in that
    case.
    """

    id: str
    question: str
    expected_doc_id: str
    expected_page_range: tuple[int, int]
    expected_span_keywords: tuple[str, ...]
    rationale: str
    source_phase: str
    requires_full_corpus: bool
    expected_no_hit: bool
    tags: tuple[str, ...]


def _coerce(row: dict[str, Any]) -> EvalQuestion:
    return EvalQuestion(
        id=str(row["id"]),
        question=str(row["question"]),
        expected_doc_id=str(row["expected_doc_id"]),
        expected_page_range=(
            int(row["expected_page_range"][0]),
            int(row["expected_page_range"][1]),
        ),
        expected_span_keywords=tuple(str(k) for k in row["expected_span_keywords"]),
        rationale=str(row["rationale"]),
        source_phase=str(row["source_phase"]),
        requires_full_corpus=bool(row.get("requires_full_corpus", False)),
        expected_no_hit=bool(row.get("expected_no_hit", False)),
        tags=tuple(str(t) for t in row.get("tags", [])),
    )


def _is_fixture_question(q: EvalQuestion) -> bool:
    """A fixture question is one whose ``expected_doc_id`` belongs to the
    Phase-3.1 markdown fixture (``fixture_*``).

    Real-corpus questions reference RACGP / NVDPA ``doc_id``s defined in
    :mod:`cardiorisk.rag.ingest.sources`. Identifying fixture rows by
    the ``fixture_`` doc_id prefix avoids adding yet another schema
    field; the prefix is a stable convention enforced by the fixture's
    ``sources.json``.
    """
    return q.expected_doc_id.startswith("fixture_")


def load_questions(
    *,
    questions_path: Path = QUESTIONS_PATH,
    schema_path: Path = SCHEMA_PATH,
    skip_full_corpus: bool = False,
    skip_fixture: bool = False,
) -> list[EvalQuestion]:
    """Load + validate ``questions.jsonl``.

    Args:
        questions_path: Override for testing.
        schema_path: Override for testing.
        skip_full_corpus: If true, drop rows with
            ``requires_full_corpus: true`` (used by the CI smoke and by
            the orchestrator when running against the markdown fixture).
        skip_fixture: If true, drop fixture rows (rows whose
            ``expected_doc_id`` starts with ``fixture_``). Used when
            running the eval against the real RACGP / NVDPA corpus so
            the headline metrics aren't dominated by guaranteed misses
            from fixture-targeted questions.

    Returns:
        A list of :class:`EvalQuestion`.

    Raises:
        :class:`jsonschema.ValidationError`: if any row fails the schema.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    out: list[EvalQuestion] = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validator.validate(row)
        question = _coerce(row)
        if skip_full_corpus and question.requires_full_corpus:
            continue
        if skip_fixture and _is_fixture_question(question):
            continue
        out.append(question)
    return out
