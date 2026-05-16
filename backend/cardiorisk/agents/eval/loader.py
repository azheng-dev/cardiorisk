"""Load + JSON-Schema-validate the agent-eval case set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from cardiorisk.agents.state import PatientInput

#: Repo-relative path to the case set. Set explicitly so test code can
#: pass an alternative path without monkey-patching.
DEFAULT_CASES_PATH = Path("eval/agents/cases.jsonl")
DEFAULT_SCHEMA_PATH = Path("eval/agents/schema.json")


@dataclass(frozen=True)
class AgentEvalCase:
    """One Phase-4 agent-eval case (validated).

    Phase 6 added :attr:`expected_recommendation_family`. Pre-Phase-6
    rows that lack the field fall through to ``"statin_consider"`` as
    the most-conservative default so old fixtures keep parsing.
    """

    id: str
    patient: PatientInput
    expected_risk_band: str  # "low" | "intermediate" | "high"
    expected_min_verified_claims: int
    expected_letter_min_words: int
    expected_sanity_flags: tuple[str, ...]
    tag: str
    rationale: str
    # Phase-6 addition; default keeps old test fixtures parsing without
    # construction changes. The loader always supplies an explicit value.
    expected_recommendation_family: str = "statin_consider"


def _validate_row(row: dict[str, Any], schema: dict[str, Any]) -> None:
    jsonschema.validate(row, schema)


def load_cases(
    cases_path: Path | None = None,
    schema_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    tag_filter: str | None = None,
    limit: int | None = None,
) -> list[AgentEvalCase]:
    """Load + JSON-Schema-validate the cases JSONL.

    Args:
        cases_path: Override the default ``eval/agents/cases.jsonl``.
        schema_path: Override the default ``eval/agents/schema.json``.
        repo_root: Resolve the default paths against this root. If
            unset, uses the conventional ``backend/`` parent so the
            CLI works whether invoked from the repo root or from
            ``backend/``.
        tag_filter: If set, keep only cases whose ``tag`` matches.
        limit: If set, keep only the first ``limit`` cases (post-tag-filter).

    Returns:
        The validated cases in file order. Order matters because the
        per-case report is written in the same order.
    """
    root = repo_root or Path(__file__).resolve().parents[4]
    cases_path = cases_path or (root / DEFAULT_CASES_PATH)
    schema_path = schema_path or (root / DEFAULT_SCHEMA_PATH)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    cases: list[AgentEvalCase] = []
    with cases_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate_row(row, schema)
            if tag_filter is not None and row["tag"] != tag_filter:
                continue
            cases.append(
                AgentEvalCase(
                    id=row["id"],
                    patient=PatientInput(**row["patient"]),
                    expected_risk_band=row["expected_risk_band"],
                    expected_min_verified_claims=row.get("expected_min_verified_claims", 1),
                    expected_letter_min_words=row.get("expected_letter_min_words", 60),
                    expected_sanity_flags=tuple(row.get("expected_sanity_flags", ())),
                    expected_recommendation_family=row.get(
                        "expected_recommendation_family", "statin_consider"
                    ),
                    tag=row["tag"],
                    rationale=row["rationale"],
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


__all__ = [
    "DEFAULT_CASES_PATH",
    "DEFAULT_SCHEMA_PATH",
    "AgentEvalCase",
    "load_cases",
]
