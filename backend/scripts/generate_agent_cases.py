"""CLI: regenerate ``eval/agents/cases.jsonl`` (Phase 6, 100 cases).

This is a one-shot generator. Run it whenever the schema changes or new
cases need adding. It:

1. Reads the existing ``cases.jsonl`` (a001..aNNN already on disk).
2. Backfills the new Phase-6 ``expected_recommendation_family`` field on
   every existing row by mapping ``(tag, expected_risk_band)`` to a
   recommendation family (the same mapping rule the scorer uses).
3. Appends new deterministic synthetic cases up to ``--total`` (default
   100), drawing from band-specific parameter pools so the distribution
   is the same across runs.
4. Validates every row against ``eval/agents/schema.json`` (Phase-6
   bumped) and rewrites the file in place.

The case generation is **fully deterministic** — given the same seed,
the same file contents come out byte-for-byte. The eval set is
locked from the moment it lands on main; later phases append, never
recycle.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

DEFAULT_CASES_PATH = Path("eval/agents/cases.jsonl")
DEFAULT_SCHEMA_PATH = Path("eval/agents/schema.json")
DEFAULT_TOTAL = 100
DEFAULT_SEED = 20260516


# ---------------------------------------------------------------------------
# Tag -> recommendation-family mapping (binding rule; ADR-019).
# ---------------------------------------------------------------------------
# A case's expected_recommendation_family is a function of (tag, band).
# The Phase 6 scorer checks the letter draft against the keyword family
# for the *expected* family. This mapping is intentionally conservative:
# the model should err on the side of including statin + BP guidance for
# high-risk cases, lifestyle-only for low-risk, and acknowledge
# refusals when the input is too sparse to recommend anything.
RECO_MAP: dict[tuple[str, str], str] = {
    ("high_risk", "high"): "statin_plus_bp",
    ("intermediate_risk", "intermediate"): "statin_consider",
    ("low_risk", "low"): "lifestyle_only",
    ("borderline", "intermediate"): "lifestyle_plus_review",
    ("borderline", "low"): "lifestyle_plus_review",
    ("borderline", "high"): "statin_consider",
    ("data_quality", "high"): "statin_plus_bp",
    ("data_quality", "intermediate"): "statin_consider",
    ("data_quality", "low"): "lifestyle_only",
    ("extreme_case", "high"): "statin_plus_bp_plus_referral",
    ("extreme_case", "intermediate"): "statin_consider",
    ("refusal", "intermediate"): "refusal_no_recommendation",
    ("refusal", "low"): "refusal_no_recommendation",
    ("refusal", "high"): "refusal_no_recommendation",
}


def _reco_family(tag: str, band: str) -> str:
    return RECO_MAP.get((tag, band), "statin_consider")


# ---------------------------------------------------------------------------
# Band-specific synthetic patient pools.
# ---------------------------------------------------------------------------
# Each pool is a dict of feature -> list of plausible values for that
# band. The generator draws one value per feature; the band invariant
# is preserved by construction (high-risk pools have high RestingBP,
# low-risk pools have clean ECG, etc.).
#
# The pools were calibrated against the Phase 2.5 SHAP attribution
# headline (ChestPainType, ST_Slope, ExerciseAngina, Oldpeak, Age as
# the top drivers) so the synthetic distribution exercises those
# features at clinically-realistic magnitudes.


@dataclass(frozen=True)
class BandPool:
    age: list[int]
    sex: list[str]
    chest_pain: list[str]
    resting_bp: list[int]
    cholesterol: list[int]
    fasting_bs: list[int]
    resting_ecg: list[str]
    max_hr: list[int]
    exercise_angina: list[str]
    oldpeak: list[float]
    st_slope: list[str]


HIGH_POOL = BandPool(
    age=list(range(58, 85)),
    sex=["M"] * 8 + ["F"] * 2,
    chest_pain=["ASY"] * 7 + ["NAP"] * 3,
    resting_bp=list(range(140, 175)),
    cholesterol=list(range(240, 320)),
    fasting_bs=[0, 1, 1, 1],
    resting_ecg=["ST", "LVH", "ST", "LVH", "Normal"],
    max_hr=list(range(85, 130)),
    exercise_angina=["Y"] * 8 + ["N"] * 2,
    oldpeak=[1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
    st_slope=["Flat", "Down", "Down", "Flat"],
)

INTERMEDIATE_POOL = BandPool(
    age=list(range(45, 70)),
    sex=["M"] * 7 + ["F"] * 3,
    chest_pain=["NAP", "ATA", "NAP", "ATA"],
    resting_bp=list(range(125, 150)),
    cholesterol=list(range(195, 245)),
    fasting_bs=[0, 0, 0, 1],
    resting_ecg=["Normal", "ST", "Normal", "LVH"],
    max_hr=list(range(125, 155)),
    exercise_angina=["N"] * 8 + ["Y"] * 2,
    oldpeak=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    st_slope=["Up", "Flat", "Flat", "Flat"],
)

LOW_POOL = BandPool(
    age=list(range(28, 50)),
    sex=["M"] * 4 + ["F"] * 6,
    chest_pain=["ATA", "NAP", "ATA"],
    resting_bp=list(range(100, 130)),
    cholesterol=list(range(165, 210)),
    fasting_bs=[0, 0, 0, 0, 0],
    resting_ecg=["Normal"] * 9 + ["ST"],
    max_hr=list(range(150, 185)),
    exercise_angina=["N"] * 10,
    oldpeak=[0.0, 0.0, 0.2, 0.4, 0.6],
    st_slope=["Up", "Up", "Up", "Flat"],
)


def _draw(pool: BandPool, rng: random.Random) -> dict[str, Any]:
    return {
        "Age": rng.choice(pool.age),
        "Sex": rng.choice(pool.sex),
        "ChestPainType": rng.choice(pool.chest_pain),
        "RestingBP": rng.choice(pool.resting_bp),
        "Cholesterol": rng.choice(pool.cholesterol),
        "FastingBS": rng.choice(pool.fasting_bs),
        "RestingECG": rng.choice(pool.resting_ecg),
        "MaxHR": rng.choice(pool.max_hr),
        "ExerciseAngina": rng.choice(pool.exercise_angina),
        # round to one decimal place; the schema accepts float and
        # downstream code formats with one decimal anyway.
        "Oldpeak": round(rng.choice(pool.oldpeak), 1),
        "ST_Slope": rng.choice(pool.st_slope),
    }


# ---------------------------------------------------------------------------
# Target distribution (binding for v1; locked once landed on main).
# Total = 100.
# ---------------------------------------------------------------------------
TARGET_DISTRIBUTION: list[tuple[str, str, int]] = [
    # (tag, band, count)
    ("high_risk", "high", 25),
    ("intermediate_risk", "intermediate", 25),
    ("low_risk", "low", 25),
    ("borderline", "intermediate", 7),
    ("borderline", "low", 3),
    ("data_quality", "high", 2),
    ("data_quality", "intermediate", 4),
    ("data_quality", "low", 2),
    ("extreme_case", "high", 4),
    ("refusal", "intermediate", 3),
]

_PatientMutator = Callable[[dict[str, Any]], dict[str, Any]]

# Sanity-flag injection rules for data_quality cases.
DATA_QUALITY_INJECTIONS: list[tuple[str, _PatientMutator]] = [
    ("cholesterol_missing_sentinel", lambda p: {**p, "Cholesterol": 0}),
    ("resting_bp_extreme", lambda p: {**p, "RestingBP": 80}),
    ("oldpeak_negative", lambda p: {**p, "Oldpeak": -0.4}),
    ("age_outside_training_range", lambda p: {**p, "Age": 27}),
    ("max_hr_outside_training_range", lambda p: {**p, "MaxHR": 55}),
]


def _existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _backfill_recommendation(row: dict[str, Any]) -> dict[str, Any]:
    if "expected_recommendation_family" in row:
        return row
    family = _reco_family(row["tag"], row["expected_risk_band"])
    return {**row, "expected_recommendation_family": family}


def _next_id(existing: list[dict[str, Any]]) -> int:
    if not existing:
        return 1
    max_id = max(int(r["id"][1:]) for r in existing)
    return max_id + 1


def _build_new_case(
    *,
    case_id: str,
    tag: str,
    band: str,
    rng: random.Random,
) -> dict[str, Any]:
    pool = {"high": HIGH_POOL, "intermediate": INTERMEDIATE_POOL, "low": LOW_POOL}[band]
    patient = _draw(pool, rng)

    expected_sanity_flags: list[str] = []
    rationale = ""

    if tag == "data_quality":
        flag, mutate = rng.choice(DATA_QUALITY_INJECTIONS)
        patient = mutate(patient)
        expected_sanity_flags = [flag]
        rationale = (
            f"Synthetic case generated for Phase 6 data-quality cell ({flag}). "
            "Triage agent must surface the flag; risk model still computes a "
            "probability against the imputed feature set."
        )
    elif tag == "extreme_case":
        patient = {**patient, "Age": rng.choice([78, 80, 82, 84]), "Oldpeak": 3.4}
        expected_sanity_flags = ["age_outside_training_range"]
        rationale = (
            "Synthetic extreme case generated for Phase 6: every risk driver "
            "maxed out, age extrapolating beyond the training distribution."
        )
    elif tag == "refusal":
        # Refusal cases keep a plausible patient but the question /
        # downstream context forces a refusal. The scorer treats the
        # letter draft as 'refusal_no_recommendation' for these.
        rationale = (
            "Synthetic refusal case generated for Phase 6: the patient "
            "profile is plausible, but the guideline question is one the "
            "Mock LLM is configured to refuse on (zero retrieved chunks). "
            "Exercises the refusal-sentinel path end-to-end."
        )
    elif tag == "borderline":
        rationale = (
            f"Synthetic borderline case generated for Phase 6 ({band} band, "
            "near the AusCVDRisk 5% / 10% cut-points). Tests the edge of "
            "the calibrated model."
        )
    else:
        rationale = (
            f"Synthetic {tag} case generated deterministically for Phase 6 "
            f"({band} band). Drawn from the band-specific parameter pool."
        )

    row: dict[str, Any] = {
        "id": case_id,
        "patient": patient,
        "expected_risk_band": band,
        "tag": tag,
        "rationale": rationale,
    }
    if expected_sanity_flags:
        row["expected_sanity_flags"] = expected_sanity_flags
    if tag == "refusal":
        # Refusal cases relax the verified-claim floor and the letter
        # word-count floor: a refusal draft is intentionally short.
        row["expected_min_verified_claims"] = 0
        row["expected_letter_min_words"] = 20
    row["expected_recommendation_family"] = _reco_family(tag, band)
    return row


def _generate_new_cases(
    *,
    existing: list[dict[str, Any]],
    target: list[tuple[str, str, int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    # Count what we already have per (tag, band).
    have: dict[tuple[str, str], int] = {}
    for row in existing:
        key = (row["tag"], row["expected_risk_band"])
        have[key] = have.get(key, 0) + 1

    new: list[dict[str, Any]] = []
    next_idx = _next_id(existing)
    for tag, band, count in target:
        need = max(0, count - have.get((tag, band), 0))
        for _ in range(need):
            case_id = f"a{next_idx:03d}"
            next_idx += 1
            new.append(_build_new_case(case_id=case_id, tag=tag, band=band, rng=rng))
    return new


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the on-disk file against the schema without rewriting.",
    )
    args = parser.parse_args()

    schema = json.loads(args.schema_path.read_text(encoding="utf-8"))

    if args.check_only:
        rows = _existing_rows(args.cases_path)
        for r in rows:
            jsonschema.validate(r, schema)
        print(f"validated {len(rows)} rows against {args.schema_path}")
        return 0

    rng = random.Random(args.seed)

    existing = _existing_rows(args.cases_path)
    backfilled = [_backfill_recommendation(r) for r in existing]

    new_rows = _generate_new_cases(
        existing=backfilled,
        target=TARGET_DISTRIBUTION,
        rng=rng,
    )

    all_rows = backfilled + new_rows
    if len(all_rows) > args.total:
        all_rows = all_rows[: args.total]

    for r in all_rows:
        jsonschema.validate(r, schema)

    # Write out canonicalised JSONL (one row per line, no trailing space,
    # newline at EOF) so the on-disk file is reproducible.
    args.cases_path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in all_rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(all_rows)} cases to {args.cases_path} "
        f"({len(existing)} kept + backfilled, {len(new_rows)} new; "
        f"seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
