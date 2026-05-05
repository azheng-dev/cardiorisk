"""Deterministic synthetic-data generator matching the HFP schema.

Used to produce ``backend/tests/fixtures/hfp_mini.csv`` — the only tabular
data file committed to the public repo. Every row is fictitious; the PRNG
is seeded from the CLI default in ``scripts/generate_fixture.py``.

The fixture exists so CI can run end-to-end (fetch, combine, EDA) without
network access or Kaggle credentials. It is **not** training data and
**not** distributionally representative of the real HFP dataset.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Final

COLUMNS: Final[tuple[str, ...]] = (
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease",
)

SEX_VALUES: Final[tuple[str, ...]] = ("M", "F")
CHEST_PAIN_VALUES: Final[tuple[str, ...]] = ("TA", "ATA", "NAP", "ASY")
RESTING_ECG_VALUES: Final[tuple[str, ...]] = ("Normal", "ST", "LVH")
EXERCISE_ANGINA_VALUES: Final[tuple[str, ...]] = ("N", "Y")
ST_SLOPE_VALUES: Final[tuple[str, ...]] = ("Up", "Flat", "Down")


def _generate_row(rng: random.Random) -> dict[str, int | float | str]:
    """Generate one synthetic patient row.

    Target ``HeartDisease`` is weakly correlated with a small number of
    features so the fixture isn't pure noise — but the correlation is
    intentionally not calibrated to real HFP marginals. Don't report
    metrics on this fixture.
    """
    age = rng.randint(28, 80)
    sex = rng.choice(SEX_VALUES)
    chest_pain = rng.choices(
        CHEST_PAIN_VALUES,
        weights=(0.10, 0.20, 0.30, 0.40),
        k=1,
    )[0]
    resting_bp = rng.randint(95, 175)
    cholesterol = 0 if rng.random() < 0.10 else rng.randint(140, 320)
    fasting_bs = 1 if rng.random() < 0.20 else 0
    resting_ecg = rng.choices(RESTING_ECG_VALUES, weights=(0.55, 0.20, 0.25), k=1)[0]
    max_hr = rng.randint(80, 195)
    exercise_angina = rng.choices(EXERCISE_ANGINA_VALUES, weights=(0.65, 0.35), k=1)[0]
    oldpeak = round(rng.uniform(-1.0, 5.5), 1)
    st_slope = rng.choices(ST_SLOPE_VALUES, weights=(0.40, 0.45, 0.15), k=1)[0]

    risk_score = (
        (age - 28) / 52 * 0.35
        + (1.0 if chest_pain == "ASY" else 0.0) * 0.20
        + (1.0 if exercise_angina == "Y" else 0.0) * 0.15
        + max(0.0, oldpeak / 5.5) * 0.20
        + (1.0 if st_slope == "Flat" else 0.0) * 0.10
    )
    heart_disease = 1 if rng.random() < risk_score else 0

    return {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
        "HeartDisease": heart_disease,
    }


def generate_fixture(n: int, seed: int) -> list[dict[str, int | float | str]]:
    """Generate ``n`` deterministic synthetic rows."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    # Deterministic synthetic-data generation, not a cryptographic context.
    rng = random.Random(seed)  # noqa: S311
    return [_generate_row(rng) for _ in range(n)]


def write_csv(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    """Write rows to ``out_path`` in canonical column order."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
