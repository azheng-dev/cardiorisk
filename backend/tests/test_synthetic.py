"""Tests for cardiorisk.data.synthetic."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cardiorisk.data.synthetic import (
    CHEST_PAIN_VALUES,
    COLUMNS,
    EXERCISE_ANGINA_VALUES,
    RESTING_ECG_VALUES,
    SEX_VALUES,
    ST_SLOPE_VALUES,
    generate_fixture,
    write_csv,
)


def test_generate_fixture_is_deterministic_for_same_seed() -> None:
    a = generate_fixture(n=20, seed=12345)
    b = generate_fixture(n=20, seed=12345)
    assert a == b


def test_generate_fixture_differs_for_different_seeds() -> None:
    a = generate_fixture(n=20, seed=1)
    b = generate_fixture(n=20, seed=2)
    assert a != b


def test_generate_fixture_row_count() -> None:
    rows = generate_fixture(n=37, seed=42)
    assert len(rows) == 37


def test_generate_fixture_rejects_invalid_n() -> None:
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        generate_fixture(n=0, seed=42)


def test_generate_fixture_columns_match_schema() -> None:
    rows = generate_fixture(n=5, seed=42)
    for row in rows:
        assert tuple(row.keys()) == COLUMNS


def test_categorical_values_in_allowed_set() -> None:
    rows = generate_fixture(n=200, seed=42)
    sex_set = {r["Sex"] for r in rows}
    cp_set = {r["ChestPainType"] for r in rows}
    ecg_set = {r["RestingECG"] for r in rows}
    angina_set = {r["ExerciseAngina"] for r in rows}
    slope_set = {r["ST_Slope"] for r in rows}

    assert sex_set <= set(SEX_VALUES)
    assert cp_set <= set(CHEST_PAIN_VALUES)
    assert ecg_set <= set(RESTING_ECG_VALUES)
    assert angina_set <= set(EXERCISE_ANGINA_VALUES)
    assert slope_set <= set(ST_SLOPE_VALUES)


def test_target_is_binary_zero_or_one() -> None:
    rows = generate_fixture(n=100, seed=42)
    targets = {r["HeartDisease"] for r in rows}
    assert targets <= {0, 1}


def test_cholesterol_zero_appears_at_some_seeds() -> None:
    rows = generate_fixture(n=500, seed=42)
    n_zero = sum(1 for r in rows if r["Cholesterol"] == 0)
    assert n_zero > 0, "expected at least some chol=0 rows to exercise the cleaning path"


def test_write_csv_round_trip(tmp_path: Path) -> None:
    rows = generate_fixture(n=10, seed=99)
    out = tmp_path / "out.csv"

    write_csv(rows=rows, out_path=out)

    with out.open() as fh:
        reader = csv.DictReader(fh)
        loaded = list(reader)

    assert len(loaded) == 10
    assert tuple(reader.fieldnames or ()) == COLUMNS
    for original, on_disk in zip(rows, loaded, strict=True):
        for col in COLUMNS:
            assert str(original[col]) == on_disk[col]


def test_write_csv_creates_parent_dir(tmp_path: Path) -> None:
    rows = generate_fixture(n=2, seed=1)
    nested = tmp_path / "deep" / "nested" / "out.csv"
    write_csv(rows=rows, out_path=nested)
    assert nested.exists()
