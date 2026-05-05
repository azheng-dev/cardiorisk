"""Tests for cardiorisk.data.combine.

Exercises the UCI-to-HFP schema mapping and the fixture-based combine path
so CI can run without network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cardiorisk.data.combine import (
    HFP_COLUMNS,
    UCI_COLUMNS,
    UCI_FILES,
    CombineError,
    build_from_fixture,
    build_from_uci,
    map_uci_to_hfp,
    read_uci_file,
    write_combined,
)

# ---------------------------------------------------------------- read_uci_file


def test_read_uci_file_reads_question_marks_as_nan(tmp_path: Path) -> None:
    src = tmp_path / "x.data"
    # 14 columns per UCI processed format. `?` in the chol slot.
    src.write_text("63,1,1,145,?,0,2,150,0,2.3,3,0,3,0\n", encoding="utf-8")

    df = read_uci_file(src)

    assert tuple(df.columns) == UCI_COLUMNS
    # pandas-stubs over-narrows `pd.isna(scalar)` to `Literal[True]` after a
    # truthy assert, marking the rest of the test as unreachable; ignored.
    assert pd.isna(df.loc[0, "chol"])
    # Numeric columns are read as strings (we coerce later in map_uci_to_hfp).
    assert df.loc[0, "age"] == "63"  # type: ignore[unreachable]


# ---------------------------------------------------------------- map_uci_to_hfp


def _uci_frame_from_rows(rows: list[list[object]]) -> pd.DataFrame:
    """Build a UCI-format DataFrame from a list of 14-element row lists."""
    return pd.DataFrame(rows, columns=list(UCI_COLUMNS), dtype=str)


def test_map_uci_to_hfp_translates_categoricals() -> None:
    uci = _uci_frame_from_rows(
        [
            ["63", "1", "1", "145", "240", "0", "2", "150", "0", "2.3", "3", "0", "3", "0"],
            ["50", "0", "4", "120", "200", "1", "0", "175", "1", "0.0", "1", "0", "3", "2"],
        ]
    )
    out = map_uci_to_hfp(uci, source_name="testsrc")

    expected_cols = (*HFP_COLUMNS, "source")
    assert tuple(out.columns) == expected_cols

    assert list(out["Sex"]) == ["M", "F"]
    assert list(out["ChestPainType"]) == ["TA", "ASY"]
    assert list(out["RestingECG"]) == ["LVH", "Normal"]
    assert list(out["ExerciseAngina"]) == ["N", "Y"]
    assert list(out["ST_Slope"]) == ["Down", "Up"]
    assert list(out["source"]) == ["testsrc", "testsrc"]


def test_map_uci_to_hfp_binarises_target() -> None:
    uci = _uci_frame_from_rows(
        [
            ["50", "1", "1", "120", "200", "0", "0", "150", "0", "1.0", "1", "0", "3", "0"],
            ["50", "1", "1", "120", "200", "0", "0", "150", "0", "1.0", "1", "0", "3", "1"],
            ["50", "1", "1", "120", "200", "0", "0", "150", "0", "1.0", "1", "0", "3", "4"],
        ]
    )
    out = map_uci_to_hfp(uci, source_name="testsrc")
    assert list(out["HeartDisease"]) == [0, 1, 1]


def test_map_uci_to_hfp_preserves_nan_in_features() -> None:
    uci = _uci_frame_from_rows(
        [["63", "1", "1", "145", None, "0", "2", "150", "0", "2.3", "3", "0", "3", "0"]],
    )
    out = map_uci_to_hfp(uci, source_name="testsrc")
    assert pd.isna(out.loc[0, "Cholesterol"])


def test_map_uci_to_hfp_unknown_categorical_becomes_nan() -> None:
    """A UCI categorical value outside the documented range maps to NaN, not a crash."""
    uci = _uci_frame_from_rows(
        [["63", "9", "9", "145", "240", "0", "2", "150", "0", "2.3", "3", "0", "3", "0"]],
    )
    out = map_uci_to_hfp(uci, source_name="testsrc")
    assert pd.isna(out.loc[0, "Sex"])
    assert pd.isna(out.loc[0, "ChestPainType"])  # type: ignore[unreachable]


# ---------------------------------------------------------------- build_from_uci


def _write_minimal_uci_file(path: Path, n_rows: int = 3) -> None:
    """Write `n_rows` of plausible UCI-format data to `path`."""
    rows = []
    for i in range(n_rows):
        rows.append(
            f"{50 + i},{i % 2},{(i % 4) + 1},{120 + i},{200 + i},"
            f"{i % 2},{i % 3},{150 - i},{i % 2},"
            f"{i * 0.5},{(i % 3) + 1},0,3,{1 if i % 2 else 0}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_from_uci_combines_all_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for filename in UCI_FILES.values():
        _write_minimal_uci_file(raw_dir / filename, n_rows=3)

    combined = build_from_uci(raw_dir=raw_dir)

    assert tuple(combined.columns) == (*HFP_COLUMNS, "source")
    assert len(combined) == 3 * len(UCI_FILES)
    assert set(combined["source"]) == set(UCI_FILES.keys())


def test_build_from_uci_raises_on_missing_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(CombineError, match="missing UCI files"):
        build_from_uci(raw_dir=raw_dir)


# ---------------------------------------------------------------- build_from_fixture


def test_build_from_fixture_tags_source_as_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "f.csv"
    fixture.write_text(
        "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,"
        "MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease\n"
        "50,M,ASY,120,200,0,Normal,150,N,1.0,Up,1\n",
        encoding="utf-8",
    )
    df = build_from_fixture(fixture_path=fixture)

    assert tuple(df.columns) == (*HFP_COLUMNS, "source")
    assert list(df["source"]) == ["fixture"]


def test_build_from_fixture_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(CombineError, match="fixture not found"):
        build_from_fixture(fixture_path=tmp_path / "absent.csv")


# ---------------------------------------------------------------- write_combined


def test_write_combined_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Age": [50, 60],
            "Sex": ["M", "F"],
            "ChestPainType": ["ASY", "TA"],
            "RestingBP": [120, 140],
            "Cholesterol": [200, 240],
            "FastingBS": [0, 1],
            "RestingECG": ["Normal", "LVH"],
            "MaxHR": [150, 130],
            "ExerciseAngina": ["N", "Y"],
            "Oldpeak": [1.0, 2.0],
            "ST_Slope": ["Up", "Flat"],
            "HeartDisease": [0, 1],
            "source": ["X", "Y"],
        }
    )
    out = tmp_path / "combined.parquet"
    write_combined(df, out)

    loaded = pd.read_parquet(out)
    assert list(loaded["source"]) == ["X", "Y"]
    assert list(loaded["HeartDisease"]) == [0, 1]
