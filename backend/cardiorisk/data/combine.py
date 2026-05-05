"""Combine the four UCI Heart Disease subsets into the HFP-schema DataFrame.

Maps the 14-column UCI processed format to the 11-feature + target schema
used by Kaggle's Heart Failure Prediction dataset (fedesoriano), then tags
each row with a ``source`` column for LODO-CV in Phase 2.3.

Column mapping (UCI processed -> HFP):

    age        -> Age              (numeric)
    sex        -> Sex              (1=M, 0=F)
    cp         -> ChestPainType    (1=TA, 2=ATA, 3=NAP, 4=ASY)
    trestbps   -> RestingBP        (numeric)
    chol       -> Cholesterol      (numeric; 0 marks missing in HFP convention)
    fbs        -> FastingBS        (0/1)
    restecg    -> RestingECG       (0=Normal, 1=ST, 2=LVH)
    thalach    -> MaxHR            (numeric)
    exang      -> ExerciseAngina   (0=N, 1=Y)
    oldpeak    -> Oldpeak          (numeric)
    slope      -> ST_Slope         (1=Up, 2=Flat, 3=Down)
    num        -> HeartDisease     (0->0; 1..4 -> 1)

``ca`` and ``thal`` are dropped (not in HFP). UCI's ``?`` missing-value
sentinel is read as NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from .paths import DATA_PROCESSED, DATA_RAW, FIXTURE_PATH

UCI_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)

UCI_FILES: Final[dict[str, str]] = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "LongBeachVA": "processed.va.data",
}

SEX_MAP: Final[dict[float, str]] = {1.0: "M", 0.0: "F"}
CHEST_PAIN_MAP: Final[dict[float, str]] = {1.0: "TA", 2.0: "ATA", 3.0: "NAP", 4.0: "ASY"}
RESTING_ECG_MAP: Final[dict[float, str]] = {0.0: "Normal", 1.0: "ST", 2.0: "LVH"}
EXERCISE_ANGINA_MAP: Final[dict[float, str]] = {0.0: "N", 1.0: "Y"}
ST_SLOPE_MAP: Final[dict[float, str]] = {1.0: "Up", 2.0: "Flat", 3.0: "Down"}

HFP_COLUMNS: Final[tuple[str, ...]] = (
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

DEFAULT_OUTPUT: Final[Path] = DATA_PROCESSED / "combined.parquet"


class CombineError(RuntimeError):
    """Raised when input files are missing or malformed."""


def read_uci_file(path: Path) -> pd.DataFrame:
    """Read one UCI ``processed.X.data`` file with ``?`` interpreted as NaN."""
    return pd.read_csv(
        path,
        header=None,
        names=list(UCI_COLUMNS),
        na_values=["?"],
        dtype=str,
    )


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a string-typed column to float, leaving NaN as NaN."""
    return pd.to_numeric(series, errors="coerce")


def map_uci_to_hfp(uci_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Map a UCI-format DataFrame to the HFP schema and tag with ``source``."""
    age = _to_numeric(uci_df["age"]).round().astype("Int64")
    sex_num = _to_numeric(uci_df["sex"])
    cp_num = _to_numeric(uci_df["cp"])
    trestbps = _to_numeric(uci_df["trestbps"])
    chol = _to_numeric(uci_df["chol"])
    fbs = _to_numeric(uci_df["fbs"]).astype("Int64")
    restecg_num = _to_numeric(uci_df["restecg"])
    thalach = _to_numeric(uci_df["thalach"])
    exang_num = _to_numeric(uci_df["exang"])
    oldpeak = _to_numeric(uci_df["oldpeak"])
    slope_num = _to_numeric(uci_df["slope"])
    num = _to_numeric(uci_df["num"])

    return pd.DataFrame(
        {
            "Age": age,
            "Sex": sex_num.map(SEX_MAP),
            "ChestPainType": cp_num.map(CHEST_PAIN_MAP),
            "RestingBP": trestbps,
            "Cholesterol": chol,
            "FastingBS": fbs,
            "RestingECG": restecg_num.map(RESTING_ECG_MAP),
            "MaxHR": thalach,
            "ExerciseAngina": exang_num.map(EXERCISE_ANGINA_MAP),
            "Oldpeak": oldpeak,
            "ST_Slope": slope_num.map(ST_SLOPE_MAP),
            "HeartDisease": (num.fillna(0) >= 1.0).astype("Int64"),
            "source": source_name,
        }
    )


def build_from_uci(
    *,
    raw_dir: Path = DATA_RAW,
    files: dict[str, str] = UCI_FILES,
) -> pd.DataFrame:
    """Combine the four UCI subsets into one HFP-schema DataFrame."""
    missing = [name for name, fn in files.items() if not (raw_dir / fn).exists()]
    if missing:
        raise CombineError(
            f"missing UCI files for: {missing}. "
            "Run `uv run python backend/scripts/fetch_hfp.py` first."
        )

    parts = [
        map_uci_to_hfp(read_uci_file(raw_dir / fn), source_name=name) for name, fn in files.items()
    ]
    combined = pd.concat(parts, ignore_index=True)
    expected_cols = (*HFP_COLUMNS, "source")
    if tuple(combined.columns) != expected_cols:
        raise CombineError(
            f"unexpected column order: got {tuple(combined.columns)}, want {expected_cols}"
        )
    return combined


def build_from_fixture(*, fixture_path: Path = FIXTURE_PATH) -> pd.DataFrame:
    """Load the synthetic fixture and tag every row with ``source='fixture'``."""
    if not fixture_path.exists():
        raise CombineError(
            f"fixture not found at {fixture_path}. "
            "Run `uv run python backend/scripts/generate_fixture.py` first."
        )
    df = pd.read_csv(fixture_path)
    df["source"] = "fixture"
    return df


def write_combined(df: pd.DataFrame, out_path: Path = DEFAULT_OUTPUT) -> None:
    """Write combined frame to parquet (faster + dtype-stable than CSV)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
