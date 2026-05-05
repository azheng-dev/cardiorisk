"""Deterministic, leakage-free cleaning steps applied before any sklearn pipeline.

These transformations are *pure functions* with no fit/transform state — they
depend only on the row's own values, so it is safe to apply them to *any*
slice (train, val, calibration, or test) without the LODO-CV leakage worry.
The stateful preprocessing (imputation, scaling, encoding) lives in
:mod:`cardiorisk.features.pipeline` and is fit per LODO fold.

Three cleaning operations, in this order:

1. :func:`clean_cholesterol_zero_to_nan` — convert the documented
   ``Cholesterol == 0`` sentinel to ``NaN`` so downstream imputers see
   real missingness. EDA findings §2.1 confirmed that 100% of Switzerland
   and 24.5% of Long Beach VA rows carry this sentinel.

2. :func:`add_missingness_indicators` — append a binary ``<col>_was_missing``
   column for each feature whose per-source missingness exceeded 10% in any
   source: ``RestingBP``, ``MaxHR``, ``ExerciseAngina``, ``Oldpeak``,
   ``ST_Slope``. This lets a downstream model learn that "missing X on a
   record from source Y" is informative (per EDA findings §3.1 item 2 and
   :doc:`../../docs/research/04-revised-design.md` §3.2).

3. :func:`replace_categorical_missing` — replace ``NaN`` in categorical
   columns with the literal string ``"Missing"`` so one-hot encoding emits
   a dedicated ``<col>_Missing`` column rather than silently producing an
   all-zeros row.

The :func:`clean_for_modelling` convenience function applies all three in
order and is what every Phase 2.3 pipeline factory calls.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from .combine import HFP_COLUMNS

#: The five features the EDA found with >=10% missingness in at least one
#: source (`05-eda-findings.md` §2.2). Each gets a `<col>_was_missing`
#: indicator added by :func:`add_missingness_indicators`.
MISSINGNESS_INDICATOR_COLUMNS: Final[tuple[str, ...]] = (
    "RestingBP",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
)

#: Categorical features in the HFP schema. These are the columns that get
#: NaN -> "Missing" replacement before one-hot encoding.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
)

#: Numeric continuous features in the HFP schema. Used by the LR baseline
#: pipeline for spline expansion and standard scaling.
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
)

#: Numeric binary features in the HFP schema. These bypass spline expansion
#: but are passed through scaling unchanged (mean centred to ~0/1).
BINARY_NUMERIC_COLUMNS: Final[tuple[str, ...]] = ("FastingBS",)

#: Sentinel value the categorical replacement uses. Exposed so tests and
#: downstream code can refer to it without hard-coding the string.
MISSING_CATEGORY_LABEL: Final[str] = "Missing"


def _require_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    """Raise ``KeyError`` if any of ``required`` is not in ``df.columns``."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"required columns missing from DataFrame: {missing}")


def clean_cholesterol_zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``Cholesterol == 0`` to ``NaN``.

    The HFP schema documents ``0`` as the missingness sentinel for
    cholesterol. Treating it as a real measurement (as the prior Honours
    pipeline did) hands any model a perfect source-membership detector for
    Switzerland (100% sentinel) and Long Beach VA (24.5% sentinel).

    Idempotent: a second call is a no-op because there are no remaining
    ``0`` values to convert. Returns a copy; the input frame is unchanged.
    """
    _require_columns(df, ("Cholesterol",))
    out = df.copy()
    out["Cholesterol"] = out["Cholesterol"].mask(out["Cholesterol"] == 0)
    return out


def add_missingness_indicators(
    df: pd.DataFrame,
    columns: tuple[str, ...] = MISSINGNESS_INDICATOR_COLUMNS,
) -> pd.DataFrame:
    """Append ``<col>_was_missing`` binary indicators for each named column.

    The indicator is computed *before* any imputation so it captures the
    original missingness pattern. Idempotent: if the indicator column
    already exists, it is left untouched. That matters because downstream
    cleaning (e.g. :func:`replace_categorical_missing`) replaces NaN with
    a sentinel string, after which a naive recompute would flip the
    indicator from 1 to 0 and silently destroy the missingness signal.
    """
    _require_columns(df, columns)
    out = df.copy()
    for col in columns:
        indicator = f"{col}_was_missing"
        if indicator in out.columns:
            continue
        out[indicator] = out[col].isna().astype("int8")
    return out


def replace_categorical_missing(
    df: pd.DataFrame,
    columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
    label: str = MISSING_CATEGORY_LABEL,
) -> pd.DataFrame:
    """Replace ``NaN`` in each categorical column with the ``label`` string.

    Downstream :class:`~sklearn.preprocessing.OneHotEncoder` will then emit
    a dedicated ``<col>_Missing`` column instead of an all-zeros encoding,
    making the missingness explicit and debuggable.

    Idempotent: a second call sees no NaN values and is a no-op.
    """
    _require_columns(df, columns)
    out = df.copy()
    for col in columns:
        out[col] = out[col].astype(object).where(out[col].notna(), label)
    return out


def coerce_numeric_to_float64(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every numeric feature column to ``float64``.

    The combine step uses pandas nullable ``Int64`` for ``Age`` and
    ``FastingBS`` so missing values can be represented as ``pd.NA``.
    Sklearn's ``ColumnTransformer`` cannot serialise ``pd.NA``-bearing
    columns into a numpy array via passthrough, so we coerce them once
    here. Float64 with ``np.nan`` sentinels is the format every downstream
    sklearn transformer expects.

    Leaves the ``HeartDisease`` target untouched (it is never NA) and
    leaves the ``source`` column untouched (never a model feature).
    Idempotent.
    """
    out = df.copy()
    for col in (*NUMERIC_COLUMNS, *BINARY_NUMERIC_COLUMNS):
        if col in out.columns:
            out[col] = out[col].astype("float64")
    return out


def clean_for_modelling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all four deterministic cleaning steps in canonical order.

    This is the function every model pipeline factory in
    :mod:`cardiorisk.features.pipeline` calls before the stateful
    preprocessing. The output frame contains:

    - All original HFP columns (with ``Cholesterol == 0`` converted to NaN
      and numeric features coerced to ``float64`` with ``np.nan`` sentinels).
    - Five new ``<col>_was_missing`` indicator columns (``int8``).
    - Categorical NaN replaced with the literal ``"Missing"`` category.
    - The ``source`` column preserved if present (used by the LODO splitter
      in :mod:`cardiorisk.features.cv`; never used as a model feature).

    Idempotent.
    """
    _require_columns(df, HFP_COLUMNS)
    out = clean_cholesterol_zero_to_nan(df)
    out = add_missingness_indicators(out)
    out = replace_categorical_missing(out)
    out = coerce_numeric_to_float64(out)
    return out
