"""Tests for cardiorisk.monitoring.reference.

Covers:

- ``compute_quantile_edges`` on well-conditioned, constant, and empty inputs.
- ``build_fold_reference`` numeric + categorical dispatch.
- Optional prediction-drift binning when models are passed.
- Round-trip ``save_reference`` / ``load_reference`` through joblib.
- Error handling: missing file, wrong type on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardiorisk.monitoring.reference import (
    DEFAULT_N_BINS,
    CategoricalReference,
    FoldReference,
    NumericReference,
    PredictionReference,
    build_fold_reference,
    compute_quantile_edges,
    load_reference,
    save_reference,
)

SEED = 20260506


# ---------------------------------------------------------------- quantile edges


def test_compute_quantile_edges_returns_n_bins_plus_one_edges() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 1_000)
    edges = compute_quantile_edges(samples, n_bins=10)
    assert edges.size == 11


def test_compute_quantile_edges_outer_edges_extend_to_infinity() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 1_000)
    edges = compute_quantile_edges(samples, n_bins=10)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_compute_quantile_edges_is_strictly_increasing() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 1_000)
    edges = compute_quantile_edges(samples, n_bins=10)
    assert np.all(np.diff(edges) > 0)


def test_compute_quantile_edges_collapses_duplicates_for_low_cardinality() -> None:
    samples = np.array([0.0] * 500 + [1.0] * 500, dtype=np.float64)
    edges = compute_quantile_edges(samples, n_bins=10)
    # Many quantiles collapse to 0 or 1; uniqueness then leaves us with
    # at most 4 edges (-inf, 0, 1, inf -> after dedup of inner repeats).
    assert edges.size >= 2
    assert np.all(np.diff(edges) > 0)


def test_compute_quantile_edges_constant_input_collapses_to_few_edges() -> None:
    """All quantiles of a constant sample collapse to the same value;
    after the outer-edge extension and unique() dedup we end up with at
    most three edges (-inf, value, inf). PSI on this reference is still
    well-defined: any current sample lands in one of the two bins."""
    samples = np.full(100, 3.14, dtype=np.float64)
    edges = compute_quantile_edges(samples, n_bins=10)
    assert edges.size <= 3
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)


def test_compute_quantile_edges_empty_input_returns_single_bin() -> None:
    edges = compute_quantile_edges(np.array([], dtype=np.float64), n_bins=10)
    np.testing.assert_array_equal(edges, np.array([-np.inf, np.inf]))


def test_compute_quantile_edges_drops_nans() -> None:
    rng = np.random.default_rng(SEED)
    samples = rng.normal(0, 1, 1_000)
    samples[::10] = np.nan
    edges = compute_quantile_edges(samples, n_bins=5)
    assert np.all(np.isfinite(edges[1:-1]))


def test_compute_quantile_edges_raises_on_zero_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        compute_quantile_edges(np.array([1.0, 2.0]), n_bins=0)


# ---------------------------------------------------------------- build_fold_reference


@pytest.fixture
def hfp_like_train() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 600
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 80, n).astype(np.float64),
            "RestingBP": rng.normal(130, 20, n),
            "Cholesterol": rng.normal(220, 50, n),
            "MaxHR": rng.normal(140, 25, n),
            "Oldpeak": rng.normal(1.0, 1.0, n),
            "FastingBS": rng.integers(0, 2, n).astype(np.float64),
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], size=n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
            "ExerciseAngina": rng.choice(["N", "Y"], size=n),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
        }
    )


def test_build_fold_reference_populates_all_numeric_features(hfp_like_train: pd.DataFrame) -> None:
    ref = build_fold_reference(held_out_source="Cleveland", X_train=hfp_like_train)
    assert set(ref.numeric.keys()) >= {
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak",
        "FastingBS",
    }
    for nref in ref.numeric.values():
        assert isinstance(nref, NumericReference)
        assert nref.edges.size >= 2
        assert nref.counts.sum() == nref.n


def test_build_fold_reference_populates_all_categorical_features(
    hfp_like_train: pd.DataFrame,
) -> None:
    ref = build_fold_reference(held_out_source="Cleveland", X_train=hfp_like_train)
    assert set(ref.categorical.keys()) == {
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    }
    for cref in ref.categorical.values():
        assert isinstance(cref, CategoricalReference)
        assert sum(cref.counts.values()) == cref.n


def test_build_fold_reference_records_n_train_and_held_out_source(
    hfp_like_train: pd.DataFrame,
) -> None:
    ref = build_fold_reference(held_out_source="LongBeachVA", X_train=hfp_like_train)
    assert ref.held_out_source == "LongBeachVA"
    assert ref.n_train == len(hfp_like_train)
    assert ref.n_bins == DEFAULT_N_BINS


def test_build_fold_reference_skips_columns_not_in_dataframe() -> None:
    minimal = pd.DataFrame({"Age": [30.0, 40.0, 50.0], "Sex": ["M", "F", "M"]})
    ref = build_fold_reference(held_out_source="x", X_train=minimal)
    assert "Age" in ref.numeric
    assert "Sex" in ref.categorical
    assert "RestingBP" not in ref.numeric


def test_build_fold_reference_counts_missing_values(hfp_like_train: pd.DataFrame) -> None:
    df = hfp_like_train.copy()
    df.loc[df.index[:30], "RestingBP"] = np.nan
    ref = build_fold_reference(held_out_source="x", X_train=df)
    assert ref.numeric["RestingBP"].n_missing == 30
    assert ref.numeric["RestingBP"].n == len(df) - 30


def test_build_fold_reference_populates_prediction_when_models_supplied(
    hfp_like_train: pd.DataFrame,
) -> None:
    class _StubModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            rng = np.random.default_rng(SEED)
            p = rng.uniform(0.05, 0.95, size=len(X))
            return np.column_stack([1 - p, p])

    ref = build_fold_reference(
        held_out_source="x",
        X_train=hfp_like_train,
        models={"stub": _StubModel()},
    )
    assert "stub" in ref.prediction
    pref = ref.prediction["stub"]
    assert isinstance(pref, PredictionReference)
    assert pref.n == len(hfp_like_train)
    assert 0.0 < pref.mean < 1.0


def test_build_fold_reference_rejects_models_with_bad_predict_proba(
    hfp_like_train: pd.DataFrame,
) -> None:
    class _BadModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))  # 1-D, not (n, 2)

    with pytest.raises(ValueError, match="predict_proba"):
        build_fold_reference(
            held_out_source="x",
            X_train=hfp_like_train,
            models={"bad": _BadModel()},
        )


# ---------------------------------------------------------------- persistence


def test_save_and_load_reference_round_trip(tmp_path: Path, hfp_like_train: pd.DataFrame) -> None:
    ref = build_fold_reference(held_out_source="Cleveland", X_train=hfp_like_train)
    path = tmp_path / "Cleveland_reference.joblib"
    save_reference(ref, path)
    loaded = load_reference(path)
    assert isinstance(loaded, FoldReference)
    assert loaded.held_out_source == ref.held_out_source
    assert loaded.n_train == ref.n_train
    np.testing.assert_array_equal(loaded.numeric["Age"].edges, ref.numeric["Age"].edges)
    np.testing.assert_array_equal(loaded.numeric["Age"].counts, ref.numeric["Age"].counts)
    assert loaded.categorical["Sex"].counts == ref.categorical["Sex"].counts


def test_load_reference_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reference artefact missing"):
        load_reference(tmp_path / "nope.joblib")


def test_load_reference_raises_on_wrong_type_on_disk(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a FoldReference"}, path)
    with pytest.raises(TypeError, match="expected FoldReference"):
        load_reference(path)
