"""Per-model preprocessing pipeline factories.

Each ``make_<model>_pipeline()`` returns an unfit :class:`sklearn.pipeline.Pipeline`
that, given a *cleaned* HFP DataFrame (pass it through
:func:`cardiorisk.data.preprocess.clean_for_modelling` first), produces the
numeric matrix that model expects.

The cleaning step is leakage-free (pure functions on individual rows) and is
done once at data-load time. Everything stateful — imputation, scaling,
spline expansion — is part of these sklearn pipelines and must be ``fit`` on
each LODO fold's training slice only, then ``transform``-applied to the
val / calibration / test slices. Sklearn's ``fit`` / ``transform`` boundary
is what mechanically enforces leakage protection.

Per-model design rationale (cross-ref `04-revised-design.md` §3):

- :func:`make_tabpfn_pipeline` — TabPFN handles NaN natively, so numerics
  pass through unchanged. Categoricals are one-hot encoded (TabPFN accepts
  one-hot-encoded categoricals as numeric features).
- :func:`make_xgboost_pipeline` — XGBoost also handles NaN natively, but
  the design doc applies the same MissForest impute (sklearn
  ``IterativeImputer`` + ``RandomForestRegressor``) so the headline
  comparison isolates the model from the imputation scheme.
- :func:`make_lr_pipeline` — L1 LR baseline: mean / mode impute,
  one-hot, restricted-cubic-spline expansion of continuous features,
  standardize. The "published worst case" path from the design doc.
- :func:`make_woa_pipeline` — same imputation as XGBoost, plus
  StandardScaler because the WOA-Ensemble (CNN+LSTM+ANN) needs scaled
  inputs.
"""

from __future__ import annotations

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  (registers IterativeImputer)
)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardiorisk.data.preprocess import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    MISSINGNESS_INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
)
from cardiorisk.features.spline import DEFAULT_N_KNOTS, RestrictedCubicSpline

#: Pinned random seed for any stochastic preprocessing component
#: (IterativeImputer's RF estimators). Same constant as
#: :data:`cardiorisk.features.cv.SEED`.
SEED: Final[int] = 20260505

#: Number of trees in the per-feature RandomForest used by IterativeImputer.
#: A modest count (50) keeps fit cost reasonable across LODO folds; the
#: imputer is not the headline model so we don't need 500 trees here.
MISSFOREST_N_ESTIMATORS: Final[int] = 50

#: Column groups in the *cleaned* frame, named for clarity at the call site.
_NUMERIC_COLS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS
_CATEGORICAL_COLS: Final[tuple[str, ...]] = CATEGORICAL_COLUMNS
_INDICATOR_COLS: Final[tuple[str, ...]] = tuple(
    f"{c}_was_missing" for c in MISSINGNESS_INDICATOR_COLUMNS
)


def _one_hot_encoder() -> OneHotEncoder:
    """OHE configured for our LODO setting.

    ``handle_unknown='ignore'`` matters: a category that appears in test
    but not training (rare but possible under LODO with small per-source
    samples — e.g. a single ``ST_Slope`` value present only in Switzerland)
    is encoded as all-zeros rather than raising. We previously specified
    ``handle_unknown='infrequent_if_exist'`` with ``min_frequency=1``, but
    ``min_frequency=1`` makes no fitted category infrequent (a fitted level
    must occur ≥1 times by definition), so the infrequent-bucket fallback
    never triggers and the behaviour collapses to ``handle_unknown='ignore'``
    semantics anyway. ``ignore`` makes the contract honest and removes
    misleading config.
    """
    return OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype="float64",
    )


def _missforest_continuous_imputer() -> IterativeImputer:
    """MissForest variant: IterativeImputer with RandomForestRegressor."""
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=MISSFOREST_N_ESTIMATORS,
            random_state=SEED,
            n_jobs=1,
        ),
        random_state=SEED,
        max_iter=10,
        sample_posterior=False,
    )


def _missforest_classifier_imputer() -> IterativeImputer:
    """MissForest variant for binary numeric features (FastingBS)."""
    return IterativeImputer(
        estimator=RandomForestClassifier(
            n_estimators=MISSFOREST_N_ESTIMATORS,
            random_state=SEED,
            n_jobs=1,
        ),
        random_state=SEED,
        max_iter=10,
        sample_posterior=False,
    )


# -------------------------------------------------------------- factories


def make_tabpfn_pipeline() -> Pipeline:
    """Preprocessing pipeline for the TabPFN headline model.

    TabPFN accepts NaN natively, so numerics (continuous + binary) and the
    missingness indicators all pass through unchanged. Categoricals are
    one-hot encoded so the model sees fixed-width numeric input.

    Returns an unfit pipeline. Caller is responsible for ``.fit(X_train,
    y_train).transform(X_test)``.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("ohe_categoricals", _one_hot_encoder(), list(_CATEGORICAL_COLS)),
            ("passthrough_numeric", "passthrough", list(_NUMERIC_COLS)),
            ("passthrough_indicators", "passthrough", list(_INDICATOR_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", column_transformer)])


def make_xgboost_pipeline() -> Pipeline:
    """Preprocessing pipeline for the calibrated-XGBoost baseline.

    Per `04-revised-design.md` §3.3, XGBoost shares the MissForest
    (IterativeImputer + RF) treatment with WOA so the comparison isolates
    model choice. No scaling — XGBoost is scale-invariant.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("ohe_categoricals", _one_hot_encoder(), list(_CATEGORICAL_COLS)),
            ("missforest_continuous", _missforest_continuous_imputer(), list(NUMERIC_COLUMNS)),
            ("missforest_binary", _missforest_classifier_imputer(), list(BINARY_NUMERIC_COLUMNS)),
            ("passthrough_indicators", "passthrough", list(_INDICATOR_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", column_transformer)])


def make_lr_pipeline(*, n_knots: int = DEFAULT_N_KNOTS) -> Pipeline:
    """Preprocessing pipeline for the L1 LR transparency baseline.

    Per `04-revised-design.md` §2.3 + §3.3, the LR baseline uses
    mean / mode imputation as the published worst-case (mean for
    continuous, most-frequent for binary), then RCS expansion of
    continuous features, then standard scaling. Categoricals are
    one-hot encoded with the explicit ``Missing`` category supplied by
    :func:`cardiorisk.data.preprocess.replace_categorical_missing`.
    """
    rcs_pipeline = Pipeline(
        steps=[
            ("mean_impute", SimpleImputer(strategy="mean")),
            ("rcs", RestrictedCubicSpline(n_knots=n_knots)),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("mode_impute", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("ohe_categoricals", _one_hot_encoder(), list(_CATEGORICAL_COLS)),
            ("rcs_continuous", rcs_pipeline, list(NUMERIC_COLUMNS)),
            ("scaled_binary", binary_pipeline, list(BINARY_NUMERIC_COLUMNS)),
            ("passthrough_indicators", "passthrough", list(_INDICATOR_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", column_transformer)])


def make_woa_pipeline() -> Pipeline:
    """Preprocessing pipeline for the WOA-Ensemble Honours-baseline run.

    Same imputation as XGBoost (shared MissForest treatment) plus
    StandardScaler because the original architecture (CNN+LSTM+ANN) was
    trained on Z-score-normalised inputs and we faithfully reproduce
    that in the Phase 2.4 comparison run.
    """
    continuous_pipeline = Pipeline(
        steps=[
            ("missforest", _missforest_continuous_imputer()),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("missforest", _missforest_classifier_imputer()),
            ("scale", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("ohe_categoricals", _one_hot_encoder(), list(_CATEGORICAL_COLS)),
            ("scaled_continuous", continuous_pipeline, list(NUMERIC_COLUMNS)),
            ("scaled_binary", binary_pipeline, list(BINARY_NUMERIC_COLUMNS)),
            ("passthrough_indicators", "passthrough", list(_INDICATOR_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", column_transformer)])
