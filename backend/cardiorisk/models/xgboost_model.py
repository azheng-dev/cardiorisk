"""XGBoost + Optuna Bayesian hyperparameter tuning.

The white-box workhorse of the v1 stack
([`04-revised-design.md`](../../../docs/research/04-revised-design.md)
§2.2 + ADR-006). Composed of:

1. The Phase-2.2 :func:`cardiorisk.features.pipeline.make_xgboost_pipeline`
   preprocessing prefix (MissForest imputation, one-hot encoding).
2. ``xgboost.XGBClassifier`` as the base learner.
3. Optuna Bayesian hyperparameter search over a tabular-classification-
   appropriate grid (per the Phase-2.3 plan: 50 trials, 10-min cap).

Module name is ``xgboost_model`` not ``xgboost`` to avoid shadowing
the upstream ``xgboost`` package — ``from cardiorisk.models import
xgboost`` would be a confusing alias.

Hyperparameter search budget (from the Phase-2.3 plan):

- 50 trials per outer LODO fold.
- 10-minute wall-clock cap (Optuna ``timeout`` in seconds).
- Inner 5-fold stratified CV on the training slice, scoring on AUROC.
- Ephemeral in-memory Optuna study (no SQLite persistence by user
  decision; trial history is not retained beyond the best params).

Calibration: the wrapper returns the bare best estimator. The training
driver applies isotonic calibration externally via
:func:`cardiorisk.calibration.calibrate_for_model` per ADR-009.
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from cardiorisk.features.pipeline import make_xgboost_pipeline
from cardiorisk.models.base import SEED

#: Optuna trial budget per LODO fold (Phase-2.3 plan).
DEFAULT_N_TRIALS: Final[int] = 50

#: Wall-clock cap for the Optuna study, in seconds (Phase-2.3 plan).
DEFAULT_TIMEOUT_SECONDS: Final[int] = 600

#: Inner CV folds for AUROC scoring during Optuna trials.
DEFAULT_INNER_CV_FOLDS: Final[int] = 5


def _xgb_search_space(trial: optuna.Trial) -> dict[str, float | int]:
    """Tabular-classification search space for XGBoost.

    Ranges are conservative defaults that work well for n=700-ish HFP
    folds with 11 features (post one-hot expansion ~25 features). No
    extreme-depth or extreme-tree-count exploration — they over-fit at
    our n; the published HFP-friendly defaults sit in the moderate
    range below.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
    }


def _xgb_estimator(seed: int, **params: float | int) -> XGBClassifier:
    """Construct an unfit XGBClassifier with our pinned defaults + ``params``."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        **params,
    )


class XGBoostModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """XGBoost + Optuna wrapper conforming to :class:`~cardiorisk.models.base.ModelWrapper`.

    Inherits ``ClassifierMixin`` + ``BaseEstimator`` so sklearn 1.8
    treats us as a proper classifier — required for
    ``CalibratedClassifierCV`` (post-hoc isotonic calibration applied
    by the training driver via :func:`cardiorisk.calibration.calibrate_for_model`).


    Parameters
    ----------
    n_trials : int, default=DEFAULT_N_TRIALS
        Optuna trial budget. Lowered to 1 in ``--smoke`` mode by the
        training driver via the ``n_trials`` constructor argument.
    timeout_seconds : int, default=DEFAULT_TIMEOUT_SECONDS
        Wall-clock cap for the study. Optuna stops at whichever of
        ``n_trials`` or ``timeout`` is reached first.
    inner_cv_folds : int, default=DEFAULT_INNER_CV_FOLDS
        Stratified-CV fold count for the AUROC scoring inside each
        Optuna trial.
    seed : int, default=SEED
        RNG seed for the TPE sampler, the CV shuffle, and the XGBoost
        estimator itself.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The full preprocessing + best-XGB pipeline, refit on the full
        training slice with the best hyperparameters.
    best_params_ : dict[str, float | int]
        The Optuna best parameters.
    best_score_ : float
        Mean inner-CV AUROC at the best parameter combination.
    """

    model_name: str = "xgboost"

    def __init__(
        self,
        *,
        n_trials: int = DEFAULT_N_TRIALS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        inner_cv_folds: int = DEFAULT_INNER_CV_FOLDS,
        seed: int = SEED,
    ) -> None:
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.inner_cv_folds = inner_cv_folds
        self.seed = seed

    def _build_unfit_pipeline(self, **xgb_params: float | int) -> Pipeline:
        """Compose the Phase-2.2 XGBoost preprocessing + a fresh estimator."""
        preprocess = make_xgboost_pipeline().named_steps["preprocess"]
        estimator = _xgb_estimator(self.seed, **xgb_params)
        return Pipeline(steps=[("preprocess", preprocess), ("clf", estimator)])

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
        """Mean inner-CV AUROC for one Optuna trial."""
        params = _xgb_search_space(trial)
        pipeline = self._build_unfit_pipeline(**params)
        cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.seed)
        scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=cv, n_jobs=1)
        return float(np.mean(scores))

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> XGBoostModel:
        """Run Optuna search; refit on the full training slice with best params."""
        y_arr = np.asarray(y)

        # Optuna's INFO logging is noisy at 50 trials per fold; quiet it
        # at the library level so the driver's own progress logging stays
        # readable.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        with warnings.catch_warnings():
            # ExperimentalWarning fires once per study; we accept the API.
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            study.optimize(
                lambda trial: self._objective(trial, X, y_arr),
                n_trials=self.n_trials,
                timeout=self.timeout_seconds,
                show_progress_bar=False,
                n_jobs=1,
            )

        self.best_params_: dict[str, float | int] = dict(study.best_params)
        self.best_score_: float = float(study.best_value)
        self.pipeline_: Pipeline = self._build_unfit_pipeline(**self.best_params_)
        self.pipeline_.fit(X, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive- and negative-class probabilities, shape ``(n, 2)``."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("XGBoostModel must be fit before predict_proba()")
        return np.asarray(self.pipeline_.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard-class prediction. Required by ``CalibratedClassifierCV``."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("XGBoostModel must be fit before predict()")
        return np.asarray(self.pipeline_.predict(X))

    @property
    def classes_(self) -> np.ndarray:
        """Sklearn-compatible classes_ attribute, used by CalibratedClassifierCV."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("XGBoostModel must be fit before classes_ is available")
        clf = self.pipeline_.named_steps["clf"]
        return np.asarray(clf.classes_)


def build_xgboost(*, n_trials: int = DEFAULT_N_TRIALS) -> XGBoostModel:
    """Factory used by the training driver. ``n_trials`` overridable for ``--smoke``."""
    return XGBoostModel(n_trials=n_trials)
