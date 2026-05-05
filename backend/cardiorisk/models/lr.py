"""L1 logistic regression with restricted-cubic-spline expansion.

The transparency anchor of the v1 stack
([`04-revised-design.md`](../../../docs/research/04-revised-design.md)
§2.3 + ADR-006). Composed of:

1. The Phase-2.2 :func:`cardiorisk.features.pipeline.make_lr_pipeline`
   preprocessing prefix (mean / mode imputation, RCS expansion of
   continuous features, StandardScaler on numerics, one-hot encoding
   of categoricals with the explicit ``"Missing"`` category).
2. ``sklearn.linear_model.LogisticRegression`` with L1 penalty,
   ``solver='saga'`` (per the user's Phase-2.3 plan), tuned over the
   ``C`` grid ``{0.001, 0.01, 0.1, 1, 10, 100}`` via
   ``GridSearchCV`` against the inner-fold validation slice.

Hyperparameter selection: 5-fold stratified CV over the inner training
slice, scoring on AUROC (matches the headline metric the eval harness
reports). The outer LODO loop handles generalisation evaluation; this
inner CV exists only to pick ``C``.

Calibration: the wrapper returns the bare best estimator. The training
driver applies sigmoid (Platt) calibration externally via
:func:`cardiorisk.calibration.calibrate_for_model` per ADR-009.
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from cardiorisk.features.pipeline import make_lr_pipeline
from cardiorisk.features.spline import DEFAULT_N_KNOTS
from cardiorisk.models.base import SEED

#: ``C`` grid from the Phase-2.3 plan (logspace from 0.001 to 100).
DEFAULT_C_GRID: Final[tuple[float, ...]] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

#: Solver per the Phase-2.3 plan (saga handles L1 + dense data well).
DEFAULT_SOLVER: Final[str] = "saga"

#: Inner CV folds for the C grid search.
DEFAULT_INNER_CV_FOLDS: Final[int] = 5

#: Max iterations for the LR solver. Saga can need many iterations to
#: converge on highly-correlated features (the RCS expansion creates
#: them by construction). 5,000 is comfortable for our fold sizes.
DEFAULT_MAX_ITER: Final[int] = 5_000


class LRModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """L1 LR + RCS wrapper conforming to :class:`~cardiorisk.models.base.ModelWrapper`.

    Inherits ``ClassifierMixin`` + ``BaseEstimator`` so sklearn 1.8's
    estimator-tags machinery treats it as a proper classifier; that's
    required for ``CalibratedClassifierCV`` to wrap us via
    ``FrozenEstimator``. ``__init__`` parameters are stored verbatim
    (sklearn's get_params introspection convention).


    Parameters
    ----------
    n_knots : int, default=4
        Number of RCS knots per continuous feature. ADR-008 fixes the
        default at 4 (the median of Harrell's recommended 3-5 range);
        no ablation in v1.
    c_grid : tuple of float, default=DEFAULT_C_GRID
        Logarithmic grid of inverse regularisation strengths to search.
    inner_cv_folds : int, default=5
        Stratified-CV fold count for the C grid search inside each
        outer LODO fold.
    seed : int, default=SEED
        RNG seed for the inner CV shuffle and the saga solver.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The full preprocessing + LR pipeline, fitted with the best ``C``.
    best_c_ : float
        The ``C`` value chosen by the inner CV.
    """

    model_name: str = "lr"

    def __init__(
        self,
        *,
        n_knots: int = DEFAULT_N_KNOTS,
        c_grid: tuple[float, ...] = DEFAULT_C_GRID,
        inner_cv_folds: int = DEFAULT_INNER_CV_FOLDS,
        seed: int = SEED,
    ) -> None:
        self.n_knots = n_knots
        self.c_grid = c_grid
        self.inner_cv_folds = inner_cv_folds
        self.seed = seed

    def _build_unfit_pipeline(self) -> Pipeline:
        """Compose the Phase-2.2 LR preprocessing + a fresh LR estimator.

        sklearn 1.8 deprecated the ``penalty=`` argument in favour of
        ``l1_ratio`` alone: ``l1_ratio=1.0`` recovers pure L1,
        ``l1_ratio=0.0`` recovers pure L2, anything in between is
        elasticnet. We pass ``l1_ratio`` only and leave ``penalty`` at
        its default to silence the FutureWarning and stay forward-
        compatible into sklearn 1.10.
        """
        preprocess = make_lr_pipeline(n_knots=self.n_knots).named_steps["preprocess"]
        estimator = LogisticRegression(
            l1_ratio=1.0,
            solver=DEFAULT_SOLVER,
            C=1.0,  # placeholder, overridden by GridSearchCV
            max_iter=DEFAULT_MAX_ITER,
            random_state=self.seed,
        )
        return Pipeline(steps=[("preprocess", preprocess), ("clf", estimator)])

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> LRModel:
        """Fit preprocessing + run inner-CV grid search over ``C``.

        Stores the best-C-fitted pipeline at ``self.pipeline_`` and the
        chosen ``C`` at ``self.best_c_``. The inner GridSearchCV refits
        on the full training slice with ``refit=True``.
        """
        cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.seed)
        param_grid = {"clf__C": list(self.c_grid)}
        search = GridSearchCV(
            estimator=self._build_unfit_pipeline(),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=1,  # saga is already multi-threaded internally
            error_score="raise",
        )
        # saga emits ConvergenceWarning at small n / extreme C; documented
        # behaviour, doesn't affect the chosen C in practice. Suppressed
        # locally so it doesn't fail tests under filterwarnings=error.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search.fit(X, np.asarray(y))
        self.pipeline_: Pipeline = search.best_estimator_
        self.best_c_: float = float(search.best_params_["clf__C"])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive- and negative-class probabilities, shape ``(n, 2)``."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("LRModel must be fit before predict_proba()")
        return np.asarray(self.pipeline_.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard-class prediction. Required by ``CalibratedClassifierCV``."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("LRModel must be fit before predict()")
        return np.asarray(self.pipeline_.predict(X))

    @property
    def classes_(self) -> np.ndarray:
        """Sklearn-compatible classes_ attribute, used by CalibratedClassifierCV."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("LRModel must be fit before classes_ is available")
        clf = self.pipeline_.named_steps["clf"]
        return np.asarray(clf.classes_)


def build_lr() -> LRModel:
    """Factory used by the training driver."""
    return LRModel()
