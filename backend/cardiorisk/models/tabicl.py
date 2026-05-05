"""TabICL — Tabular In-Context Learning foundation model.

The v1 TFM headline. Replaces the original TabPFN choice from
ADR-006 per [ADR-011](../../../docs/adr/011-tfm-tabicl-supersedes-tabpfn.md);
the substitution rationale lives in that ADR. Same in-context-learning
TFM family as TabPFN (transformer pretrained on synthetic tabular
tasks, zero-shot inference), but BSD-3-licensed and freely
distributable, which preserves ADR-010's reproducibility commitment.

Composed of:

1. The Phase-2.2
   :func:`cardiorisk.features.pipeline.make_tabpfn_pipeline`
   preprocessing prefix (NaN-passthrough on numerics + one-hot
   encoding of categoricals + indicator passthrough). The factory's
   name predates ADR-011 — its semantics ("TFM-compatible
   preprocessing: pass NaN through, one-hot the categoricals") are
   the same for TabICL as they were for TabPFN.
2. ``tabicl.TabICLClassifier`` as the base learner, pinned to
   ``device='cpu'`` and ``random_state=SEED`` for determinism.

No hyperparameter search. TFMs are zero-shot by construction; their
"hyperparameters" are baked into the pretrained weights. We accept
the published defaults.

Calibration: TabICL is calibrated by construction (its training
objective is direct probabilistic prediction). The
:func:`cardiorisk.calibration.calibrate_for_model` dispatcher passes
``"tabicl"`` through unwrapped — no post-hoc calibrator is fitted.

NaN handling: empirically verified during Phase 2.3b implementation:
TabICL accepts NaN-bearing inputs and produces sensible probabilities
on rows with missing values. The Phase-2.2 NaN-passthrough pipeline
works unmodified.

Model weights: downloaded automatically from the public Hugging Face
repo ``jingang/TabICL`` on first use (~50 MB, cached under the
HuggingFace cache dir). No authentication required.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from tabicl import TabICLClassifier

from cardiorisk.features.pipeline import make_tabpfn_pipeline
from cardiorisk.models.base import SEED

#: Inference device. CPU only — TabICL is light enough that the GPU
#: round-trip overhead exceeds its benefit at our LODO fold sizes.
DEFAULT_DEVICE: Final[str] = "cpu"


class TabICLModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """TabICL wrapper conforming to :class:`~cardiorisk.models.base.ModelWrapper`.

    Inherits ``ClassifierMixin`` + ``BaseEstimator`` for API symmetry
    with the other wrappers; TabICL doesn't use post-hoc calibration
    (passes through unwrapped per ADR-009 / ADR-011) so the inheritance
    is mostly cosmetic for this wrapper, but consistency matters.


    Parameters
    ----------
    device : str, default='cpu'
        Inference device. Set to ``'cuda'`` if a GPU is available and
        the LODO fold size grows materially (n > ~5000). At our HFP
        sizes (~700 train rows per fold) CPU is faster end-to-end.
    seed : int, default=SEED
        RNG seed forwarded to ``TabICLClassifier``.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The full preprocessing + TabICL pipeline, fitted on the
        training slice.
    """

    model_name: str = "tabicl"

    def __init__(
        self,
        *,
        device: str = DEFAULT_DEVICE,
        seed: int = SEED,
    ) -> None:
        self.device = device
        self.seed = seed

    def _build_unfit_pipeline(self) -> Pipeline:
        """Compose the Phase-2.2 TFM preprocessing + a fresh TabICL classifier."""
        preprocess = make_tabpfn_pipeline().named_steps["preprocess"]
        estimator = TabICLClassifier(
            device=self.device,
            random_state=self.seed,
            verbose=False,
            n_jobs=1,
        )
        return Pipeline(steps=[("preprocess", preprocess), ("clf", estimator)])

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> TabICLModel:
        """Fit the preprocessing prefix + run TabICL's in-context fit.

        For TabICL, "fit" caches the (X_train, y_train) tensor that
        in-context inference will condition on at predict time; there
        is no gradient-based optimisation. Fast.
        """
        self.pipeline_: Pipeline = self._build_unfit_pipeline()
        self.pipeline_.fit(X, np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive- and negative-class probabilities, shape ``(n, 2)``."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("TabICLModel must be fit before predict_proba()")
        return np.asarray(self.pipeline_.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard-class prediction. Kept for API symmetry with the other wrappers."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("TabICLModel must be fit before predict()")
        return np.asarray(self.pipeline_.predict(X))

    @property
    def classes_(self) -> np.ndarray:
        """Sklearn-compatible classes_ attribute, used by CalibratedClassifierCV."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("TabICLModel must be fit before classes_ is available")
        clf = self.pipeline_.named_steps["clf"]
        return np.asarray(clf.classes_)


def build_tabicl() -> TabICLModel:
    """Factory used by the training driver."""
    return TabICLModel()
