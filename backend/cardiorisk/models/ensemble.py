"""Honours-Ensemble baseline (PyTorch port of the 4-net mean-averaged ensemble).

Phase 2.4 Honours-baseline reproduction. The Honours team's
``Demos/Data_Pre-processing.ipynb`` cell 55 defines an ensemble of four
parallel sub-networks — DNN, 1D CNN, LSTM, BiLSTM — each trained
independently on binary cross-entropy with Adam, batch size 32, 100
epochs, dropout 0.2 throughout. Inference takes the **mean of the four
sigmoid outputs** (no learned meta-model).

This module is a faithful PyTorch port of that architecture. The
original is TensorFlow/Keras; we port to PyTorch to keep one DL
framework in the repo (TabICL already pulls torch in).

**What is reproduced** (all from the archive code):

- DNN: ``Linear(n→100) → ReLU → Dropout(0.2) → Linear(100→64) → ReLU →
  Dropout(0.2) → Linear(64→128) → ReLU → Dropout(0.2) → Linear(128→1)``
  with sigmoid at inference.
- 1D CNN: ``Conv1d(1→64, k=3) → MaxPool1d(2) → Dropout(0.2) → Flatten
  → Linear(...→128) → Linear(128→64) → Dropout(0.2) → Linear(64→1)``.
- LSTM: ``LSTM(input_size=1, hidden=128, batch_first=True) → take last
  step → Dropout(0.2) → Linear(128→1)``.
- BiLSTM: same as LSTM but ``bidirectional=True`` → ``Linear(256→1)``.

**What is *not* reproduced** (and is documented honestly):

- The Honours WOA feature-selection layer is **not in the archive**
  (the ``WOA`` markdown header in ``Data_Pre-processing.ipynb`` cell 40
  has no code under it). We do not invent a WOA implementation here;
  see ADR-012 + ``docs/research/09-honours-vs-v1.md``.
- Keras's ``LSTM(recurrent_dropout=0.2)`` has no PyTorch ``nn.LSTM``
  equivalent. We omit recurrent dropout. This is a deliberate departure
  documented in ADR-012; the architectural class (LSTM with dropout
  on the output layer) is preserved.
- Keras initialisers default to Glorot uniform; PyTorch defaults to
  Kaiming uniform for Linear. We accept PyTorch defaults — different
  initialiser, same architecture family.

Calibration (per ADR-012): the wrapper returns the bare best estimator.
The training driver applies **sigmoid (Platt) calibration** externally
via :func:`cardiorisk.calibration.calibrate_for_model`. Choice rationale:
Phase 2.3b's isotonic-on-~50-rows recipe collapsed XGBoost to
slope 0.21 (see :doc:`../../../docs/research/08-v1-model-results.md`
§4); the Ensemble's mean-averaged sigmoid output has the same tail-
saturation profile, so we apply the calibration recipe that works at
small calibration-set sizes (Niculescu-Mizil & Caruana 2005).

Determinism: ``torch.manual_seed`` + ``np.random.seed`` are pinned per
``fit``; we do *not* enable ``torch.use_deterministic_algorithms(True)``
because it disables several PyTorch CPU kernels we rely on. The
ensemble outputs are reproducible to ~1e-6 across CPU runs at the same
seed in our smoke tests.
"""

from __future__ import annotations

import warnings
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cardiorisk.features.pipeline import make_woa_pipeline
from cardiorisk.models.base import SEED

#: Default training epochs (matches the Honours notebook). Overridable for smoke.
DEFAULT_N_EPOCHS: Final[int] = 100

#: Smoke-mode epoch budget — enough to flow gradients through every layer
#: without spending CI time on convergence.
SMOKE_N_EPOCHS: Final[int] = 1

#: Mini-batch size (matches the Honours notebook).
DEFAULT_BATCH_SIZE: Final[int] = 32

#: Adam learning rate. Honours notebook uses Keras Adam defaults
#: (lr=1e-3); PyTorch ``torch.optim.Adam`` default is the same.
DEFAULT_LR: Final[float] = 1e-3

#: Dropout rate applied throughout each sub-network (Honours: 0.2 across the board).
DEFAULT_DROPOUT: Final[float] = 0.2


# ---------------------------------------------------------------- sub-nets


class _DNN(nn.Module):
    """Dense feed-forward sub-network (Honours: Dense 100→64→128→1, ReLU, dropout 0.2)."""

    def __init__(self, n_features: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class _CNN(nn.Module):
    """1D convolutional sub-network.

    Honours: Conv1D(64, k=3) → MaxPool(2) → Flatten → Dense(128) → Dense(64) → Dense(1).
    Input shape: ``(batch, n_features, 1)`` in Keras becomes
    ``(batch, 1, n_features)`` in PyTorch (channel-first).
    """

    def __init__(self, n_features: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        # Conv1d(k=3) over n_features -> (n_features - 2); MaxPool1d(2) -> floor((n_features-2)/2)
        self._flat_dim = 64 * ((n_features - 2) // 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_features)
        h = self.conv(x)
        return self.head(h)  # type: ignore[no-any-return]


class _LSTM(nn.Module):
    """LSTM sub-network.

    Honours: ``LSTM(128, dropout=0.2, recurrent_dropout=0.2)`` over an
    n_features-long "sequence" of scalar inputs. PyTorch ``nn.LSTM`` has
    no recurrent-dropout knob; we apply dropout to the output only and
    document the departure in ADR-012.
    """

    def __init__(self, n_features: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, 1) — seq_len=n_features, input_size=1
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))  # type: ignore[no-any-return]


class _BiLSTM(nn.Module):
    """Bidirectional LSTM sub-network. Honours: ``Bidirectional(LSTM(128))``."""

    def __init__(self, n_features: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.n_features = n_features
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional doubles hidden -> 256
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))  # type: ignore[no-any-return]


# ---------------------------------------------------------------- wrapper


class EnsembleModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Honours-Ensemble wrapper conforming to :class:`~cardiorisk.models.base.ModelWrapper`.

    Trains four PyTorch sub-networks independently with binary cross-
    entropy, then mean-averages their sigmoid outputs at inference time.

    Inherits ``ClassifierMixin`` + ``BaseEstimator`` so sklearn 1.8's
    estimator-tags machinery treats it as a classifier — required for
    the ``CalibratedClassifierCV`` Platt-calibration wrapper applied
    externally by the training driver.

    Parameters
    ----------
    n_epochs : int, default=DEFAULT_N_EPOCHS
        Training epochs per sub-network. Honours notebook uses 100.
        Reduce to 1 in ``--smoke`` mode via the constructor argument.
    batch_size : int, default=DEFAULT_BATCH_SIZE
        Mini-batch size (matches Honours).
    learning_rate : float, default=DEFAULT_LR
        Adam learning rate (matches Keras Adam default).
    dropout : float, default=DEFAULT_DROPOUT
        Dropout probability (matches Honours).
    seed : int, default=SEED
        Pinned RNG seed for ``torch.manual_seed``,
        ``np.random.seed``, and the DataLoader shuffling.

    Attributes
    ----------
    preprocessor_ : sklearn.pipeline.Pipeline
        The Phase-2.2 :func:`~cardiorisk.features.pipeline.make_woa_pipeline`
        prefix, fitted on the training slice.
    n_features_in_ : int
        Width of the post-preprocessing feature matrix.
    submodels_ : tuple[nn.Module, ...]
        The four trained sub-networks (DNN, CNN, LSTM, BiLSTM).
    """

    model_name: str = "ensemble"

    def __init__(
        self,
        *,
        n_epochs: int = DEFAULT_N_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LR,
        dropout: float = DEFAULT_DROPOUT,
        seed: int = SEED,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed

    # ----------------------------------------------- internal helpers

    def _seed_everything(self) -> None:
        """Pin every RNG we touch.

        ``torch.use_deterministic_algorithms(True)`` is *not* set — it
        disables several CPU kernels (e.g. some scatter ops in LSTM
        backward) we depend on. The seeded RNGs reproduce outputs to
        ~1e-6 in our smoke tests, which is tight enough for the LODO
        comparison.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _to_tensors(
        self,
        X_dense: np.ndarray,
        y: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pack the preprocessed dense matrix into the three input shapes.

        Returns ``(x_dnn, x_seq, x_seq_for_lstm, y_or_None)`` where:

        - ``x_dnn``: ``(N, n_features)`` for the DNN.
        - ``x_seq``: ``(N, 1, n_features)`` for the CNN (channel-first).
        - ``x_seq_for_lstm``: ``(N, n_features, 1)`` for the LSTMs
          (seq-first in batch-first mode: seq_len=n_features, input=1).
        - ``y``: ``(N, 1)`` float32 for BCEWithLogitsLoss, or ``None``
          at inference time.
        """
        # ``np.ascontiguousarray`` materialises a writable copy when the
        # source array is a sklearn ``ColumnTransformer`` view (which is
        # read-only). ``torch.as_tensor`` on a read-only array emits a
        # ``UserWarning`` that ``filterwarnings='error'`` would escalate
        # to a test failure.
        X_writable = np.ascontiguousarray(X_dense, dtype=np.float32)
        X_t = torch.from_numpy(X_writable)
        x_dnn = X_t
        x_cnn = X_t.unsqueeze(1)  # (N, 1, n_features)
        x_lstm = X_t.unsqueeze(2)  # (N, n_features, 1)
        if y is None:
            return x_dnn, x_cnn, x_lstm, None
        y_writable = np.ascontiguousarray(np.asarray(y), dtype=np.float32)
        y_t = torch.from_numpy(y_writable).view(-1, 1)
        return x_dnn, x_cnn, x_lstm, y_t

    def _train_one_submodel(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> nn.Module:
        """Train a sub-network for ``n_epochs`` with BCEWithLogitsLoss + Adam.

        ``BCEWithLogitsLoss`` is the numerically stable equivalent of
        Keras's ``loss='binary_crossentropy'`` on a sigmoid output (it
        fuses the sigmoid into the loss). The sub-network's final layer
        emits raw logits; we sigmoid only at inference.
        """
        model.train()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        for _epoch in range(self.n_epochs):
            for x_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimiser.step()
        model.eval()
        return model

    # ----------------------------------------------- ModelWrapper API

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> EnsembleModel:
        """Fit the WOA preprocessing prefix + train the four sub-networks."""
        self._seed_everything()

        self.preprocessor_: Pipeline = make_woa_pipeline()
        X_dense_unknown = self.preprocessor_.fit_transform(X, np.asarray(y))
        X_dense = np.asarray(X_dense_unknown, dtype=np.float64)
        self.n_features_in_: int = X_dense.shape[1]

        x_dnn, x_cnn, x_lstm, y_t_or_none = self._to_tensors(X_dense, np.asarray(y))
        # ``_to_tensors`` returns ``y_t`` when called with ``y is not None``;
        # ``cast`` documents that contract for the type-checker without
        # tripping ruff S101 on a runtime assert.
        y_t = cast(torch.Tensor, y_t_or_none)

        # Per the Honours notebook, sub-networks are trained independently.
        # PyTorch's RNN modules emit a benign UserWarning about cudnn / contiguous
        # tensors at fit time on CPU; suppress so it doesn't muddy the driver log.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")
            dnn = self._train_one_submodel(_DNN(self.n_features_in_, self.dropout), x_dnn, y_t)
            cnn = self._train_one_submodel(_CNN(self.n_features_in_, self.dropout), x_cnn, y_t)
            lstm = self._train_one_submodel(_LSTM(self.n_features_in_, self.dropout), x_lstm, y_t)
            bilstm = self._train_one_submodel(
                _BiLSTM(self.n_features_in_, self.dropout), x_lstm, y_t
            )

        self.submodels_: tuple[nn.Module, nn.Module, nn.Module, nn.Module] = (
            dnn,
            cnn,
            lstm,
            bilstm,
        )
        # sklearn convention: classes_ for the calibration wrapper.
        self.classes_: np.ndarray = np.unique(np.asarray(y))
        return self

    def _predict_proba_positive(self, X: pd.DataFrame) -> np.ndarray:
        """Mean-averaged P(y=1) across the four sub-networks."""
        if not hasattr(self, "submodels_"):
            raise RuntimeError("EnsembleModel must be fit before predict_proba()")
        X_dense_unknown = self.preprocessor_.transform(X)
        X_dense = np.asarray(X_dense_unknown, dtype=np.float64)
        x_dnn, x_cnn, x_lstm, _ = self._to_tensors(X_dense, None)
        dnn, cnn, lstm, bilstm = self.submodels_
        with torch.no_grad():
            p_dnn = torch.sigmoid(dnn(x_dnn)).numpy().ravel()
            p_cnn = torch.sigmoid(cnn(x_cnn)).numpy().ravel()
            p_lstm = torch.sigmoid(lstm(x_lstm)).numpy().ravel()
            p_bi = torch.sigmoid(bilstm(x_lstm)).numpy().ravel()
        return np.asarray(np.mean(np.stack([p_dnn, p_cnn, p_lstm, p_bi], axis=0), axis=0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive- and negative-class probabilities, shape ``(n, 2)``."""
        p1 = self._predict_proba_positive(X)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard-class prediction at the standard 0.5 threshold."""
        p1 = self._predict_proba_positive(X)
        return (p1 >= 0.5).astype(np.int64)


def build_ensemble(*, n_epochs: int = DEFAULT_N_EPOCHS) -> EnsembleModel:
    """Factory used by the training driver. ``n_epochs`` overridable for ``--smoke``."""
    return EnsembleModel(n_epochs=n_epochs)
