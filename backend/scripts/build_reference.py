"""One-shot: build per-fold reference snapshots for the v1 stack.

For each LODO fold (4 outer iterations on the four UCI sources), this
script:

1. Loads ``data/processed/combined.parquet`` (fail loudly if absent).
2. Reproduces the same within-fold split the training driver used.
3. Loads each calibrated model artefact for that fold.
4. Builds a :class:`~cardiorisk.monitoring.reference.FoldReference`
   from the in-fold training rows + each model's ``predict_proba`` on
   that same training pool.
5. Persists the reference to ``models/v1/<source>_reference.joblib``
   (gitignored, per ADR-010 + ADR-014).

The drift orchestrator (``compute_drift.py``) doesn't actually need
these on-disk references — it builds equivalent references in memory
during its own LODO sweep. They exist for the production-monitoring
use case where new "current" data arrives at deploy-time and a
standalone script needs to score drift against the reference *the
deployed model was trained on* without rerunning the full LODO loop.

Usage::

    uv run --project backend python backend/scripts/build_reference.py
"""

from __future__ import annotations

import argparse
import logging
import os

# MUST come before any sklearn / xgboost / torch / TabICL import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

torch.set_num_threads(1)

import joblib  # noqa: E402

from cardiorisk.data.paths import COMBINED_PARQUET, MODELS_V1_DIR  # noqa: E402
from cardiorisk.data.preprocess import clean_for_modelling  # noqa: E402
from cardiorisk.features.cv import (  # noqa: E402
    SOURCE_COLUMN,
    TARGET_COLUMN,
    iter_lodo_folds,
    within_fold_split,
)
from cardiorisk.models.base import MODEL_NAMES, SEED  # noqa: E402
from cardiorisk.monitoring.reference import (  # noqa: E402
    DEFAULT_N_BINS,
    build_fold_reference,
    save_reference,
)

logger = logging.getLogger("scripts.build_reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not COMBINED_PARQUET.exists():
        raise FileNotFoundError(
            f"missing {COMBINED_PARQUET}. Run "
            "`uv run --project backend python backend/scripts/build_combined.py` first."
        )

    import pandas as pd

    df = pd.read_parquet(COMBINED_PARQUET)
    df = clean_for_modelling(df)
    y_full = df[TARGET_COLUMN].to_numpy()
    X_full = df.drop(columns=[TARGET_COLUMN, SOURCE_COLUMN])

    for fold in iter_lodo_folds(df):
        logger.info(
            "fold: held out source=%s (n_train=%d)",
            fold.held_out_source,
            fold.train_idx.size,
        )
        wf = within_fold_split(fold.train_idx, y_full, seed=args.seed)
        X_train = X_full.iloc[wf.train_idx]

        models = {}
        for model_name in MODEL_NAMES:
            artefact = MODELS_V1_DIR / f"{model_name}_{fold.held_out_source}.joblib"
            if not artefact.exists():
                raise FileNotFoundError(
                    f"missing calibrated artefact {artefact}. "
                    "Run `uv run --project backend python backend/scripts/train_v1.py` first."
                )
            models[model_name] = joblib.load(artefact)

        ref = build_fold_reference(
            held_out_source=fold.held_out_source,
            X_train=X_train,
            models=models,
            n_bins=args.n_bins,
        )
        out_path = MODELS_V1_DIR / f"{fold.held_out_source}_reference.joblib"
        save_reference(ref, out_path)
        logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
