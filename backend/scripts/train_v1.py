"""CLI wrapper for the Phase 2.3b training driver.

Logic lives in :mod:`cardiorisk.training.train_v1`. This script's only
responsibilities are:

1. Set the macOS OpenMP guards BEFORE any model wrapper is imported.
   On macOS, xgboost (linked against brew libomp) and torch (linked
   against its own bundled libomp via TabICL) deadlock when loaded
   in the same process unless we constrain OpenMP up-front. Linux CI
   is unaffected.
2. Constrain torch to a single thread (CPU-only inference; matches
   ``OMP_NUM_THREADS=1``).
3. Hand off to :func:`cardiorisk.training.train_v1.main`.

Usage::

    # CI / smoke (~15s):
    uv run --project backend python backend/scripts/train_v1.py --smoke

    # Full local run (~30-50 min CPU):
    uv run --project backend python backend/scripts/train_v1.py --full
"""

from __future__ import annotations

import os

# MUST come before any sklearn / xgboost / torch / TabICL import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

torch.set_num_threads(1)

from cardiorisk.training.train_v1 import main  # noqa: E402

if __name__ == "__main__":
    main()
