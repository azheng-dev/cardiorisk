"""CLI wrapper for the Phase 2.5 explainability driver.

Logic lives in :mod:`cardiorisk.explainability.orchestrator`. This
script's only responsibilities are:

1. Set the macOS OpenMP guards BEFORE any model wrapper is imported
   (xgboost links libomp via brew, torch ships its own libomp via
   TabICL; loading both in one process deadlocks unless we constrain
   OpenMP up-front). Linux CI is unaffected.
2. Constrain torch to a single thread.
3. Hand off to :func:`cardiorisk.explainability.orchestrator.main`.

Usage::

    # CI / smoke (~30s):
    uv run --project backend python backend/scripts/compute_explanations.py --smoke

    # Full local run (~30-60 min CPU per ADR-013):
    uv run --project backend python backend/scripts/compute_explanations.py
"""

from __future__ import annotations

import os

# MUST come before any sklearn / xgboost / torch / TabICL import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

torch.set_num_threads(1)

from cardiorisk.explainability.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    main()
