"""Session-wide test fixtures + macOS OpenMP guard.

Set ``OMP_NUM_THREADS=1`` and ``KMP_DUPLICATE_LIB_OK=TRUE`` BEFORE any
test imports a model wrapper. Phase 2.3b adds three model wrappers
that each pull large native libs (xgboost links libomp via brew, torch
ships its own libomp via TabICL); on macOS Apple Silicon, loading both
in one process deadlocks unless we constrain OpenMP up-front. Linux CI
is unaffected because both libs use libgomp consistently there.

This file runs at pytest collection time, before any test module is
imported, so model wrappers never see the unconstrained env.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Constrain torch threading too. Set after the env var so torch's own
# init reads the right value, but also defensively re-set after import.
import torch

torch.set_num_threads(1)
