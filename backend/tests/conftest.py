"""Session-wide test fixtures + macOS OpenMP guard + clean-shutdown hook.

Set ``OMP_NUM_THREADS=1`` and ``KMP_DUPLICATE_LIB_OK=TRUE`` BEFORE any
test imports a model wrapper. Phase 2.3b adds three model wrappers
that each pull large native libs (xgboost links libomp via brew, torch
ships its own libomp via TabICL); on macOS Apple Silicon, loading both
in one process deadlocks unless we constrain OpenMP up-front. Linux CI
is unaffected because both libs use libgomp consistently there.

This file runs at pytest collection time, before any test module is
imported, so model wrappers never see the unconstrained env.

Phase 3.2 amendment 2026-05-15: Linux CI runners segfault during
interpreter shutdown (exit code 139) after all 617 tests pass. The
segfault is in PyTorch's destructor chain (sentence-transformers /
FlagEmbedding cross-encoder weights are GC'd against torch's CUDA
shutdown sequence even on the CPU-only wheel). Tests themselves are
green; only the post-pytest interpreter teardown fails. We hook
``pytest_sessionfinish`` to force a clean exit after pytest has
finished reporting, which sidesteps the broken __del__ chain.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Constrain torch threading too. Set after the env var so torch's own
# init reads the right value, but also defensively re-set after import.
import torch

torch.set_num_threads(1)


_PYTEST_EXITSTATUS: int = 0


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Capture the final pytest exit status for the unconfigure hook."""
    global _PYTEST_EXITSTATUS
    _PYTEST_EXITSTATUS = int(exitstatus)


def pytest_unconfigure(config: object) -> None:
    """Force a clean process exit to skip torch's segfault-prone shutdown.

    Pytest's normal exit triggers Python's full GC + atexit chain, and
    sentence-transformers / FlagEmbedding cross-encoder __del__ paths can
    segfault during torch tensor cleanup. ``pytest_unconfigure`` runs
    after the terminal summary has been printed, so ``os._exit`` is safe.

    Skip the shortcut when ``CARDIORISK_NO_FAST_EXIT=1`` so a developer
    debugging the actual shutdown path can still see it.
    """
    if os.environ.get("CARDIORISK_NO_FAST_EXIT") == "1":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXITSTATUS)
