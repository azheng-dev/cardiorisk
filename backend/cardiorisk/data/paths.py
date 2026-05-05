"""Repo-relative path constants for the data layer.

All paths are absolute, computed from this module's location, so they work
regardless of the caller's current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# This file is at backend/cardiorisk/data/paths.py — repo root is 3 levels up.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_RAW: Final[Path] = REPO_ROOT / "data" / "raw"
DATA_PROCESSED: Final[Path] = REPO_ROOT / "data" / "processed"
DATA_CHECKSUMS: Final[Path] = REPO_ROOT / "data" / "checksums"
FIXTURE_PATH: Final[Path] = REPO_ROOT / "backend" / "tests" / "fixtures" / "hfp_mini.csv"
COMBINED_PARQUET: Final[Path] = DATA_PROCESSED / "combined.parquet"

# Phase 2.3b model artefact + reports layout (ADR-010).
MODELS_V1_DIR: Final[Path] = REPO_ROOT / "models" / "v1"
REPORTS_V1_DIR: Final[Path] = REPO_ROOT / "reports" / "v1"
REPORTS_V1_FIGURES: Final[Path] = REPORTS_V1_DIR / "figures"
