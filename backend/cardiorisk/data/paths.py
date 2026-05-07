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

# Phase 2.5 explainability layout (ADR-013).
REPORTS_V1_EXPLAIN: Final[Path] = REPORTS_V1_DIR / "explainability"
REPORTS_V1_EXPLAIN_FIGURES: Final[Path] = REPORTS_V1_FIGURES / "explainability"

# Phase 2.6 drift / monitoring layout (ADR-014).
REPORTS_V1_DRIFT: Final[Path] = REPORTS_V1_DIR / "drift"
REPORTS_V1_DRIFT_FIGURES: Final[Path] = REPORTS_V1_FIGURES / "drift"

# Phase 3.1 corpus ingestion layout (ADR-015). All gitignored except the
# sha256 lockfiles in DATA_CHECKSUMS.
DATA_EXTERNAL: Final[Path] = REPO_ROOT / "data" / "external"
CORPUS_DIR: Final[Path] = DATA_EXTERNAL / "corpus"
CORPUS_RAW: Final[Path] = CORPUS_DIR / "raw"
CORPUS_PARSED: Final[Path] = CORPUS_DIR / "parsed"
CORPUS_CHUNKS: Final[Path] = CORPUS_DIR / "chunks"
CORPUS_MANIFEST: Final[Path] = CORPUS_DIR / "manifest.json"
# Markdown fixture used by --use-fixture (CI + unit tests). PDF-free,
# network-free; exercises every chunker through the same ParsedDoc
# schema as the real PDF path.
FIXTURE_CORPUS_DIR: Final[Path] = REPO_ROOT / "backend" / "tests" / "fixtures" / "corpus_mini"
