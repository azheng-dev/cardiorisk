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

# Phase 3.2 retrieval layout (ADR-016). All gitignored.
#   index/<strategy>/{vector.bin, bm25.pkl, ids.json}
#   embed_cache/<embedder_name>/<chunk_id>.npy
CORPUS_INDEX: Final[Path] = CORPUS_DIR / "index"
CORPUS_EMBED_CACHE: Final[Path] = CORPUS_DIR / "embed_cache"
REPORTS_V1_RETRIEVAL: Final[Path] = REPORTS_V1_DIR / "retrieval"
REPORTS_V1_RETRIEVAL_FIGURES: Final[Path] = REPORTS_V1_FIGURES / "retrieval"

# Phase 3.3 generation eval layout (ADR-017). The reports directory
# is committed (small JSON + 2 PNGs); the smoke variants under
# */smoke/ are gitignored to keep CI noise out of git.
REPORTS_V1_GENERATION: Final[Path] = REPORTS_V1_DIR / "generation"
REPORTS_V1_GENERATION_FIGURES: Final[Path] = REPORTS_V1_FIGURES / "generation"

# Phase 4 agent eval layout (ADR-018). 30-case eval over synthetic
# patients; per-case + aggregate JSONs + 3 figures (success-rate
# by stage, latency p50/p95, suppression-by-stage). Smoke variants
# under */smoke/ are gitignored.
REPORTS_V1_AGENTS: Final[Path] = REPORTS_V1_DIR / "agents"
REPORTS_V1_AGENTS_FIGURES: Final[Path] = REPORTS_V1_FIGURES / "agents"
