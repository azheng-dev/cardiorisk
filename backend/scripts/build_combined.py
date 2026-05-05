"""CLI: combine the UCI subsets into one HFP-schema parquet.

Thin wrapper around :mod:`cardiorisk.data.combine`. Reads the UCI files
fetched by ``scripts/fetch_hfp.py`` (or the synthetic fixture in
``--use-fixture`` mode), maps to the HFP schema with a ``source`` column,
and writes a parquet to ``data/processed/combined.parquet``.

Usage:
    uv run python backend/scripts/build_combined.py
    uv run python backend/scripts/build_combined.py --use-fixture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cardiorisk.data.combine import (
    DEFAULT_OUTPUT,
    CombineError,
    build_from_fixture,
    build_from_uci,
    write_combined,
)
from cardiorisk.data.paths import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="combine from the synthetic fixture instead of UCI files (CI mode)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output parquet path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    try:
        combined = build_from_fixture() if args.use_fixture else build_from_uci()
    except CombineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_combined(combined, args.out)

    print(
        f"wrote {len(combined):,} rows x {len(combined.columns)} cols -> "
        f"{args.out.relative_to(REPO_ROOT)}"
    )
    counts = combined["source"].value_counts().sort_index()
    print("rows per source:")
    for source_name, count in counts.items():
        print(f"  {source_name:<12}  {count:>5}")

    pos_rate = float(combined["HeartDisease"].mean())
    print(f"target rate: {pos_rate:.1%} positive")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
