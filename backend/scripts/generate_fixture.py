"""CLI: generate the synthetic HFP-schema test fixture.

Thin wrapper around :mod:`cardiorisk.data.synthetic`. The fixture is the
only tabular data file committed to the public repo and is byte-for-byte
deterministic given the seed.

Usage:
    uv run python backend/scripts/generate_fixture.py
    uv run python backend/scripts/generate_fixture.py --n 50 --seed 42
    uv run python backend/scripts/generate_fixture.py \\
        --out backend/tests/fixtures/hfp_mini.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cardiorisk.data.paths import FIXTURE_PATH, REPO_ROOT
from cardiorisk.data.synthetic import generate_fixture, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=20, help="number of synthetic rows (default: 20)")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260505,
        help="PRNG seed (default: 20260505 — locked for reproducibility)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIXTURE_PATH,
        help=f"output CSV path (default: {FIXTURE_PATH.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    rows = generate_fixture(n=args.n, seed=args.seed)
    write_csv(rows=rows, out_path=args.out)

    n_missing_chol = sum(1 for r in rows if r["Cholesterol"] == 0)
    n_positive = sum(1 for r in rows if r["HeartDisease"] == 1)
    print(
        f"wrote {len(rows)} synthetic rows to {args.out} "
        f"(seed={args.seed}, "
        f"chol=0 in {n_missing_chol} rows, "
        f"HeartDisease=1 in {n_positive} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
