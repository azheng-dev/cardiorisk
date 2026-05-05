"""CLI: fetch the HFP source data (UCI primary, Kaggle optional).

Thin wrapper around :mod:`cardiorisk.data.fetch`. Downloads the four UCI
Heart Disease subsets, verifies their pinned SHA-256 lockfiles, and
optionally fetches the Kaggle HFP combined CSV for cross-check.

Modes:
    --use-fixture     skip network entirely; use the synthetic fixture
                      (CI smoke-test mode).
    --include-kaggle  also fetch the Kaggle HFP combined file (requires
                      KAGGLE_USERNAME + KAGGLE_KEY env vars).
    --force           re-download even if checksums match.

Usage:
    uv run python backend/scripts/fetch_hfp.py
    uv run python backend/scripts/fetch_hfp.py --use-fixture
    uv run python backend/scripts/fetch_hfp.py --include-kaggle
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import requests

from cardiorisk.data.fetch import (
    FetchError,
    fetch_all_uci,
    fetch_kaggle,
    use_fixture,
)
from cardiorisk.data.paths import REPO_ROOT


def _format_action(items: Iterable[tuple[str, Path, str, str]]) -> str:
    return "\n".join(
        f"  [{action}] {name:<13} -> {path.relative_to(REPO_ROOT)}  sha256={digest[:12]}..."
        for name, path, digest, action in items
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="skip network; use the synthetic fixture (CI mode)",
    )
    parser.add_argument(
        "--include-kaggle",
        action="store_true",
        help="also fetch Kaggle HFP combined (requires KAGGLE_USERNAME / KAGGLE_KEY)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if checksums match",
    )
    args = parser.parse_args()

    if args.use_fixture:
        try:
            target = use_fixture()
        except FetchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"  [fixture]    hfp_mini      -> {target.relative_to(REPO_ROOT)}")
        return 0

    try:
        uci_results = fetch_all_uci(force=args.force)
    except (FetchError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("UCI Heart Disease subsets:")
    print(
        _format_action(
            (source.name, path, digest, action) for source, path, digest, action in uci_results
        )
    )

    if args.include_kaggle:
        try:
            kaggle_path, kaggle_digest, kaggle_action = fetch_kaggle(force=args.force)
        except FetchError as exc:
            print(f"ERROR fetching Kaggle: {exc}", file=sys.stderr)
            return 1
        print("Kaggle HFP combined:")
        print(_format_action([("hfp_kaggle", kaggle_path, kaggle_digest, kaggle_action)]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
