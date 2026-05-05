#!/usr/bin/env bash
# Refuse to commit raw data: anything under data/raw/, or any *.csv outside
# explicitly-allowed fixture directories. Keeps the public repo from ever
# absorbing clinical data, even if .gitignore is bypassed.
#
# Allowed paths for CSVs:
#   - backend/tests/fixtures/**/*.csv  (synthetic test fixtures only)
#   - frontend/tests/fixtures/**/*.csv
set -euo pipefail

violations=()

for f in "$@"; do
  # Allowlist: empty .gitkeep markers are fine (they only mark the dir).
  case "$(basename "$f")" in
    .gitkeep) continue ;;
  esac

  case "$f" in
    data/raw/*|data/interim/*|data/external/*|data/processed/*)
      violations+=("$f (under data/ — never commit raw data)")
      ;;
    *.csv|*.tsv|*.parquet|*.feather)
      case "$f" in
        backend/tests/fixtures/*|frontend/tests/fixtures/*|tests/fixtures/*)
          ;;
        *)
          violations+=("$f (data file outside tests/fixtures/)")
          ;;
      esac
      ;;
  esac
done

if [ ${#violations[@]} -gt 0 ]; then
  echo "ERROR: Refusing to commit data files. Move them into data/raw/ (gitignored)" >&2
  echo "       or, if they are synthetic fixtures, into tests/fixtures/." >&2
  echo "" >&2
  for v in "${violations[@]}"; do
    echo "  - $v" >&2
  done
  exit 1
fi
