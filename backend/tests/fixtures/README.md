# Test fixtures

Synthetic data only. **No real patient records, ever.** This is the only directory in the repo where tabular data files are allowed (enforced by the `no-raw-data` pre-commit hook in `scripts/no_raw_data.sh` and gated again by branch protection's `secret-scan` CI job).

## Files

| File | Generator | Schema | Rows | Seed |
|---|---|---|---|---|
| `hfp_mini.csv` | [`backend/scripts/generate_fixture.py`](../../scripts/generate_fixture.py) | Heart Failure Prediction (Kaggle, fedesoriano) | 20 | `20260505` |

## Regenerating

```bash
uv run python backend/scripts/generate_fixture.py
```

The output is byte-for-byte deterministic given the seed. If you change the generator, regenerate the fixture and commit both in the same PR.

## Why a fixture at all

CI must be able to run end-to-end (fetch script, build script, EDA notebook execution, schema tests) without network access or Kaggle credentials. The fixture is what `--use-fixture` mode points at. It mirrors the HFP schema column-for-column so any test that passes against `hfp_mini.csv` is a valid smoke test for the same code paths against the real dataset.

## What this fixture is not

- **Not training data.** 20 rows is far below what any model needs.
- **Not representative.** The PRNG draws from broad uniform / weighted-categorical distributions; per-feature distributions are *not* calibrated against the real HFP marginals.
- **Not a benchmark.** Don't report metrics on it.

The only contract is: **same column names, same dtypes, same sentinel values (e.g. `Cholesterol == 0` marking "missing")** as the real HFP CSV.
