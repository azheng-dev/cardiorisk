# ADR-002: Python tooling — uv, Ruff, mypy strict, pytest

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Decision

For the Python backend:

- **Package management + virtualenvs:** [`uv`](https://docs.astral.sh/uv/) (Astral).
- **Lint + format:** [`ruff`](https://docs.astral.sh/ruff/) (replaces flake8, isort, pyupgrade, black).
- **Type check:** `mypy` with `strict = true`.
- **Test runner:** `pytest` with `filterwarnings = ["error"]` and strict markers.

Python version: **3.12**, pinned in `backend/pyproject.toml` as `>=3.12,<3.13`.

## Context

Python tooling has consolidated significantly in 2024–2026. `uv` (resolver in Rust, ~10–100× faster than pip) and `ruff` (linter/formatter in Rust, replaces 5+ tools) are now the defaults at most fast-moving Python shops. `mypy strict` is the most widely-understood type-checker — `pyright` is a credible alternative but ties more tightly to VS Code.

## Consequences

- **Positive:** very fast local + CI loop. `uv sync` finishes in seconds; `ruff check .` finishes in under a second on a small codebase. Lower CI bill.
- **Positive:** one config file (`pyproject.toml`) for everything except `pre-commit`. Less surface to misconfigure.
- **Positive:** `ruff` enables a strong default ruleset (E, F, I, B, C4, UP, N, S, SIM, RUF) without slowing the loop.
- **Negative:** uv is young (1.x); breaking changes more likely than with pip. Mitigated by pinning the uv version in CI.
- **Negative:** `filterwarnings = ["error"]` makes deprecation warnings fail tests. This is a feature (you find them early) but adds friction when upstream deps add deprecations.

## Alternatives considered

- **`poetry`.** Rejected: slower, more opinionated, lock-file format makes it harder to install one-off scripts. Was the right choice in 2022; in 2026 the momentum is clearly with uv.
- **`pip-tools` + `venv`.** Rejected: workable but slower and more steps.
- **`pyright` instead of `mypy`.** Rejected: marginally faster, slightly stricter, but worse third-party type-stub coverage in the data-science ecosystem we're heading into.
- **`black` + `isort` + `flake8`.** Rejected: three tools where one (`ruff`) suffices.

## Trigger to revisit

- A `uv` breaking change costs more than two days in CI debugging.
- We need a feature `mypy` lacks (e.g. better narrowing on `TypedDict`s) and `pyright` ships it cleanly.
