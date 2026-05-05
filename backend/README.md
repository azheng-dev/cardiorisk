# Backend

FastAPI service, ML models, and agentic orchestration. Pre-alpha.

## Layout

```
backend/
  cardiorisk/         # importable package (empty in Phase 0)
  tests/              # pytest tree
  pyproject.toml      # project metadata + tool config (ruff, mypy, pytest)
```

## Local commands

```bash
# from repo root
cd backend

uv sync                    # install deps + dev deps
uv run pytest -q           # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy .              # type-check (strict)
```

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the full local CI sequence.
