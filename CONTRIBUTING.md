# Contributing

Thanks for your interest in rarecell. This project is in early
pre-release; expect rough edges.

## Development setup

This repo uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Install uv, then from the repo root:

```bash
uv sync --all-packages --all-extras --dev
```

This creates `.venv/` and installs every workspace package with its
dev extras.

## Running tests

```bash
uv run pytest
```

## Pre-commit hooks

Install pre-commit hooks (ruff lint + format + mypy on src/):

```bash
uv run pre-commit install
```

This will lint and type-check on every commit.

If a hook fails, fix the reported issue and re-stage. Do not bypass with `--no-verify`.

## Conduct

By participating, you agree to abide by `CODE_OF_CONDUCT.md`.
