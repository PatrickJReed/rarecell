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

Once pre-commit configuration lands (see Plan 1, Task 24), install the
hooks with:

```bash
uv run pre-commit install
```

## Conduct

By participating, you agree to abide by `CODE_OF_CONDUCT.md`.
