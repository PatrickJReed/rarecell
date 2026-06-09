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

## Building the CNS reference bundle (maintainers / power users)

The CNS reference bundle (per-level CellTypist models + marker panels) is built
**offline, once per BICCN release**, and published as a GitHub release asset.
End users never run this — they fetch the small bundle at runtime.

```bash
uv sync --group build-reference          # installs httpx (build-only)
uv run python -m scripts.build_cns_reference \
    --out ./cns-reference-WHB-2023 \
    --cache-dir ./biccn_cache \
    --cells-per-class 5000 --min-donors 10
```

This downloads the BICCN Human Brain Cell Atlas v1.0 H5ADs from the CELLxGENE
Discover collection (`283d65eb-...`), balanced-subsamples to ~equal cells per
class, trains the 31-way supercluster model plus per-supercluster cluster
models, and writes a bundle directory. Upload the directory (tarred) as a
GitHub release asset; pin its tag as `reference_release` in profiles.

**Power-user retrain:** point `--out` at a local path and load it at runtime via
`reference_release="local:<path>"` (see Plan 2 runtime) to use a custom bundle.
