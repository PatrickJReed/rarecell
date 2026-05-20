# rarecell

rarecell is an agentic toolkit for isolating rare cell populations from
single-cell RNA-seq data. It pairs standard scanpy-based preprocessing,
batch integration, and clustering with a typed Python core that an LLM
agent can drive end-to-end against an input `.h5ad` dataset.

**Status: pre-release (v0.x).** APIs, file layouts, and CLI surfaces are
unstable and may change without notice until v1.0.

## Monorepo layout

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/).
Packages live under `packages/`:

- `packages/rarecell/` — the core library (preprocessing, integration,
  clustering, rare-cell scoring, marker discovery).

Design docs and implementation plans live under `docs/`.

## Getting started

```bash
uv sync --all-packages --all-extras --dev
uv run pytest
```

See `CONTRIBUTING.md` for the full development workflow.
