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
- `packages/rarecell-mcp-knowledge/` — FastMCP server for literature + marker
  retrieval (CellMarker, PanglaoDB, MSigDB, Enrichr, Europe PMC). Consumable
  from any MCP client.
- `packages/rarecell-mcp/` — exposed FastMCP workflow server. Drive
  `draft | validate | isolate | inspect` from Claude Desktop / Claude
  Code / Cursor.

Design docs and implementation plans live under `docs/`.

## Getting started

```bash
uv sync --all-packages --all-extras --dev
uv run pytest
```

See `CONTRIBUTING.md` for the full development workflow.

## Advisor agent (Plan 3 — `rarecell.agent`)

The advisor experience lives in `rarecell.agent` and is gated by the
`[agent]` optional extra:

```bash
pip install 'rarecell[agent]'
```

This installs the Anthropic SDK and `rarecell-mcp-knowledge`. The agent
provides `ClaudeRecommender` (LLM-backed swap-in for the heuristic
`BasicRecommender`) and a profile drafting flow that turns natural-language
prompts into reviewable `TargetCellProfile` YAMLs.

Without the extra, `rarecell.core` works unchanged with `BasicRecommender`.

## CLI

After `pip install rarecell`:

```bash
rarecell isolate --input adata.h5ad --profile profile.yaml --out-dir runs/run1
rarecell draft --prompt "rare T cells in PBMC" --out draft.yaml
rarecell review --report runs/run1
```

Drafting requires the `[agent]` extra and `ANTHROPIC_API_KEY` in the environment.

## Driving from an MCP client

Install `rarecell-mcp` and wire it into Claude Desktop or Claude Code:

```bash
pip install rarecell-mcp
```

```json
{"mcpServers": {"rarecell": {"command": "rarecell-mcp", "args": ["serve"]}}}
```
