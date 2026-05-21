# rarecell-mcp-knowledge

FastMCP server exposing literature retrieval (Europe PMC) and marker-database
retrieval (CellMarker 2.0, PanglaoDB, MSigDB, Enrichr) behind a single MCP
surface. Designed to be consumed by the `rarecell` agent or any MCP client.

## Install

```bash
pip install rarecell-mcp-knowledge
```

## Run the server

```bash
rarecell-mcp-knowledge serve
```

## Seed the local marker catalog

On first run, the local SQLite catalog needs to be seeded with CellMarker 2.0
+ PanglaoDB data:

```bash
rarecell-mcp-knowledge seed
```

This downloads ~50 MB of TSVs and builds a local SQLite at
`~/.cache/rarecell/markers.sqlite`.

## Tools advertised

- `search_literature(query, year_range?, tissue?)`
- `fetch_abstract(pmid_or_doi)`
- `search_markers(cell_type, tissue?)`
- `get_canonical_panel(name)`
- `enrichr_enrich(genes, library)`

This is pre-release v0.x.
