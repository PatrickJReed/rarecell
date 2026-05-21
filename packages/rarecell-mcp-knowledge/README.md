# rarecell-mcp-knowledge

FastMCP server exposing literature retrieval (Europe PMC) and marker-database
retrieval (CellMarker 2.0, PanglaoDB, MSigDB, Enrichr) behind a single MCP
surface.

## Install

```bash
pip install rarecell-mcp-knowledge
```

## Seed the local marker catalog

The local SQLite catalog (`~/.cache/rarecell/markers.sqlite`) is built from
CellMarker 2.0 + PanglaoDB TSV downloads:

```bash
rarecell-mcp-knowledge seed \
  --cellmarker-tsv /path/to/Cell_marker_Human.tsv \
  --panglaodb-tsv /path/to/PanglaoDB_markers_27_Mar_2020.tsv
```

Source URLs (download manually for v0.1):
- CellMarker 2.0: <http://yikedaxue.slwshop.cn/CellMarker_download_files/file/Cell_marker_Human.xlsx>
- PanglaoDB: <https://panglaodb.se/markers.html>

## Run the server

```bash
rarecell-mcp-knowledge serve
```

Stdio MCP server. Wire into Claude Desktop / Claude Code by adding this to
the MCP client config:

```json
{
  "mcpServers": {
    "rarecell-knowledge": {
      "command": "rarecell-mcp-knowledge",
      "args": ["serve"]
    }
  }
}
```

## Tools advertised

| Tool | Purpose |
|------|---------|
| `search_literature(query, year_range?, tissue?, page_size?)` | Europe PMC search |
| `fetch_abstract(pmid_or_doi)` | Fetch a single abstract |
| `search_markers(cell_type, tissue?)` | Local catalog query |
| `get_canonical_panel(name)` | Aggregate marker panel across sources |
| `enrichr_enrich(genes, library)` | Enrichr gene set enrichment |
| `fetch_msigdb_gene_set(name)` | MSigDB single gene set lookup |

## Cache

Query results are cached at `~/.cache/rarecell/mcp_knowledge.sqlite` with a
30-day TTL. Delete the file to force fresh fetches.

## Status

Pre-release v0.x.
