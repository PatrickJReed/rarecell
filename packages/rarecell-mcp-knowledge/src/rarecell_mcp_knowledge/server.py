"""FastMCP server: wires up literature + markers + Enrichr + MSigDB tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rarecell_mcp_knowledge.cache import QueryCache
from rarecell_mcp_knowledge.citation import RetrievalHit
from rarecell_mcp_knowledge.enrichr import enrichr_enrich
from rarecell_mcp_knowledge.literature.europepmc import EuropePMCClient
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.msigdb import fetch_msigdb_gene_set


def _hash(query: dict) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()


def _serialize(hit_or_hits):
    if isinstance(hit_or_hits, RetrievalHit):
        return hit_or_hits.model_dump(mode="json")
    return [h.model_dump(mode="json") for h in hit_or_hits]


class KnowledgeApp:
    """Tool implementations + state (catalog, cache, clients).

    Used by both the in-process test wrapper and the production FastMCP server.
    """

    def __init__(
        self,
        catalog: MarkersCatalog,
        cache: QueryCache,
        literature: EuropePMCClient | None = None,
    ):
        self.catalog = catalog
        self.cache = cache
        self.literature = literature or EuropePMCClient()

    def search_literature(
        self,
        query: str,
        year_range: list[int] | None = None,
        tissue: str | None = None,
        page_size: int = 10,
    ):
        key = _hash({"q": query, "yr": year_range, "tissue": tissue, "ps": page_size})
        cached = self.cache.get("europepmc", key)
        if cached is not None:
            return cached
        yr = tuple(year_range) if year_range else None
        hits = self.literature.search(query, year_range=yr, tissue=tissue, page_size=page_size)
        out = _serialize(hits)
        self.cache.set("europepmc", key, out)
        return out

    def fetch_abstract(self, pmid_or_doi: str):
        key = _hash({"fetch": pmid_or_doi})
        cached = self.cache.get("europepmc", key)
        if cached is not None:
            return cached
        hit = self.literature.fetch_abstract(pmid_or_doi)
        out = _serialize(hit)
        self.cache.set("europepmc", key, out)
        return out

    def search_markers(self, cell_type: str, tissue: str | None = None):
        return _serialize(self.catalog.search_markers(cell_type, tissue))

    def get_canonical_panel(self, name: str):
        return _serialize(self.catalog.get_canonical_panel(name))

    def enrichr_enrich(self, genes: list[str], library: str):
        key = _hash({"genes": sorted(genes), "lib": library})
        cached = self.cache.get("enrichr", key)
        if cached is not None:
            return cached
        hits = enrichr_enrich(genes=genes, library=library)
        out = _serialize(hits)
        self.cache.set("enrichr", key, out)
        return out

    def fetch_msigdb_gene_set(self, name: str):
        key = _hash({"msigdb": name})
        cached = self.cache.get("msigdb", key)
        if cached is not None:
            return cached
        hit = fetch_msigdb_gene_set(name)
        out = _serialize(hit)
        self.cache.set("msigdb", key, out)
        return out


def build_app(*, catalog: MarkersCatalog, cache_path: Path):
    """Build an in-process app object exposing the tools.

    Returns an object with:
      .list_tool_names() -> list[str]
      .call_tool(name, kwargs) -> result
      .knowledge_app          # the KnowledgeApp instance
      .tool_map               # {name: callable}
    """
    cache = QueryCache(cache_path)
    knowledge = KnowledgeApp(catalog=catalog, cache=cache)

    tools = {
        "search_literature": knowledge.search_literature,
        "fetch_abstract": knowledge.fetch_abstract,
        "search_markers": knowledge.search_markers,
        "get_canonical_panel": knowledge.get_canonical_panel,
        "enrichr_enrich": knowledge.enrichr_enrich,
        "fetch_msigdb_gene_set": knowledge.fetch_msigdb_gene_set,
    }

    class _App:
        knowledge_app = knowledge
        tool_map = tools

        def list_tool_names(self) -> list[str]:
            return list(tools.keys())

        def call_tool(self, name: str, kwargs: dict):
            return tools[name](**kwargs)

    return _App()


def build_fastmcp_app(*, catalog: MarkersCatalog, cache_path: Path):
    """Build the production FastMCP server with tools registered."""
    from fastmcp import FastMCP

    inner = build_app(catalog=catalog, cache_path=cache_path)
    mcp = FastMCP("rarecell-mcp-knowledge")
    for name, fn in inner.tool_map.items():
        mcp.tool(name)(fn)
    return mcp
