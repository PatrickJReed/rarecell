"""MSigDB single-gene-set client."""

from __future__ import annotations

import httpx

from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError

MSIGDB_BASE = "https://www.gsea-msigdb.org/gsea/msigdb/cards"


def fetch_msigdb_gene_set(name: str, timeout: float = 15.0) -> RetrievalHit:
    """Fetch a single MSigDB gene set by its standard name."""
    url = f"{MSIGDB_BASE}/{name}.json"
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        raise BackendUnreachableError(f"MSigDB unreachable: {e}") from e
    if r.status_code == 404:
        raise InvalidQueryError(f"MSigDB has no gene set named {name!r}")
    if r.status_code >= 500:
        raise BackendUnreachableError(f"MSigDB returned {r.status_code}")
    r.raise_for_status()
    data = r.json()

    genes = data.get("members") or data.get("geneSymbols") or []
    citation = Citation(
        source_id=f"msigdb:{name}",
        source="msigdb",
        title=data.get("standardName", name),
        url=f"https://www.gsea-msigdb.org/gsea/msigdb/cards/{name}.html",
    )
    return RetrievalHit(
        citation=citation,
        title=data.get("standardName", name),
        snippet=data.get("description", "")[:300],
        payload={"genes": genes, "description": data.get("description"), "pmid": data.get("pmid")},
        source="msigdb",
    )
