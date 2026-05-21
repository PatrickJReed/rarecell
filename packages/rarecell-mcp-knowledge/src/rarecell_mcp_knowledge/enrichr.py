"""Enrichr REST client."""

from __future__ import annotations

import httpx

from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"


def enrichr_enrich(
    *,
    genes: list[str],
    library: str,
    timeout: float = 15.0,
) -> list[RetrievalHit]:
    """Submit a gene list to Enrichr and return enrichment hits."""
    add_url = f"{ENRICHR_BASE}/addList"
    enrich_url = f"{ENRICHR_BASE}/enrich"
    try:
        post = httpx.post(
            add_url,
            files={"list": (None, "\n".join(genes)), "description": (None, "rarecell")},
            timeout=timeout,
        )
        post.raise_for_status()
        user_list_id = post.json()["userListId"]
        r = httpx.get(
            enrich_url,
            params={"userListId": user_list_id, "backgroundType": library},
            timeout=timeout,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise BackendUnreachableError(f"Enrichr unreachable: {e}") from e

    data = r.json().get(library, [])
    hits: list[RetrievalHit] = []
    for row in data:
        rank, term, pvalue, zscore, combined, overlap, adj_p, *_ = row
        citation = Citation(
            source_id=f"enrichr:{library}:{term}",
            source="enrichr",
            title=term,
        )
        hits.append(
            RetrievalHit(
                citation=citation,
                title=term,
                snippet=f"pvalue={pvalue:.2e}, combined_score={combined:.1f}",
                payload={
                    "rank": rank,
                    "pvalue": pvalue,
                    "zscore": zscore,
                    "combined_score": combined,
                    "overlap_genes": overlap,
                    "adjusted_pvalue": adj_p,
                    "library": library,
                },
                source="enrichr",
            )
        )
    return hits
