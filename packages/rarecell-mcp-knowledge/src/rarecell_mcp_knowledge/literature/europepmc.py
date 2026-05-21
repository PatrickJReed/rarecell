"""Europe PMC REST client."""

from __future__ import annotations

import httpx

from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _record_to_hit(rec: dict) -> RetrievalHit:
    pmid = rec.get("pmid") or rec.get("id") or ""
    title = rec.get("title", "")
    abstract = rec.get("abstractText", "")
    doi = rec.get("doi")
    url = f"https://europepmc.org/article/MED/{pmid}" if pmid else None
    citation = Citation(
        source_id=f"pmid:{pmid}" if pmid else f"doi:{doi}",
        source="europepmc",
        title=title or None,
        url=url,
    )
    return RetrievalHit(
        citation=citation,
        title=title,
        snippet=abstract,
        payload={
            "doi": doi,
            "year": rec.get("pubYear"),
            "authors": rec.get("authorString"),
        },
        source="europepmc",
    )


class EuropePMCClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0):
        self.base_url = base_url
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict) -> dict:
        try:
            r = httpx.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise BackendUnreachableError(f"Europe PMC unreachable: {e}") from e
        if r.status_code >= 500:
            raise BackendUnreachableError(f"Europe PMC returned {r.status_code}")
        if r.status_code == 400:
            raise InvalidQueryError(f"Europe PMC rejected query: {r.text[:200]}")
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        *,
        year_range: tuple[int, int] | None = None,
        tissue: str | None = None,
        page_size: int = 10,
    ) -> list[RetrievalHit]:
        q = query
        if tissue:
            q = f"({q}) AND {tissue}"
        if year_range:
            q = f"({q}) AND (FIRST_PDATE:[{year_range[0]}-01-01 TO " f"{year_range[1]}-12-31])"
        data = self._get(
            "search",
            {
                "query": q,
                "format": "json",
                "pageSize": page_size,
                "resultType": "core",
            },
        )
        return [_record_to_hit(r) for r in data.get("resultList", {}).get("result", [])]

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit:
        ident = pmid_or_doi.replace("pmid:", "").replace("doi:", "")
        data = self._get(
            "search",
            {
                "query": f"EXT_ID:{ident}",
                "format": "json",
                "pageSize": 1,
                "resultType": "core",
            },
        )
        results = data.get("resultList", {}).get("result", [])
        if not results:
            raise InvalidQueryError(f"No record for {pmid_or_doi}")
        return _record_to_hit(results[0])
