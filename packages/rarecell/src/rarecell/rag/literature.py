"""LiteratureRetriever — adapts KnowledgeSession to a Retriever-shaped API."""

from __future__ import annotations

from rarecell_mcp_knowledge.citation import RetrievalHit

from rarecell.rag.knowledge import KnowledgeSession


class LiteratureRetriever:
    """Wraps KnowledgeSession's `search_literature` and `fetch_abstract` tools."""

    def __init__(self, session: KnowledgeSession):
        self.session = session

    def search(
        self,
        query: str,
        *,
        year_range: tuple[int, int] | None = None,
        tissue: str | None = None,
        page_size: int = 10,
    ) -> list[RetrievalHit]:
        kwargs: dict = {"query": query, "page_size": page_size}
        if year_range is not None:
            kwargs["year_range"] = list(year_range)
        if tissue is not None:
            kwargs["tissue"] = tissue
        raw_hits = self.session.call("search_literature", kwargs)
        return [RetrievalHit.model_validate(h) for h in raw_hits]

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit:
        raw = self.session.call("fetch_abstract", {"pmid_or_doi": pmid_or_doi})
        return RetrievalHit.model_validate(raw)
