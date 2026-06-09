"""MarkersDBRetriever — adapts KnowledgeSession to a Retriever-shaped API."""

from __future__ import annotations

from rarecell_mcp_knowledge.citation import RetrievalHit

from rarecell.rag.knowledge import KnowledgeSession


class MarkersDBRetriever:
    """Wraps KnowledgeSession's `search_markers` and `get_canonical_panel` tools."""

    def __init__(self, session: KnowledgeSession):
        self.session = session

    def search(
        self,
        cell_type: str,
        *,
        tissue: str | None = None,
    ) -> list[RetrievalHit]:
        kwargs: dict = {"cell_type": cell_type}
        if tissue is not None:
            kwargs["tissue"] = tissue
        raw_hits = self.session.call("search_markers", kwargs)
        return [RetrievalHit.model_validate(h) for h in raw_hits]

    def canonical_panel(self, name: str) -> RetrievalHit:
        raw = self.session.call("get_canonical_panel", {"name": name})
        return RetrievalHit.model_validate(raw)
