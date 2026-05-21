"""Literature backend protocol."""

from __future__ import annotations

from typing import Protocol

from rarecell_mcp_knowledge.citation import RetrievalHit


class LiteratureBackend(Protocol):
    def search(
        self,
        query: str,
        *,
        year_range: tuple[int, int] | None = None,
        tissue: str | None = None,
        page_size: int = 10,
    ) -> list[RetrievalHit]: ...

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit: ...
