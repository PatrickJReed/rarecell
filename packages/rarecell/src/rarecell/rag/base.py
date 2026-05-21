"""Retriever protocol for the rag layer."""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    """Anything that returns RetrievalHits given a query.

    Concrete retrievers in rarecell.rag adapt rarecell-mcp-knowledge's
    KnowledgeApp surface to this interface.
    """

    def search(self, query: str, **kwargs) -> list: ...
