"""Marker backend protocol."""

from __future__ import annotations

from typing import Protocol

from rarecell_mcp_knowledge.citation import RetrievalHit


class MarkerBackend(Protocol):
    def search_markers(
        self,
        cell_type: str,
        tissue: str | None = None,
    ) -> list[RetrievalHit]: ...

    def get_canonical_panel(self, name: str) -> RetrievalHit: ...
