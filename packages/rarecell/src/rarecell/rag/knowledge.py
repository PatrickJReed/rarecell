"""In-process wrapper around rarecell-mcp-knowledge's KnowledgeApp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.server import build_app


@dataclass
class KnowledgeSession:
    """Holds a single KnowledgeApp + tool_map. Reuse across retrievers."""

    _app: Any
    catalog_path: Path
    cache_path: Path

    @property
    def tool_names(self) -> list[str]:
        return self._app.list_tool_names()

    def call(self, name: str, kwargs: dict) -> Any:
        return self._app.call_tool(name, kwargs)


def build_knowledge_session(
    *,
    catalog_path: Path,
    cache_path: Path,
) -> KnowledgeSession:
    """Build a KnowledgeSession wrapping a KnowledgeApp."""
    catalog = MarkersCatalog(catalog_path)
    app = build_app(catalog=catalog, cache_path=cache_path)
    return KnowledgeSession(_app=app, catalog_path=catalog_path, cache_path=cache_path)
