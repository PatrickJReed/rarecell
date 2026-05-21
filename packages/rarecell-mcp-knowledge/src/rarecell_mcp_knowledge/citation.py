"""Citation + RetrievalHit pydantic models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal[
    "europepmc", "pubmed", "cellmarker", "panglaodb", "msigdb", "enrichr", "manual", "preset"
]


class Citation(BaseModel):
    source_id: str
    source: Source
    title: str | None = None
    url: str | None = None


class RetrievalHit(BaseModel):
    citation: Citation
    title: str
    snippet: str
    payload: dict
    source: Source
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
