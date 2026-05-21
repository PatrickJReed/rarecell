"""Recommender protocol + Recommendation dataclass."""

from __future__ import annotations

from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel

Decision = Literal["keep", "drop", "purify"]


class Recommendation(BaseModel):
    cluster_id: str
    recommendation: Decision
    confidence: float
    evidence_summary: dict
    reasoning: str
    citations: list[str] = []


class Recommender(Protocol):
    """Anything that turns a consensus-table DataFrame into per-cluster Recommendations."""

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]: ...
