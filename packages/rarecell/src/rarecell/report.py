"""IsolationReport — manifest + decisions.jsonl + figures + bibliography.

This file grows in Task 21 to include the full Report writer.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Decision(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    gate: Literal[1, 2, 3]
    cluster_id: str
    recommendation: Literal["keep", "drop", "purify", "accept", "abort"]
    user_decision: Literal["keep", "drop", "purify", "accept", "abort"]
    confidence: float
    evidence: dict
    reasoning: str
    citations: list[str] = []


class DecisionLog:
    """Append-only JSONL log of gate decisions."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision) -> None:
        with self.path.open("a") as f:
            f.write(decision.model_dump_json() + "\n")

    @staticmethod
    def iter_decisions(path: Path) -> Iterator[Decision]:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                yield Decision.model_validate_json(line)
