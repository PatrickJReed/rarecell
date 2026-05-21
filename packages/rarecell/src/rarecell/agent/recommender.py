"""ClaudeRecommender — LLM-backed implementation of the Recommender protocol."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommendation, Recommender

_log = get_logger("rarecell.agent.recommender")


def _extract_json_block(text: str) -> dict | None:
    """Find the first ```json ... ``` block in text and parse it.

    Falls back to parsing the entire text as JSON.
    """
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class ClaudeRecommender(Recommender):
    """Asks Claude to recommend keep/drop/purify per cluster.

    Output format: a single JSON object with key "recommendations" mapping
    to a list of {cluster_id, recommendation, confidence, evidence_summary,
    reasoning, citations} objects.
    """

    def __init__(self, *, profile: TargetCellProfile, client: Any):
        self.profile = profile
        self.client = client

    def _build_user_message(self, table: pd.DataFrame) -> str:
        positive_panels = list(self.profile.positive_markers.keys())
        negative_panels = list(self.profile.negative_markers.keys())
        return (
            "Profile target: " + self.profile.name + " in " + ", ".join(self.profile.tissue) + ".\n"
            "Positive panels: " + ", ".join(positive_panels) + "\n"
            "Negative panels: " + ", ".join(negative_panels) + "\n\n"
            "Consensus table (one row per cluster):\n" + table.to_string(index=False) + "\n\n"
            "Return a single ```json``` block with this exact shape:\n"
            '```json\n{\n  "recommendations": [\n    {\n'
            '      "cluster_id": "<id>",\n'
            '      "recommendation": "keep" | "drop" | "purify",\n'
            '      "confidence": <float 0-1>,\n'
            '      "evidence_summary": {...},\n'
            '      "reasoning": "<short string>",\n'
            '      "citations": ["<id>", ...]\n'
            "    }\n  ]\n}\n```"
        )

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]:
        user = self._build_user_message(table)
        resp = self.client.messages_create(messages=[{"role": "user", "content": user}])
        text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
        if not text_blocks:
            _log.warning("claude_recommender.no_text_blocks")
            return []
        parsed = _extract_json_block(text_blocks[0]["text"])
        if parsed is None or "recommendations" not in parsed:
            _log.warning("claude_recommender.parse_failure")
            return []
        out: list[Recommendation] = []
        for entry in parsed["recommendations"]:
            try:
                out.append(Recommendation(**entry))
            except Exception as e:
                _log.warning(
                    "claude_recommender.validation_failure",
                    error=str(e),
                    entry=entry,
                )
        return out
