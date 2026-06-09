"""Agent target resolution: choose node-vs-program from a retrieval shortlist."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rarecell.cns.retrieve import NodeMatch, build_catalog, score_nodes
from rarecell.logging import get_logger
from rarecell.profile.schema import CNSTaxonomyConfig

_log = get_logger("rarecell.agent.resolve")


class TargetResolution(BaseModel):
    mode: Literal["node", "program"]
    gate_node: str
    gate_level: Literal["supercluster", "cluster"]
    characterize_level: Literal["cluster", "subcluster"]
    rationale: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def _extract_json_block(text: str) -> dict[str, Any] | None:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _build_resolution_prompt(profile: Any, candidates: list[NodeMatch]) -> str:
    lines = [
        "You map a target cell population to the BICCN whole-human-brain taxonomy.",
        "Decide whether the target is a DISCRETE reference node (gate on it: mode='node')",
        "or a gene PROGRAM / state within a broader container (gate on the container and",
        "isolate by markers: mode='program').",
        "",
        f"Target name: {getattr(profile, 'name', '')}",
        f"Lineage: {getattr(profile, 'target_lineage', '')}",
        f"Description: {getattr(profile, 'description', '')}",
        f"Target markers: {_profile_markers(profile)[:30]}",
        "",
        "Candidate reference nodes (name | level | class | top markers | signals):",
    ]
    for m in candidates:
        lines.append(
            f"- {m.node.name} | {m.node.level} | {m.node.cell_class} | "
            f"{m.node.markers[:8]} | {m.signals}"
        )
    lines += [
        "",
        "Reply with ONLY a json fenced block:",
        "```json",
        '{"mode":"node|program","gate_node":"<one candidate name or its parent>",',
        '"gate_level":"supercluster|cluster","characterize_level":"cluster|subcluster",',
        '"rationale":"...","citations":["pmid:..."],"confidence":0.0}',
        "```",
    ]
    return "\n".join(lines)


def resolve_target(profile: Any, *, candidates: list[NodeMatch], client: Any) -> TargetResolution:
    """Ask the agent to resolve the target from the candidate shortlist.

    (v1 grounds only on the retrieval candidates + profile fields; literature/RAG grounding is a future enhancement.)
    """
    msg = _build_resolution_prompt(profile, candidates)
    resp = client.messages_create(messages=[{"role": "user", "content": msg}])
    text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise ValueError("Resolution response had no text blocks.")
    parsed = _extract_json_block(text_blocks[0]["text"])
    if parsed is None:
        raise ValueError("Resolution response did not contain a parseable JSON block.")
    return TargetResolution.model_validate(parsed)


def _profile_markers(profile: Any) -> list[str]:
    genes: list[str] = []
    for panel in (getattr(profile, "positive_markers", {}) or {}).values():
        genes.extend(getattr(panel, "genes", []) or [])
    return genes


def resolve_cns_target(
    profile: Any,
    *,
    bundle_dir: Path,
    reference_release: str,
    client: Any,
    top_k_per_level: int = 8,
    min_resolution_confidence: float = 0.5,
) -> CNSTaxonomyConfig:
    """Run retrieval + agent resolution, returning a populated CNSTaxonomyConfig."""
    catalog = build_catalog(bundle_dir)
    tissue = " ".join(getattr(profile, "tissue", []) or []) or None
    candidates = score_nodes(
        catalog,
        markers=_profile_markers(profile),
        lineage=getattr(profile, "target_lineage", None),
        tissue=tissue,
        top_k_per_level=top_k_per_level,
    )
    res = resolve_target(profile, candidates=candidates, client=client)
    if res.confidence < min_resolution_confidence:
        _log.info(
            "resolve.low_confidence_gate_disabled",
            confidence=res.confidence,
            node=res.gate_node,
        )
        return CNSTaxonomyConfig(enabled=False, rationale=res.rationale, citations=res.citations)
    return CNSTaxonomyConfig(
        enabled=True,
        target_node=res.gate_node,
        target_level=res.gate_level,
        mode=res.mode,
        characterize_level=res.characterize_level,
        reference_release=reference_release,
        rationale=res.rationale,
        citations=res.citations,
    )
