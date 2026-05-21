"""NL → draft TargetCellProfile, grounded by literature + markers RAG."""

from __future__ import annotations

import json
import re
from typing import Any

from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile
from rarecell.rag.knowledge import KnowledgeSession
from rarecell.rag.literature import LiteratureRetriever
from rarecell.rag.markers_db import MarkersDBRetriever

_log = get_logger("rarecell.agent.draft")


def _extract_json_block(text: str) -> dict | None:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _build_drafting_prompt(
    user_prompt: str,
    literature_hits: list,
    marker_hits: list,
    anchor_hit: Any | None = None,
) -> str:
    lit_lines = [
        f"- {h.citation.source_id}: {h.title} — {h.snippet[:150]}" for h in literature_hits[:5]
    ]
    marker_lines = [
        f"- {h.title}: {', '.join(h.payload.get('genes', [])[:10])}" for h in marker_hits[:5]
    ]
    anchor_section = ""
    if anchor_hit is not None:
        anchor_section = (
            "Anchor paper (primary grounding):\n"
            f"- {anchor_hit.citation.source_id}: {anchor_hit.title}\n"
            f"  {anchor_hit.snippet[:600]}\n\n"
        )
    return (
        anchor_section + f"User prompt:\n{user_prompt}\n\n"
        "Literature hits:\n" + "\n".join(lit_lines) + "\n\n"
        "Marker DB hits:\n" + "\n".join(marker_lines) + "\n\n"
        "Draft a TargetCellProfile as a single ```json``` block matching this shape:\n"
        "```json\n{\n"
        '  "profile_id": "<kebab-case-slug>",\n'
        '  "name": "<short name>",\n'
        '  "description": "<one paragraph>",\n'
        '  "target_lineage": "lymphoid" | "myeloid" | "neural" | "epithelial",\n'
        '  "tissue": ["<tissue1>", ...],\n'
        '  "expected_abundance": {"min_fraction": <float>, "max_fraction": <float>},\n'
        '  "positive_markers": {"<panel_name>": {"genes": [...], "threshold_z": 1.0, "citations": [...]}},\n'
        '  "negative_markers": {},\n'
        '  "qc": {"min_genes_per_cell": <int>, "max_pct_mt": <float>}\n'
        "}\n```\n"
        "The draft must NOT set frozen or human_reviewed; the user reviews + freezes separately."
    )


def draft_profile_from_prompt(
    *,
    prompt: str,
    client: Any,
    session: KnowledgeSession,
    anchor_paper: str | None = None,
) -> TargetCellProfile:
    """Draft a TargetCellProfile from a natural-language prompt.

    Retrieves literature + marker hits, asks Claude to compose a profile,
    and returns the parsed (un-frozen) TargetCellProfile.

    Args:
        prompt: Natural-language description of the target cell type.
        client: Claude client exposing ``messages_create``.
        session: Knowledge session backing literature + marker retrieval.
        anchor_paper: Optional PMID or DOI to anchor the draft against. When
            provided, the function fetches that paper's abstract via
            ``LiteratureRetriever.fetch_abstract`` and prepends it to the
            drafting prompt as the primary grounding source. Generic
            literature + marker searches still run and supplement the
            anchor. If the anchor fetch fails (e.g., unknown PMID), the
            failure is logged and drafting proceeds without the anchor.

    Raises ValueError if the model's response doesn't parse.
    """
    lit_retriever = LiteratureRetriever(session=session)
    marker_retriever = MarkersDBRetriever(session=session)

    anchor_hit: Any | None = None
    if anchor_paper is not None:
        try:
            anchor_hit = lit_retriever.fetch_abstract(anchor_paper)
        except Exception as e:
            _log.warning(
                "draft.anchor_fetch_failed",
                anchor_paper=anchor_paper,
                error=str(e),
            )
            anchor_hit = None

    try:
        literature_hits = lit_retriever.search(prompt, page_size=5)
    except Exception as e:
        _log.warning("draft.literature_search_failed", error=str(e))
        literature_hits = []
    try:
        marker_hits = marker_retriever.search(prompt)
    except Exception as e:
        _log.warning("draft.marker_search_failed", error=str(e))
        marker_hits = []

    user_msg = _build_drafting_prompt(
        prompt,
        literature_hits,
        marker_hits,
        anchor_hit=anchor_hit,
    )
    resp = client.messages_create(messages=[{"role": "user", "content": user_msg}])
    text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise ValueError("Drafting response had no text blocks.")
    parsed = _extract_json_block(text_blocks[0]["text"])
    if parsed is None:
        raise ValueError("Drafting response did not contain a parseable JSON block.")

    return TargetCellProfile.model_validate(parsed)
