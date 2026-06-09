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


_VALID_CITATION_SOURCES = {
    "europepmc",
    "pubmed",
    "cellmarker",
    "panglaodb",
    "msigdb",
    "enrichr",
    "manual",
    "preset",
}


def _extract_json_block(text: str) -> dict | None:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_citation(value: Any) -> dict:
    """Normalize a model-emitted citation to the Citation dict shape.

    LLMs frequently emit citations as plain strings ("pmid:38448582") or
    with a non-canonical source value. We tolerate that by promoting
    strings to dicts and falling back to a "manual" source when the
    declared source isn't in the Literal whitelist.
    """
    if isinstance(value, str):
        source_id = value
        prefix = source_id.split(":", 1)[0].lower() if ":" in source_id else ""
        source = "europepmc" if prefix in {"pmid", "doi"} else "manual"
        return {"source_id": source_id, "source": source}
    if isinstance(value, dict):
        out = dict(value)
        src = out.get("source")
        if not isinstance(src, str) or src not in _VALID_CITATION_SOURCES:
            out["source"] = "manual"
        if "source_id" not in out:
            # Some models emit {"id": "...", ...} or {"pmid": "..."}
            for k in ("id", "pmid", "doi", "ref"):
                if k in out:
                    out["source_id"] = str(out[k])
                    break
        return out
    # Unknown shape — bury it in a manual citation so we don't crash
    return {"source_id": str(value), "source": "manual"}


def _coerce_panel_citations(panels: Any) -> None:
    """Walk a {panel_name: {... "citations": [...]}} dict in place."""
    if not isinstance(panels, dict):
        return
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        cites = panel.get("citations")
        if isinstance(cites, list):
            panel["citations"] = [_coerce_citation(c) for c in cites]


def _coerce_parsed_profile(parsed: dict) -> dict:
    """Normalize an LLM-emitted profile dict in place before validation.

    Today: coerces string citations to Citation dicts on every panel.
    Future: this is the place to add other lenient pre-validation tweaks.
    """
    _coerce_panel_citations(parsed.get("positive_markers"))
    _coerce_panel_citations(parsed.get("negative_markers"))
    return parsed


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
        '  "positive_markers": {\n'
        '    "<panel_name>": {\n'
        '      "genes": ["GENE1", "GENE2", ...],\n'
        '      "threshold_z": 1.0,\n'
        '      "citations": [{"source_id": "pmid:12345", "source": "europepmc"}]\n'
        "    }\n"
        "  },\n"
        '  "negative_markers": {},\n'
        '  "qc": {"min_genes_per_cell": <int>, "max_pct_mt": <float>}\n'
        "}\n```\n"
        'Each citation MUST be a dict with `source_id` (e.g. "pmid:38448582") '
        'and `source` (one of: "europepmc", "pubmed", "cellmarker", '
        '"panglaodb", "msigdb", "enrichr", "manual", "preset"). Do not emit '
        "citations as plain strings.\n"
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

    parsed = _coerce_parsed_profile(parsed)
    return TargetCellProfile.model_validate(parsed)
