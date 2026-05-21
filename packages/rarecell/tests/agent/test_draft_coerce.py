"""Tests for citation coercion inside draft_profile_from_prompt.

LLMs occasionally emit citations as plain strings ("pmid:38448582")
instead of full Citation dicts, or use a non-canonical `source` value.
We coerce those into the canonical shape before pydantic validation so
real drafted profiles don't fail at the boundary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rarecell.agent.draft import (
    _coerce_citation,
    _coerce_parsed_profile,
    draft_profile_from_prompt,
)
from rarecell.rag.knowledge import build_knowledge_session

PLAN2_FIXTURES = Path(__file__).resolve().parents[4] / "packages/rarecell-mcp-knowledge/tests/data"


def test_coerce_string_citation_with_pmid_prefix():
    out = _coerce_citation("pmid:38448582")
    assert out == {"source_id": "pmid:38448582", "source": "europepmc"}


def test_coerce_string_citation_with_doi_prefix():
    out = _coerce_citation("doi:10.1234/abc")
    assert out == {"source_id": "doi:10.1234/abc", "source": "europepmc"}


def test_coerce_string_citation_unknown_prefix_falls_back_to_manual():
    out = _coerce_citation("Smith 2024")
    assert out == {"source_id": "Smith 2024", "source": "manual"}


def test_coerce_dict_citation_with_invalid_source_promoted_to_manual():
    out = _coerce_citation({"source_id": "pmid:1", "source": "Ling 2024"})
    assert out == {"source_id": "pmid:1", "source": "manual"}


def test_coerce_dict_citation_with_valid_source_passes_through():
    out = _coerce_citation({"source_id": "pmid:1", "source": "europepmc"})
    assert out["source_id"] == "pmid:1"
    assert out["source"] == "europepmc"


def test_coerce_dict_citation_extracts_source_id_from_alt_keys():
    out = _coerce_citation({"id": "pmid:1", "source": "europepmc"})
    assert out["source_id"] == "pmid:1"
    out = _coerce_citation({"pmid": "12345", "source": "europepmc"})
    assert out["source_id"] == "12345"


def test_coerce_parsed_profile_walks_positive_and_negative_panels():
    parsed = {
        "positive_markers": {
            "pan_t": {
                "genes": ["CD3D"],
                "threshold_z": 1.0,
                "citations": ["pmid:1", {"source_id": "pmid:2", "source": "europepmc"}],
            }
        },
        "negative_markers": {
            "b_cell": {
                "genes": ["MS4A1"],
                "exclusion_threshold_z": 1.5,
                "citations": ["pmid:3"],
            }
        },
    }
    out = _coerce_parsed_profile(parsed)
    pos = out["positive_markers"]["pan_t"]["citations"]
    assert pos[0] == {"source_id": "pmid:1", "source": "europepmc"}
    assert pos[1] == {"source_id": "pmid:2", "source": "europepmc"}
    neg = out["negative_markers"]["b_cell"]["citations"]
    assert neg[0] == {"source_id": "pmid:3", "source": "europepmc"}


def test_draft_profile_from_prompt_accepts_string_citations(tmp_path):
    """End-to-end: a drafting response with string citations validates cleanly."""
    from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
    from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

    catalog = MarkersCatalog(tmp_path / "m.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite", cache_path=tmp_path / "c.sqlite"
    )

    response_with_string_citations = """```json
{
  "profile_id": "claude-emitted-strings",
  "name": "Astrocytes",
  "description": "Drafted by Claude with string citations.",
  "target_lineage": "neural",
  "tissue": ["brain"],
  "expected_abundance": {"min_fraction": 0.05, "max_fraction": 0.2},
  "positive_markers": {
    "pan_astrocyte": {
      "genes": ["GFAP", "AQP4"],
      "threshold_z": 1.0,
      "citations": ["pmid:38448582"]
    }
  },
  "negative_markers": {
    "neuron": {
      "genes": ["RBFOX3"],
      "exclusion_threshold_z": 1.5,
      "citations": ["pmid:38448582"]
    }
  },
  "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10}
}
```"""

    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": response_with_string_citations}]
    }

    profile = draft_profile_from_prompt(prompt="astrocytes", client=mock_client, session=session)
    # Validation succeeded — citations are now Citation instances
    cite = profile.positive_markers["pan_astrocyte"].citations[0]
    assert cite.source_id == "pmid:38448582"
    assert cite.source == "europepmc"
