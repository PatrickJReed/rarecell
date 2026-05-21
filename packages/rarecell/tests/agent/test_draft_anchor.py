"""Tests for the anchor_paper kwarg of draft_profile_from_prompt."""

from pathlib import Path
from unittest.mock import MagicMock

from rarecell.agent.draft import draft_profile_from_prompt
from rarecell.rag.knowledge import build_knowledge_session
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

PLAN2_FIXTURES = Path(__file__).resolve().parents[4] / "packages/rarecell-mcp-knowledge/tests/data"


CANNED_PROFILE_JSON = """```json
{
  "profile_id": "anchor-astrocyte-snap",
  "name": "SNAP-divested astrocytes",
  "description": "Drafted from Ling 2024",
  "target_lineage": "neural",
  "tissue": ["brain"],
  "expected_abundance": {"min_fraction": 0.05, "max_fraction": 0.3},
  "positive_markers": {
    "pan_astrocyte": {
      "genes": ["GFAP", "AQP4", "ALDH1L1"],
      "threshold_z": 1.0,
      "citations": [{"source_id": "pmid:38448582", "source": "europepmc"}]
    }
  },
  "negative_markers": {},
  "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10}
}
```"""


def _seeded_session(tmp_path):
    catalog = MarkersCatalog(tmp_path / "m.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    return build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite",
        cache_path=tmp_path / "c.sqlite",
    )


def test_anchor_paper_fetches_abstract_first(tmp_path, monkeypatch):
    """When anchor_paper is provided, the function calls fetch_abstract first."""
    from rarecell.rag import literature as lit_mod

    fetch_abstract_calls: list[str] = []

    def fake_fetch_abstract(self, pmid_or_doi):
        fetch_abstract_calls.append(pmid_or_doi)
        from rarecell.rag import Citation, RetrievalHit

        return RetrievalHit(
            citation=Citation(
                source_id="pmid:38448582",
                source="europepmc",
                title="A concerted neuron-astrocyte program declines in ageing and schizophrenia",
            ),
            title="A concerted neuron-astrocyte program declines in ageing and schizophrenia",
            snippet="SNAP is a coordinated gene program in neurons and astrocytes; "
            "astrocytes divested from SNAP are enriched in schizophrenia and aging.",
            payload={"doi": "10.1038/s41586-024-07109-5", "year": "2024"},
            source="europepmc",
        )

    monkeypatch.setattr(lit_mod.LiteratureRetriever, "fetch_abstract", fake_fetch_abstract)

    session = _seeded_session(tmp_path)

    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": CANNED_PROFILE_JSON}],
    }

    profile = draft_profile_from_prompt(
        prompt="SNAP-divested astrocytes in schizophrenia DLPFC",
        client=mock_client,
        session=session,
        anchor_paper="38448582",
    )

    assert fetch_abstract_calls == [
        "38448582"
    ], f"Expected fetch_abstract to be called once with anchor_paper id; got {fetch_abstract_calls}"
    # Anchor paper PMID must appear somewhere in the prompt Claude saw
    sent_messages = mock_client.messages_create.call_args.kwargs["messages"]
    assert any(
        "38448582" in m["content"] for m in sent_messages
    ), "anchor paper PMID must appear in Claude's prompt"
    assert profile.profile_id == "anchor-astrocyte-snap"


def test_anchor_paper_optional(tmp_path):
    """Without anchor_paper, draft_profile_from_prompt works as before."""
    session = _seeded_session(tmp_path)
    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": CANNED_PROFILE_JSON}],
    }

    profile = draft_profile_from_prompt(
        prompt="astrocytes in brain",
        client=mock_client,
        session=session,
    )
    # Should still succeed and produce a profile
    assert profile.profile_id == "anchor-astrocyte-snap"


def test_anchor_paper_failure_degrades_gracefully(tmp_path, monkeypatch):
    """If fetch_abstract fails, drafting should still proceed without the anchor."""
    from rarecell.rag import literature as lit_mod

    def fake_fetch_abstract(self, pmid_or_doi):
        raise RuntimeError("Europe PMC unreachable")

    monkeypatch.setattr(lit_mod.LiteratureRetriever, "fetch_abstract", fake_fetch_abstract)

    session = _seeded_session(tmp_path)
    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": CANNED_PROFILE_JSON}],
    }

    profile = draft_profile_from_prompt(
        prompt="astrocytes",
        client=mock_client,
        session=session,
        anchor_paper="bogus-id",
    )
    assert profile.profile_id == "anchor-astrocyte-snap"
