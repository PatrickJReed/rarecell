from pathlib import Path
from unittest.mock import MagicMock

from rarecell.agent.draft import draft_profile_from_prompt
from rarecell.rag.knowledge import build_knowledge_session
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

PLAN2_FIXTURES = Path(__file__).resolve().parents[4] / "packages/rarecell-mcp-knowledge/tests/data"


CANNED_PROFILE_JSON = """```json
{
  "profile_id": "draft-tcell-pbmc",
  "name": "T cells (drafted)",
  "description": "Drafted from NL prompt",
  "target_lineage": "lymphoid",
  "tissue": ["pbmc"],
  "expected_abundance": {"min_fraction": 0.1, "max_fraction": 0.6},
  "positive_markers": {
    "pan_t": {
      "genes": ["CD3D", "CD3E"],
      "threshold_z": 1.0,
      "citations": [{"source_id": "panglaodb:T_cell", "source": "panglaodb"}]
    }
  },
  "negative_markers": {},
  "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10}
}
```"""


def test_draft_profile_from_prompt_returns_target_cell_profile(tmp_path):
    catalog = MarkersCatalog(tmp_path / "m.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite",
        cache_path=tmp_path / "c.sqlite",
    )

    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": CANNED_PROFILE_JSON}],
    }

    profile = draft_profile_from_prompt(
        prompt="rare T cells in PBMC",
        client=mock_client,
        session=session,
    )
    assert profile.profile_id == "draft-tcell-pbmc"
    assert "CD3D" in profile.positive_markers["pan_t"].genes
    assert profile.frozen is False
    assert profile.human_reviewed is False
