from pathlib import Path

from rarecell.rag.knowledge import build_knowledge_session
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

# This test file is at packages/rarecell/tests/rag/test_knowledge.py
# parents[0]=rag, parents[1]=tests, parents[2]=rarecell, parents[3]=packages,
# parents[4]=worktree-root. So PLAN2_FIXTURES needs parents[4].
PLAN2_FIXTURES = Path(__file__).resolve().parents[4] / "packages/rarecell-mcp-knowledge/tests/data"


def test_session_exposes_tool_map(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    names = sorted(session.tool_names)
    assert "search_literature" in names
    assert "search_markers" in names
    assert "get_canonical_panel" in names


def test_session_search_markers_returns_hits(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    hits = session.call("search_markers", {"cell_type": "T cell", "tissue": "blood"})
    all_genes = {g for h in hits for g in h["payload"]["genes"]}
    assert "CD3D" in all_genes
