from pathlib import Path

from rarecell.rag.knowledge import build_knowledge_session
from rarecell.rag.markers_db import MarkersDBRetriever
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

PLAN2_FIXTURES = Path(__file__).resolve().parents[4] / "packages/rarecell-mcp-knowledge/tests/data"


def test_markers_retriever_search(tmp_path):
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
    retriever = MarkersDBRetriever(session=session)
    hits = retriever.search("T cell", tissue="blood")
    all_genes = {g for h in hits for g in h.payload["genes"]}
    assert "CD3D" in all_genes


def test_markers_retriever_canonical_panel(tmp_path):
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
    retriever = MarkersDBRetriever(session=session)
    hit = retriever.canonical_panel("T cell")
    assert "CD3D" in hit.payload["genes"]
