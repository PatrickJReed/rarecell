"""In-process smoke test of the FastMCP server."""

from pathlib import Path

import pytest
import respx
from httpx import Response
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv
from rarecell_mcp_knowledge.server import build_app

FIXTURES = Path(__file__).parent / "data"


@pytest.fixture
def app(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=FIXTURES / "panglaodb_tiny.tsv",
    )
    return build_app(catalog=catalog, cache_path=tmp_path / "cache.sqlite")


def test_app_has_required_tools(app):
    names = sorted(app.list_tool_names())
    assert "search_literature" in names
    assert "fetch_abstract" in names
    assert "search_markers" in names
    assert "get_canonical_panel" in names
    assert "enrichr_enrich" in names


@respx.mock
def test_search_literature_via_app(app):
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(
            200,
            json={
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "id": "111",
                            "pmid": "111",
                            "title": "T cell",
                            "abstractText": "abstract",
                            "doi": "10.1/x",
                            "pubYear": "2024",
                            "authorString": "X",
                        }
                    ]
                },
            },
        )
    )
    hits = app.call_tool("search_literature", {"query": "T cell"})
    assert len(hits) == 1
    assert hits[0]["citation"]["source_id"] == "pmid:111"


def test_search_markers_via_app(app):
    hits = app.call_tool("search_markers", {"cell_type": "T cell", "tissue": "blood"})
    all_genes = {g for h in hits for g in h["payload"]["genes"]}
    assert "CD3D" in all_genes
