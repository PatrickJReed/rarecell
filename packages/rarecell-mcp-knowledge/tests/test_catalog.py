from pathlib import Path

import pytest
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

FIXTURES = Path(__file__).parent / "data"


@pytest.fixture
def seeded_catalog(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=FIXTURES / "panglaodb_tiny.tsv",
    )
    return catalog


def test_search_t_cell_blood(seeded_catalog):
    hits = seeded_catalog.search_markers("T cell", tissue="blood")
    sources = {h.source for h in hits}
    assert "cellmarker" in sources or "panglaodb" in sources
    all_markers = {g for h in hits for g in h.payload["genes"]}
    assert "CD3D" in all_markers
    assert "CD3E" in all_markers


def test_search_microglia_brain(seeded_catalog):
    hits = seeded_catalog.search_markers("Microglia", tissue="brain")
    all_markers = {g for h in hits for g in h.payload["genes"]}
    assert "CX3CR1" in all_markers
    assert "P2RY12" in all_markers


def test_search_unknown_returns_empty(seeded_catalog):
    hits = seeded_catalog.search_markers("Cardiomyocyte", tissue="heart")
    assert hits == []


def test_get_canonical_panel_t_cell(seeded_catalog):
    panel = seeded_catalog.get_canonical_panel("T cell")
    assert panel.payload["genes"]
    assert "CD3D" in panel.payload["genes"]
