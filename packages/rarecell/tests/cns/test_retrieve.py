from pathlib import Path

from rarecell.cns.retrieve import NodeDescriptor, NodeMatch, build_catalog, score_nodes


def _catalog() -> list[NodeDescriptor]:
    return [
        NodeDescriptor("Astrocyte", "supercluster", None, "Astrocyte", ["AQP4", "GFAP", "SLC1A2"]),
        NodeDescriptor(
            "MGE interneuron", "supercluster", None, "MGE interneuron", ["LHX6", "GAD1"]
        ),
        NodeDescriptor(
            "Astro_52",
            "cluster",
            "Astrocyte",
            "ASTRO",
            ["AQP4", "GFAP"],
            regions=["Cerebral cortex"],
        ),
        NodeDescriptor("MGE_259", "cluster", "MGE interneuron", "NEUR", ["LHX6", "PVALB"]),
    ]


def test_score_ranks_marker_match_first() -> None:
    matches = score_nodes(_catalog(), markers=["AQP4", "GFAP"], lineage="astrocyte")
    top_sc = next(m for m in matches if m.node.level == "supercluster")
    assert top_sc.node.name == "Astrocyte"
    assert top_sc.signals["marker_overlap"] > 0
    assert top_sc.signals["class_match"] == 1.0
    assert isinstance(top_sc, NodeMatch)


def test_score_region_signal_and_top_k() -> None:
    matches = score_nodes(
        _catalog(),
        markers=["AQP4"],
        lineage="astrocyte",
        tissue="cerebral cortex",
        top_k_per_level=1,
    )
    # only the best per level returned
    assert sum(1 for m in matches if m.node.level == "cluster") == 1
    astro_cl = next(m for m in matches if m.node.level == "cluster")
    assert astro_cl.node.name == "Astro_52"
    assert astro_cl.signals["region_match"] == 1.0


def test_build_catalog_has_superclusters_and_clusters(tiny_bundle: Path) -> None:
    catalog = build_catalog(tiny_bundle)
    levels = {n.level for n in catalog}
    assert levels == {"supercluster", "cluster"}
    sc = [n for n in catalog if n.level == "supercluster"]
    assert any(n.name == "Astrocyte" for n in sc)
    # superclusters carry a marker panel from the bundle
    astro = next(n for n in sc if n.name == "Astrocyte")
    assert isinstance(astro, NodeDescriptor) and isinstance(astro.markers, list)
    # clusters carry a parent supercluster
    clusters = [n for n in catalog if n.level == "cluster"]
    assert all(n.parent for n in clusters)
