"""Deterministic retrieval over the CNS reference taxonomy.

Builds a searchable catalog of reference nodes (superclusters + clusters) from
the bundle, and scores them against a query's markers / lineage / tissue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rarecell.cns.format import load_annotations, load_markers, load_taxonomy

# Map a profile's free-text lineage to a Siletti class code.
_LINEAGE_TO_CLASS = {
    "astrocyte": "ASTRO",
    "microglia": "MGL",
    "oligodendrocyte": "OLIGO",
    "oligodendrocyte precursor": "OPC",
    "opc": "OPC",
    "neuron": "NEUR",
    "endothelial": "ENDO",
    "fibroblast": "FIB",
    "pericyte": "PER",
    "ependymal": "EPEN",
}


@dataclass
class NodeDescriptor:
    name: str
    level: str  # "supercluster" | "cluster"
    parent: str | None
    cell_class: str
    markers: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    neurotransmitter: str = ""


@dataclass
class NodeMatch:
    node: NodeDescriptor
    score: float
    signals: dict[str, float]


def _overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = {x.upper() for x in a}, {x.upper() for x in b}
    return len(sa & sb) / min(len(sa), len(sb))


def _class_match(lineage: str | None, cell_class: str) -> float:
    if not lineage or not cell_class:
        return 0.0
    le, ce = lineage.lower().strip(), cell_class.lower().strip()
    if le == ce or le in ce or ce in le:
        return 1.0
    code = _LINEAGE_TO_CLASS.get(le)
    return 1.0 if code and code.lower() == ce else 0.0


def _region_match(tissue: str | None, regions: list[str]) -> float:
    if not tissue or not regions:
        return 0.0
    toks = set(tissue.lower().split())
    if not toks:
        return 0.0
    rtoks = set(" ".join(regions).lower().split())
    return len(toks & rtoks) / len(toks)


def score_nodes(
    catalog: list[NodeDescriptor],
    *,
    markers: list[str],
    lineage: str | None = None,
    tissue: str | None = None,
    top_k_per_level: int = 8,
) -> list[NodeMatch]:
    matches: list[NodeMatch] = []
    for n in catalog:
        s_marker = _overlap(markers, n.markers)
        s_class = _class_match(lineage, n.cell_class)
        s_region = _region_match(tissue, n.regions)
        score = 0.5 * s_marker + 0.3 * s_class + 0.2 * s_region
        matches.append(
            NodeMatch(
                node=n,
                score=score,
                signals={
                    "marker_overlap": s_marker,
                    "class_match": s_class,
                    "region_match": s_region,
                },
            )
        )
    out: list[NodeMatch] = []
    for level in ("supercluster", "cluster"):
        ranked = sorted((m for m in matches if m.node.level == level), key=lambda m: -m.score)
        out.extend(ranked[:top_k_per_level])
    return out


def build_catalog(bundle_dir: Path) -> list[NodeDescriptor]:
    tree = load_taxonomy(bundle_dir)  # supercluster -> [cluster names]
    ann = load_annotations(bundle_dir)  # cluster name -> {class, markers, regions, ...}
    sc_panels = load_markers(bundle_dir, "nodes/supercluster/_markers.json")

    catalog: list[NodeDescriptor] = []
    for sc, clusters in tree.items():
        catalog.append(
            NodeDescriptor(
                name=sc,
                level="supercluster",
                parent=None,
                cell_class=sc,
                markers=list(sc_panels.get(sc, [])),
            )
        )
        for cl in clusters:
            a = ann.get(cl, {})
            raw_markers = a.get("markers", [])
            raw_regions = a.get("regions", [])
            catalog.append(
                NodeDescriptor(
                    name=cl,
                    level="cluster",
                    parent=sc,
                    cell_class=str(a.get("class", "")),
                    markers=[str(g) for g in raw_markers] if isinstance(raw_markers, list) else [],
                    regions=[str(x) for x in raw_regions] if isinstance(raw_regions, list) else [],
                    neurotransmitter=str(a.get("neurotransmitter", "")),
                )
            )
    return catalog
