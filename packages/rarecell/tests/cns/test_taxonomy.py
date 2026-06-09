from __future__ import annotations

from pathlib import Path

import pytest
from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.errors import ReferenceBuildError


def test_path_to_supercluster_target(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astrocyte", "supercluster")
    assert len(path) == 1
    artifact, keep_class = path[0]
    assert artifact.level == "supercluster"
    assert keep_class == "Astrocyte"


def test_path_to_cluster_target(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astro-1", "cluster")
    assert [a.level for a, _ in path] == ["supercluster", "cluster"]
    assert [keep for _, keep in path] == ["Astrocyte", "Astro-1"]
    assert path[1][0].parent == "Astrocyte"


def test_path_to_unknown_target_raises(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    with pytest.raises(ReferenceBuildError):
        tax.path_to("NotACluster", "cluster")
