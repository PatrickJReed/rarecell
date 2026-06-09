"""Resolve the ordered decision path toward a target node from a bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rarecell.cns.format import (
    BundleManifest,
    DecisionArtifact,
    DecisionLevel,
    load_manifest,
    load_taxonomy,
)
from rarecell.errors import ReferenceBuildError


@dataclass
class TaxonomyTree:
    manifest: BundleManifest
    tree: dict[str, list[str]]  # supercluster -> [clusters]

    @classmethod
    def load(cls, bundle_dir: Path) -> TaxonomyTree:
        return cls(load_manifest(bundle_dir), load_taxonomy(bundle_dir))

    def _decision(self, level: DecisionLevel, parent: str | None) -> DecisionArtifact:
        for d in self.manifest.decisions:
            if d.level == level and d.parent == parent:
                return d
        raise ReferenceBuildError(f"No {level} decision (parent={parent!r}) in bundle manifest")

    def supercluster_of(self, cluster: str) -> str:
        for sc, clusters in self.tree.items():
            if cluster in clusters:
                return sc
        raise ReferenceBuildError(f"Cluster {cluster!r} not found in taxonomy tree")

    def path_to(
        self, target: str, target_level: DecisionLevel
    ) -> list[tuple[DecisionArtifact, str]]:
        """Ordered [(decision, on-path class to keep), ...] from root to target."""
        if target_level == "supercluster":
            if target not in self.tree:
                raise ReferenceBuildError(f"Supercluster {target!r} not in taxonomy tree")
            return [(self._decision("supercluster", None), target)]
        # cluster target
        parent = self.supercluster_of(target)  # raises if unknown
        return [
            (self._decision("supercluster", None), parent),
            (self._decision("cluster", parent), target),
        ]
