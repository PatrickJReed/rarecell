"""On-disk contract for a CNS reference bundle (read + write).

A bundle is a directory:

    <bundle>/
      manifest.json                              # BundleManifest
      taxonomy.json                              # {supercluster: [clusters...]}
      nodes/supercluster/_decision.celltypist.pkl
      nodes/supercluster/_markers.json
      nodes/cluster/<supercluster>/_decision.celltypist.pkl
      nodes/cluster/<supercluster>/_markers.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from rarecell.errors import ReferenceBuildError

FORMAT_VERSION = 1
DecisionLevel = Literal["supercluster", "cluster"]


class _ModelWriter(Protocol):
    def write(self, path: str) -> None: ...


class ClassStat(BaseModel):
    n_cells: int
    n_donors: int
    included: bool


class DecisionArtifact(BaseModel):
    level: DecisionLevel
    parent: str | None  # supercluster name for cluster-level, else None
    classes: list[str]
    model_file: str  # path relative to bundle root
    model_sha256: str
    markers_file: str  # path relative to bundle root
    metrics: dict[str, float]
    per_class: dict[str, ClassStat]


class BundleManifest(BaseModel):
    format_version: int = FORMAT_VERSION
    biccn_release: str
    created_with: str
    decisions: list[DecisionArtifact]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bundle_dir: Path, manifest: BundleManifest) -> None:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def load_manifest(bundle_dir: Path) -> BundleManifest:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        raise ReferenceBuildError(f"No manifest.json in bundle {bundle_dir}")
    return BundleManifest.model_validate_json(path.read_text())


def write_taxonomy(bundle_dir: Path, tree: dict[str, list[str]]) -> None:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "taxonomy.json").write_text(json.dumps(tree, indent=2, sort_keys=True))


def load_taxonomy(bundle_dir: Path) -> dict[str, list[str]]:
    path = Path(bundle_dir) / "taxonomy.json"
    if not path.exists():
        raise ReferenceBuildError(f"No taxonomy.json in bundle {bundle_dir}")
    tree: dict[str, list[str]] = json.loads(path.read_text())
    return tree


def write_markers(bundle_dir: Path, rel_path: str, panels: dict[str, list[str]]) -> None:
    out = Path(bundle_dir) / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panels, indent=2, sort_keys=True))


def load_markers(bundle_dir: Path, rel_path: str) -> dict[str, list[str]]:
    path = Path(bundle_dir) / rel_path
    if not path.exists():
        raise ReferenceBuildError(f"No {rel_path} in bundle {bundle_dir}")
    panels: dict[str, list[str]] = json.loads(path.read_text())
    return panels


def write_annotations(bundle_dir: Path, annotations: dict[str, dict[str, object]]) -> None:
    """Write an optional per-node biological-annotation sidecar (e.g. ABC
    neurotransmitter + Siletti class/subtype/markers per cluster) to
    ``annotations.json``. Values may be strings or lists (curated marker panels)."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "annotations.json").write_text(json.dumps(annotations, indent=2, sort_keys=True))


def load_annotations(bundle_dir: Path) -> dict[str, dict[str, object]]:
    """Load the annotation sidecar; returns ``{}`` if the bundle has none."""
    path = Path(bundle_dir) / "annotations.json"
    if not path.exists():
        return {}
    annotations: dict[str, dict[str, object]] = json.loads(path.read_text())
    return annotations


def load_model(bundle_dir: Path, artifact: DecisionArtifact) -> object:
    """Load and sha-verify the CellTypist model for a decision.

    celltypist is imported lazily so this module stays import-light.
    """
    path = Path(bundle_dir) / artifact.model_file
    if not path.exists():
        raise ReferenceBuildError(
            f"Missing model file {artifact.model_file} in bundle {bundle_dir}"
        )
    actual = sha256_file(path)
    if actual != artifact.model_sha256:
        raise ReferenceBuildError(
            f"Model sha mismatch for {artifact.model_file}: {actual} != {artifact.model_sha256}"
        )
    from celltypist.models import Model

    return Model.load(str(path))


def _decision_dir(level: DecisionLevel, parent: str | None) -> str:
    if level == "supercluster":
        return "nodes/supercluster"
    if level == "cluster":
        if not parent:
            raise ReferenceBuildError("cluster-level decision requires a parent supercluster")
        safe = parent.replace("/", "_").replace(" ", "_")
        return f"nodes/cluster/{safe}"
    raise ReferenceBuildError(f"Unknown decision level: {level!r}")


def write_decision(
    bundle_dir: Path,
    *,
    level: DecisionLevel,
    parent: str | None,
    model: _ModelWriter,
    marker_panels: dict[str, list[str]],
    per_class: dict[str, ClassStat],
    metrics: dict[str, float],
) -> DecisionArtifact:
    bundle_dir = Path(bundle_dir)
    rel_dir = _decision_dir(level, parent)
    model_rel = f"{rel_dir}/_decision.celltypist.pkl"
    markers_rel = f"{rel_dir}/_markers.json"

    model_path = bundle_dir / model_rel
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(model_path))
    write_markers(bundle_dir, markers_rel, marker_panels)

    return DecisionArtifact(
        level=level,
        parent=parent,
        classes=sorted(c for c, s in per_class.items() if s.included),
        model_file=model_rel,
        model_sha256=sha256_file(model_path),
        markers_file=markers_rel,
        metrics=metrics,
        per_class=per_class,
    )
