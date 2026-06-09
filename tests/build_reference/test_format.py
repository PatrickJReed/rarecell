import json
from pathlib import Path

from rarecell.cns import format as fmt


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = fmt.BundleManifest(
        biccn_release="WHB-2023",
        created_with="rarecell-test",
        decisions=[
            fmt.DecisionArtifact(
                level="supercluster",
                parent=None,
                classes=["Astrocyte", "Oligodendrocyte"],
                model_file="nodes/supercluster/_decision.celltypist.pkl",
                model_sha256="0" * 64,
                markers_file="nodes/supercluster/_markers.json",
                metrics={"heldout_accuracy": 0.9},
                per_class={
                    "Astrocyte": fmt.ClassStat(n_cells=100, n_donors=12, included=True),
                    "Oligodendrocyte": fmt.ClassStat(n_cells=100, n_donors=11, included=True),
                },
            )
        ],
    )
    fmt.write_manifest(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()
    loaded = fmt.load_manifest(tmp_path)
    assert loaded == manifest
    assert json.loads((tmp_path / "manifest.json").read_text())["format_version"] == 1


class _FakeModel:
    """Stand-in for a celltypist Model: only needs .write(path)."""

    def __init__(self, payload: bytes = b"MODELBYTES"):
        self.payload = payload

    def write(self, path: str) -> None:
        Path(path).write_bytes(self.payload)


def test_write_decision_emits_files_and_artifact(tmp_path: Path) -> None:
    artifact = fmt.write_decision(
        tmp_path,
        level="supercluster",
        parent=None,
        model=_FakeModel(),
        marker_panels={"Astrocyte": ["AQP4", "GFAP"]},
        per_class={
            "Astrocyte": fmt.ClassStat(n_cells=50, n_donors=10, included=True),
            "Doublet": fmt.ClassStat(n_cells=0, n_donors=1, included=False),
        },
        metrics={"heldout_accuracy": 0.95},
    )
    model_path = tmp_path / artifact.model_file
    markers_path = tmp_path / artifact.markers_file
    assert model_path.exists() and markers_path.exists()
    assert artifact.level == "supercluster" and artifact.parent is None
    assert artifact.classes == ["Astrocyte"]
    assert artifact.model_sha256 == fmt.sha256_file(model_path)
    assert fmt.load_markers(tmp_path, artifact.markers_file) == {"Astrocyte": ["AQP4", "GFAP"]}
