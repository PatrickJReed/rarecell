"""Tests for the rarecell CLI."""

import sys
from pathlib import Path

import yaml
from rarecell.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "isolate" in result.stdout
    assert "draft" in result.stdout
    assert "review" in result.stdout


def test_isolate_subcommand_runs_on_synthetic(tmp_path: Path):
    """Smoke test: write synthetic AnnData + frozen profile, invoke isolate."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic

    adata = make_synthetic(seed=0)
    adata_path = tmp_path / "input.h5ad"
    adata.write_h5ad(adata_path)

    profile_yaml = {
        "schema_version": "1.0",
        "profile_id": "cli-test-tcell",
        "name": "CLI test T cells",
        "description": "Smoke test profile",
        "target_lineage": "lymphoid",
        "tissue": ["pbmc"],
        "expected_abundance": {"min_fraction": 0.02, "max_fraction": 0.10},
        "positive_markers": {
            "pan_t": {
                "genes": ["CD3D", "CD3E", "CD3G", "TRAC"],
                "threshold_z": 1.0,
                "citations": [{"source_id": "pmid:1", "source": "europepmc"}],
            }
        },
        "negative_markers": {},
        "qc": {
            "min_genes_per_cell": 10,
            "max_pct_mt": 100,
            "max_genes_per_cell": 10000,
            "min_cells_per_gene": 1,
        },
        "purify": {"enabled": False},
        "batch_correction": {"in_dataset": "harmony", "batch_key": "sample_id"},
        "human_reviewed": True,
        "reviewer": "ci@x",
        "frozen": False,
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_yaml))

    from rarecell.profile.schema import TargetCellProfile

    frozen = TargetCellProfile.from_yaml_path(profile_path).freeze()
    frozen.to_yaml_path(profile_path)

    out_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "isolate",
            "--input",
            str(adata_path),
            "--profile",
            str(profile_path),
            "--out-dir",
            str(out_dir),
            "--auto-policy",
            "recommendation",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "isolated.h5ad").exists()
    assert (out_dir / "manifest.json").exists()
