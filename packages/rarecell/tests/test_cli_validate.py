"""Tests for the rarecell CLI validate-profile subcommand."""

import sys
from pathlib import Path

import yaml
from rarecell.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_validate_profile_help():
    result = runner.invoke(app, ["validate-profile", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout.lower()


def test_validate_profile_passes_on_good_match(tmp_path: Path):
    """Smoke test: a profile whose markers ARE in the synthetic fixture passes."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic

    adata = make_synthetic(seed=0)
    adata_path = tmp_path / "input.h5ad"
    adata.write_h5ad(adata_path)

    profile_yaml = {
        "schema_version": "1.0",
        "profile_id": "cli-validate-test",
        "name": "smoke test",
        "description": "d",
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
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_yaml))

    result = runner.invoke(
        app,
        [
            "validate-profile",
            "--input",
            str(adata_path),
            "--profile",
            str(profile_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    # The output should mention the panel and the overlap
    assert "pan_t" in result.stdout
    assert "CD3D" in result.stdout or "4/4" in result.stdout or "100" in result.stdout


def test_validate_profile_fails_on_gene_mismatch(tmp_path: Path):
    """A profile whose markers don't match the dataset exits non-zero."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic

    adata = make_synthetic(seed=0)
    adata_path = tmp_path / "input.h5ad"
    adata.write_h5ad(adata_path)

    profile_yaml = {
        "schema_version": "1.0",
        "profile_id": "cli-validate-test-bad",
        "name": "smoke test",
        "description": "d",
        "target_lineage": "lymphoid",
        "tissue": ["pbmc"],
        "expected_abundance": {"min_fraction": 0.02, "max_fraction": 0.10},
        "positive_markers": {
            "pan_t": {
                "genes": ["FAKE_GENE_1", "FAKE_GENE_2", "FAKE_GENE_3", "FAKE_GENE_4"],
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
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_yaml))

    result = runner.invoke(
        app,
        [
            "validate-profile",
            "--input",
            str(adata_path),
            "--profile",
            str(profile_path),
        ],
    )
    assert result.exit_code == 1, result.stdout
