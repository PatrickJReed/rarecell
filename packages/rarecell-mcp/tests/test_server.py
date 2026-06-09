"""In-process smoke test of rarecell-mcp server."""

import json
import sys
from pathlib import Path

import pytest
from rarecell_mcp.server import build_app


@pytest.fixture
def app():
    return build_app()


def test_app_advertises_four_tools(app):
    names = sorted(app.list_tool_names())
    assert "draft_profile" in names
    assert "validate_input" in names
    assert "run_isolation" in names
    assert "inspect_report" in names


def test_validate_input_finds_counts(app, tmp_path: Path):
    """validate_input on a synthetic AnnData returns the counts location."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic

    adata = make_synthetic(seed=0)
    p = tmp_path / "input.h5ad"
    adata.write_h5ad(p)
    result = app.call_tool("validate_input", {"adata_path": str(p)})
    assert result["counts_location"] in ("X", "counts", "raw")
    assert result["n_obs"] == 5000


def test_inspect_report_reads_manifest(app, tmp_path: Path):
    """inspect_report reads a manifest.json and returns its key fields."""
    manifest = {
        "schema_version": "1.0",
        "run_id": "test-run",
        "started_at": "2026-05-20T10:00:00+00:00",
        "finished_at": "2026-05-20T10:02:00+00:00",
        "profile_id": "test-profile",
        "profile_content_hash": "sha256:abc",
        "isolated_summary": {
            "n_cells": 100,
            "abundance_fraction": 0.05,
            "within_expected_bounds": True,
        },
        "input_summary": {"n_cells": 2000, "n_genes": 500, "samples": ["s1"]},
        "decision_count": {"gate_1": 5, "gate_2": 2},
        "status": "ok",
    }
    report_dir = tmp_path / "run"
    report_dir.mkdir()
    (report_dir / "manifest.json").write_text(json.dumps(manifest))

    result = app.call_tool("inspect_report", {"report_path": str(report_dir)})
    assert result["run_id"] == "test-run"
    assert result["isolated_summary"]["n_cells"] == 100
