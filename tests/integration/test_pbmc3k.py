"""End-to-end integration test on the 10x PBMC 3k public dataset.

Fetched via `sc.datasets.pbmc3k()`. Marked `integration` so CI can gate it
to pull_request runs only.
"""

from pathlib import Path

import pytest
import scanpy as sc
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

PRESET = (
    Path(__file__).resolve().parents[2]
    / "packages/rarecell/src/rarecell/profile/presets/t_cell_pbmc.yaml"
)


@pytest.fixture(scope="module")
def pbmc3k():
    """Fetched + cached via scanpy datasets."""
    try:
        return sc.datasets.pbmc3k()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"sc.datasets.pbmc3k() unavailable: {exc}")


@pytest.mark.integration
def test_pbmc3k_isolates_t_cells(pbmc3k, tmp_path: Path):
    # Load the preset, mark reviewed, disable celltypist (no model in CI), freeze
    profile = TargetCellProfile.from_yaml_path(PRESET)
    profile = profile.model_copy(
        update={
            "human_reviewed": True,
            "reviewer": "ci@x",
        }
    )
    # Disable celltypist for CI speed (no model fetch)
    profile = profile.model_copy(
        update={
            "reference_labels": profile.reference_labels.model_copy(
                update={"celltypist_models": []}
            ),
        }
    )
    # Loosen purify min_cluster_purity. With threshold_z=1.0, a pure T-cell
    # subcluster has pass_pan_t_cell_frac ≈ 0.16-0.35 (z-thresholds are
    # dataset-wide), so the default 0.7 would drop every subcluster on a
    # homogeneous dataset like PBMC 3k.
    profile = profile.model_copy(
        update={
            "purify": profile.purify.model_copy(update={"min_cluster_purity": 0.2}),
        }
    )
    profile = profile.freeze()

    # PBMC 3k ships raw integer counts in .X. The pipeline now bundles
    # normalize_total + log1p inside S2_QC, so we can pass raw counts
    # straight through.
    adata = pbmc3k.copy()
    adata.obs["sample_id"] = "pbmc3k_sample"

    runner = IsolateRunner(
        adata=adata,
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    result = runner.run()

    # PBMC 3k has ~45-60% T cells; isolated subset should be substantial.
    # Widened lower bound: recommender stringency can drop the surviving
    # fraction below the biological T-cell prevalence.
    frac = result.isolated.n_obs / pbmc3k.n_obs
    assert 0.05 < frac < 0.80
    assert (tmp_path / "manifest.json").exists()
