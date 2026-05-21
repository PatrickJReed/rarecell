"""Tests for the IsolateRunner state machine driver."""
from __future__ import annotations

import sys
from pathlib import Path

from rarecell.profile.schema import (
    BatchCorrection,
    BICCNRules,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    PurifyParams,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

# Make top-level tests/fixtures importable
_fixtures_path = Path(__file__).resolve().parents[4] / "tests" / "fixtures"
if str(_fixtures_path) not in sys.path:
    sys.path.insert(0, str(_fixtures_path))

from make_synthetic import make_synthetic  # noqa: E402


def _profile_for_synthetic() -> TargetCellProfile:
    return TargetCellProfile(
        profile_id="syn-t",
        name="syn T",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.02, max_fraction=0.10),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D", "CD3E", "CD3G", "TRAC"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(
            min_genes_per_cell=10,
            max_pct_mt=100,
            max_genes_per_cell=10000,
            min_cells_per_gene=1,
        ),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()


def test_runner_completes_and_returns_isolated_subset(tmp_path: Path) -> None:
    adata = make_synthetic(seed=0)
    profile = _profile_for_synthetic()
    runner = IsolateRunner(
        adata=adata,
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    result = runner.run()
    assert result.isolated.n_obs > 0
    # the planted rare cluster should dominate the kept set
    isolated_true = result.isolated.obs["true_cluster"]
    rare_frac = (isolated_true == "3").mean()
    assert rare_frac > 0.5
