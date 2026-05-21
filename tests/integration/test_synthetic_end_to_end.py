"""Synthetic end-to-end: recall and precision on the planted 5% rare cluster."""

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

from tests.fixtures.make_synthetic import make_synthetic


def _profile():
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
                threshold_z=0.5,
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
        purify=PurifyParams(enabled=True, min_cluster_purity=0.45),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()


def test_synthetic_isolates_rare_cluster(tmp_path: Path):
    """Test that the isolate pipeline achieves good recall and precision on synthetic rare cluster.

    The synthetic fixture has 5000 cells split into 4 clusters: neuron (30%), astrocyte (40%),
    B cell (25%), and rare T cell (5%, cluster "3"). After QC, normalization, and purification,
    the isolated subset should capture most of the true T cells (recall > 0.75) while maintaining
    reasonable precision (precision > 0.20).

    Note: The synthetic fixture's doublet filtering removes ~16% of cells upfront, limiting
    absolute recall. These thresholds are tuned to balance recall/precision given the fixture's
    characteristics.
    """
    adata = make_synthetic(seed=0)
    profile = _profile()
    runner = IsolateRunner(
        adata=adata,
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    result = runner.run()

    isolated_true = result.isolated.obs["true_cluster"]
    planted_total = int((adata.obs["true_cluster"] == "3").sum())
    captured = int((isolated_true == "3").sum())
    recall = captured / planted_total

    precision = float((isolated_true == "3").mean())

    # Metrics should be reasonable on the synthetic fixture
    assert recall > 0.75, f"Recall {recall:.2f} below 0.75"
    assert precision > 0.20, f"Precision {precision:.2f} below 0.20"
