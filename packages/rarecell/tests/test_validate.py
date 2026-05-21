"""Tests for rarecell.validate.validate_profile_against_adata."""

import anndata as ad
import numpy as np
from rarecell.profile.schema import (
    BatchCorrection,
    BICCNRules,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    NegativePanel,
    PurifyParams,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)
from rarecell.validate import validate_profile_against_adata


def _profile():
    return TargetCellProfile(
        profile_id="test",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.05, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D", "CD3E", "MISSING_GENE_1", "MISSING_GENE_2"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            ),
        },
        negative_markers={
            "b_cell": NegativePanel(
                genes=["MS4A1", "CD79A"],
                exclusion_threshold_z=1.5,
            ),
        },
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
    )


def _toy_adata():
    rng = np.random.default_rng(0)
    n_cells = 200
    genes = ["CD3D", "CD3E", "MS4A1", "CD79A", "FOO", "BAR"]
    X = rng.poisson(1, size=(n_cells, len(genes))).astype(float)
    X[:50, :2] += 10  # boost CD3D/CD3E in first 50 cells
    X[50:100, 2:4] += 10  # boost MS4A1/CD79A in next 50
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * n_cells})
    a.var_names = genes
    return a


def test_validate_returns_per_panel_overlap():
    profile = _profile()
    adata = _toy_adata()
    report = validate_profile_against_adata(adata, profile)
    # report is a dict with per-panel results
    pan_t = report["positive_markers"]["pan_t"]
    # pan_t panel has 4 genes; 2 are in adata (CD3D, CD3E)
    assert pan_t["gene_overlap_count"] == 2
    assert pan_t["gene_overlap_total"] == 4
    assert pan_t["gene_overlap_fraction"] == 0.5
    assert "CD3D" in pan_t["genes_found"]
    assert "MISSING_GENE_1" in pan_t["genes_missing"]


def test_validate_negative_panel():
    profile = _profile()
    adata = _toy_adata()
    report = validate_profile_against_adata(adata, profile)
    b_cell = report["negative_markers"]["b_cell"]
    # both MS4A1 and CD79A are present
    assert b_cell["gene_overlap_count"] == 2
    assert b_cell["gene_overlap_fraction"] == 1.0


def test_validate_dataset_summary():
    profile = _profile()
    adata = _toy_adata()
    report = validate_profile_against_adata(adata, profile)
    ds = report["dataset"]
    assert ds["n_obs"] == 200
    assert ds["n_vars"] == 6
    assert ds["samples"] == ["s1"]
    # expected_abundance from profile
    assert report["expected_abundance"] == {"min_fraction": 0.05, "max_fraction": 0.6}


def test_validate_overall_status_pass_when_all_panels_above_threshold():
    profile = _profile()
    adata = _toy_adata()
    # Replace the pan_t panel with one fully present in adata
    new_profile = profile.model_copy(
        update={
            "positive_markers": {
                "pan_t": MarkerPanel(
                    genes=["CD3D", "CD3E"],
                    threshold_z=1.0,
                    citations=[Citation(source_id="pmid:1", source="europepmc")],
                )
            }
        }
    )
    report = validate_profile_against_adata(adata, new_profile)
    assert report["overall_status"] == "pass"


def test_validate_overall_status_fail_when_panel_below_threshold():
    profile = _profile()
    adata = _toy_adata()
    # The pan_t panel has 2/4 = 50% overlap; that's the boundary; let's use a
    # panel with fewer matches to be clearly below threshold.
    new_profile = profile.model_copy(
        update={
            "positive_markers": {
                "pan_t": MarkerPanel(
                    genes=["CD3D", "MISSING_1", "MISSING_2", "MISSING_3", "MISSING_4"],
                    threshold_z=1.0,
                    citations=[Citation(source_id="pmid:1", source="europepmc")],
                )
            }
        }
    )
    report = validate_profile_against_adata(adata, new_profile)
    # 1/5 = 0.2 < 0.5 threshold
    assert report["overall_status"] == "fail"
