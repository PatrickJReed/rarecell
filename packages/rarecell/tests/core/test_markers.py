import anndata as ad
import numpy as np
import scanpy as sc
from rarecell.core.markers import (
    score_negative_panels,
    score_panel,
    score_profile_markers,
)
from rarecell.profile.schema import (
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    NegativePanel,
    QCParams,
    TargetCellProfile,
)


def _toy_adata_with_panels():
    rng = np.random.default_rng(0)
    n = 200
    genes = ["CD3D", "CD3E", "CD4", "CD8A", "MS4A1", "NEUROD1"]
    X = rng.poisson(1, size=(n, len(genes))).astype(float)
    X[:50, :2] += 10
    X[50:100, 4] += 10
    a = ad.AnnData(X=X, var={"gene": genes})
    a.var_names = genes
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_score_panel_writes_obs():
    a = _toy_adata_with_panels()
    score_panel(a, "pan_t", ["CD3D", "CD3E"], threshold_z=0.5, use_raw=False)
    assert "score_pan_t" in a.obs.columns
    assert "pass_pan_t" in a.obs.columns
    assert a.obs["pass_pan_t"][:50].sum() > 30
    assert a.obs["pass_pan_t"][50:100].sum() < 10


def _profile():
    return TargetCellProfile(
        profile_id="t",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D", "CD3E"],
                threshold_z=0.5,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={
            "b_cell": NegativePanel(genes=["MS4A1"], exclusion_threshold_z=0.5),
        },
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_score_profile_markers_writes_all_panels():
    a = _toy_adata_with_panels()
    score_profile_markers(a, _profile(), use_raw=False)
    assert "score_pan_t" in a.obs
    assert "pass_pan_t" in a.obs


def test_score_negative_panels_flags_contaminants():
    a = _toy_adata_with_panels()
    score_negative_panels(a, _profile(), use_raw=False)
    assert "is_contaminant" in a.obs
    assert a.obs["is_contaminant"][50:100].sum() > 30


def test_score_panel_with_use_raw_true_but_no_raw():
    """Regression: score_panel(use_raw=True) must not crash when adata.raw is None."""
    a = _toy_adata_with_panels()
    assert a.raw is None
    # Default use_raw=True path:
    score_panel(a, "pan_t", ["CD3D", "CD3E"], threshold_z=1.0)
    assert "score_pan_t" in a.obs.columns
    assert "pass_pan_t" in a.obs.columns
