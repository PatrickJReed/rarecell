import anndata as ad
import numpy as np
import scanpy as sc
from rarecell.core.clustering import taxonomy_cluster
from rarecell.profile.schema import (
    BatchCorrection,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    QCParams,
    TargetCellProfile,
)
from scipy import sparse


def _profile(in_dataset="harmony"):
    return TargetCellProfile(
        profile_id="t",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        batch_correction=BatchCorrection(in_dataset=in_dataset, batch_key="sample_id"),
    )


def _toy(n=400):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 100)).astype(float))
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * (n // 2) + ["s2"] * (n // 2)})
    a.var_names = [f"G{i}" for i in range(100)]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_taxonomy_cluster_writes_leiden_and_pca():
    a = _toy()
    taxonomy_cluster(a, _profile(), stage="class")
    assert "leiden" in a.obs.columns
    assert "X_pca" in a.obsm
    assert "X_pca_harmony" in a.obsm  # because in_dataset == "harmony"


def test_taxonomy_cluster_skips_harmony_when_none():
    a = _toy()
    taxonomy_cluster(a, _profile(in_dataset="none"), stage="class")
    assert "leiden" in a.obs.columns
    assert "X_pca" in a.obsm
    assert "X_pca_harmony" not in a.obsm
