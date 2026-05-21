import anndata as ad
import numpy as np
import scanpy as sc
from rarecell.core.purify import subcluster_and_purify
from rarecell.profile.schema import (
    BatchCorrection,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    PurifyParams,
    QCParams,
    TargetCellProfile,
)
from scipy import sparse


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
                genes=["CD3D"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        purify=PurifyParams(enabled=True, high_resolution=2.0, min_cluster_purity=0.5),
        batch_correction=BatchCorrection(in_dataset="none", batch_key="sample_id"),
    )


def test_purify_returns_filtered_adata():
    rng = np.random.default_rng(0)
    n = 300
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 50)).astype(float))
    a = ad.AnnData(X=X, obs={"leiden": ["0"] * n, "sample_id": ["s1"] * n})
    a.var_names = [f"G{i}" for i in range(50)]
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    out = subcluster_and_purify(a, _profile(), suspect_clusters=["0"], cluster_key="leiden")
    assert isinstance(out, ad.AnnData)
    assert out.n_obs <= a.n_obs


def test_purify_disabled_returns_input_unchanged():
    rng = np.random.default_rng(0)
    n = 50
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 10)).astype(float))
    a = ad.AnnData(X=X, obs={"leiden": ["0"] * n, "sample_id": ["s1"] * n})
    a.var_names = [f"G{i}" for i in range(10)]

    profile = _profile()
    profile = profile.model_copy(
        update={
            "purify": PurifyParams(enabled=False, high_resolution=2.0, min_cluster_purity=0.5),
        }
    )
    out = subcluster_and_purify(a, profile, suspect_clusters=["0"], cluster_key="leiden")
    assert out.n_obs == a.n_obs


def test_purify_empty_suspects_returns_input_unchanged():
    rng = np.random.default_rng(0)
    n = 50
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 10)).astype(float))
    a = ad.AnnData(X=X, obs={"leiden": ["0"] * n, "sample_id": ["s1"] * n})
    a.var_names = [f"G{i}" for i in range(10)]

    out = subcluster_and_purify(a, _profile(), suspect_clusters=[], cluster_key="leiden")
    assert out.n_obs == a.n_obs
