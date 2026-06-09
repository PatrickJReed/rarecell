import anndata as ad
import numpy as np
from rarecell.core.qc import run_qc, run_scrublet
from rarecell.profile.schema import QCParams
from scipy import sparse


def _toy_adata(n=1000):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 200)).astype(float))
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * n})
    a.var_names = [f"MT-{i}" if i < 10 else f"GENE{i}" for i in range(200)]
    return a


def test_run_qc_filters_with_profile_params():
    a = _toy_adata()
    params = QCParams(
        min_genes_per_cell=150, max_pct_mt=10, max_genes_per_cell=10000, min_cells_per_gene=3
    )
    out = run_qc(a, params)
    assert "n_genes_by_counts" in out.obs.columns
    assert "pct_counts_mt" in out.obs.columns
    assert (out.obs["n_genes_by_counts"] >= 150).all()
    assert (out.obs["pct_counts_mt"] <= 10).all()


def test_run_scrublet_marks_doublets():
    a = _toy_adata(n=300)
    out = run_scrublet(a, batch_key="sample_id", expected_doublet_rate=0.05)
    assert "predicted_doublet" in out.obs.columns
    assert "doublet_score" in out.obs.columns


def test_run_qc_drops_low_gene_and_high_mt_cells():
    a = _toy_adata(n=800)
    params = QCParams(
        min_genes_per_cell=150, max_pct_mt=10, max_genes_per_cell=10000, min_cells_per_gene=3
    )
    out = run_qc(a, params)
    assert (out.obs["n_genes_by_counts"] >= 150).all()
    assert (out.obs["pct_counts_mt"] <= 10).all()
    assert out.n_vars <= a.n_vars
    assert out.n_obs <= a.n_obs


def test_run_qc_is_deterministic():
    a1, a2 = _toy_adata(), _toy_adata()
    p = QCParams(
        min_genes_per_cell=150, max_pct_mt=10, max_genes_per_cell=10000, min_cells_per_gene=3
    )
    assert run_qc(a1, p).n_obs == run_qc(a2, p).n_obs
