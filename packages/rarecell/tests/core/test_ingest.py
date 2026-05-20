import anndata as ad
import numpy as np
import pandas as pd
import pytest
from rarecell.core.ingest import (
    get_protein_coding_autosomal_genes,
    make_obs_names_unique_across_samples,
    validate_counts,
)
from rarecell.errors import MissingRawCountsError
from scipy import sparse


def _toy_adata(layer_with_counts=None, x_is_counts=True, has_raw=False):
    n = 100
    rng = np.random.default_rng(0)
    counts = sparse.csr_matrix(rng.poisson(2, size=(n, 50)).astype(float))
    X = counts.copy() if x_is_counts else counts.copy().multiply(0.1)
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * n})
    if layer_with_counts:
        a.layers[layer_with_counts] = counts.copy()
    if has_raw:
        a.raw = ad.AnnData(X=counts.copy())
    return a


def test_validate_counts_finds_X():
    a = _toy_adata(x_is_counts=True)
    assert validate_counts(a) == "X"


def test_validate_counts_finds_layer():
    a = _toy_adata(x_is_counts=False, layer_with_counts="counts")
    assert validate_counts(a) == "counts"


def test_validate_counts_finds_raw():
    a = _toy_adata(x_is_counts=False, has_raw=True)
    assert validate_counts(a) == "raw"


def test_validate_counts_missing_raises():
    a = _toy_adata(x_is_counts=False, has_raw=False)
    with pytest.raises(MissingRawCountsError):
        validate_counts(a)


def test_make_obs_names_unique():
    a = ad.AnnData(X=np.zeros((3, 2)))
    a.obs_names = ["c1", "c2", "c3"]
    b = ad.AnnData(X=np.zeros((3, 2)))
    b.obs_names = ["c1", "c2", "c3"]
    out_a, out_b = make_obs_names_unique_across_samples([a, b], ["s1", "s2"])
    assert list(out_a.obs_names) == ["s1_c1", "s1_c2", "s1_c3"]
    assert list(out_b.obs_names) == ["s2_c1", "s2_c2", "s2_c3"]


def test_get_protein_coding_autosomal_genes(monkeypatch):
    fake_ann = pd.DataFrame({
        "gene_name": ["CD3D", "MT-ATP6", "XIST", "GAPDH"],
        "chromosome_name": ["11", "MT", "X", "12"],
        "gene_biotype": ["protein_coding"] * 4,
    })
    monkeypatch.setattr(
        "rarecell.core.ingest._load_or_query_gene_annotations",
        lambda *a, **kw: fake_ann,
    )
    a = ad.AnnData(X=np.zeros((1, 4)),
                   var={"gene_symbols": ["CD3D", "MT-ATP6", "XIST", "GAPDH"]})
    a.var_names = ["CD3D", "MT-ATP6", "XIST", "GAPDH"]
    keep = get_protein_coding_autosomal_genes(a)
    assert set(keep) == {"CD3D", "GAPDH"}
