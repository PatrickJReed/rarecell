import anndata as ad
import numpy as np
from rarecell.core.io import load_h5ad, save_h5ad


def test_save_load_roundtrip_sanitizes_uns(tmp_path):
    a = ad.AnnData(X=np.zeros((5, 3)))
    a.uns = {1: "non-string-key", "nested": {2: "x"}}  # non-string keys break h5ad writer
    save_h5ad(a, tmp_path / "out.h5ad")
    b = load_h5ad(tmp_path / "out.h5ad")
    assert b.n_obs == 5
