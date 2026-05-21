"""Generate a synthetic AnnData with planted T-cell-like rare population.

5000 cells; 4 clusters at 30%, 40%, 25%, 5%. The 5% cluster has high
CD3D/CD3E expression (the "rare T cells"). Other clusters express
neuron-like (RBFOX3), astrocyte-like (GFAP), or B-cell-like (MS4A1) markers.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse

GENES = [
    "CD3D", "CD3E", "CD3G", "TRAC",     # T cell positive
    "MS4A1", "CD79A",                    # B cell (negative panel)
    "GFAP", "AQP4", "ALDH1L1",           # astrocyte (negative)
    "RBFOX3", "SNAP25", "SYT1",          # neuron (negative)
] + [f"GENE{i}" for i in range(38)]      # filler — 50 genes total


def make_synthetic(seed: int = 0, n_cells: int = 5000) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    sizes = [int(n_cells * f) for f in (0.30, 0.40, 0.25, 0.05)]
    sizes[-1] = n_cells - sum(sizes[:-1])
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])

    X = rng.poisson(0.5, size=(n_cells, len(GENES))).astype(float)
    for ci, marker_idxs in enumerate([(9, 10, 11),     # neuron
                                       (6, 7, 8),       # astrocyte
                                       (4, 5),          # B cell
                                       (0, 1, 2, 3)]):  # T cell (the rare one)
        rows = np.where(labels == ci)[0]
        for j in marker_idxs:
            X[rows, j] += rng.poisson(15, size=rows.shape[0])

    n_per_sample = n_cells // 4
    sample_id = np.repeat([f"s{i}" for i in range(4)], n_per_sample)
    sample_id = np.concatenate([sample_id, ["s3"] * (n_cells - len(sample_id))])

    a = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs={"sample_id": sample_id, "true_cluster": labels.astype(str)},
        var={"gene": GENES},
    )
    a.var_names = GENES
    a.layers["counts"] = a.X.copy()
    return a
