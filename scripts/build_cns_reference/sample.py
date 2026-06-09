"""Balanced, donor-aware subsampling of a labeled AnnData for classifier training."""

from __future__ import annotations

import anndata as ad
import numpy as np
from numpy.typing import NDArray
from rarecell.cns.format import ClassStat


def balanced_subsample(
    adata: ad.AnnData,
    label_key: str,
    *,
    donor_key: str,
    cells_per_class: int,
    min_donors: int,
    seed: int = 0,
) -> tuple[ad.AnnData, dict[str, ClassStat]]:
    """Down-sample to ~`cells_per_class` cells per label, dropping classes seen
    in fewer than `min_donors` donors. Returns (subset, per-class stats)."""
    rng = np.random.default_rng(seed)
    stats: dict[str, ClassStat] = {}
    keep: list[str] = []

    for cls, grp in adata.obs.groupby(label_key, observed=True):
        n_donors = int(grp[donor_key].nunique())
        if n_donors < min_donors:
            stats[str(cls)] = ClassStat(n_cells=0, n_donors=n_donors, included=False)
            continue
        names: NDArray[np.str_] = grp.index.to_numpy(dtype=str)
        if len(names) > cells_per_class:
            names = rng.choice(names, size=cells_per_class, replace=False)
        keep.extend(names.tolist())
        stats[str(cls)] = ClassStat(n_cells=len(names), n_donors=n_donors, included=True)

    # Preserve original ordering for determinism independent of dict order.
    keep_set = set(keep)
    ordered = [n for n in adata.obs_names if n in keep_set]
    return adata[ordered].copy(), stats
