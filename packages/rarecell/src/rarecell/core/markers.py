"""Profile-driven marker scoring.

score_panel is a thin wrapper over scanpy.tl.score_genes that also writes
a boolean pass_<name> column based on a z-score threshold.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.profile.schema import TargetCellProfile


def score_panel(
    adata: ad.AnnData,
    name: str,
    genes: list[str],
    threshold_z: float | None = None,
    use_raw: bool = True,
) -> None:
    """Score a marker panel via sc.tl.score_genes.

    Writes adata.obs[f"score_{name}"]. If threshold_z is not None, also writes
    adata.obs[f"pass_{name}"] = score > mean + threshold_z * std.
    """
    if use_raw and adata.raw is not None:
        var_names_set = set(adata.raw.var_names)
    else:
        var_names_set = set(adata.var_names)
    present = [g for g in genes if g in var_names_set]
    if not present:
        adata.obs[f"score_{name}"] = 0.0
        if threshold_z is not None:
            adata.obs[f"pass_{name}"] = False
        return

    sc.tl.score_genes(
        adata,
        gene_list=present,
        score_name=f"score_{name}",
        use_raw=use_raw and adata.raw is not None,
    )
    if threshold_z is not None:
        s = adata.obs[f"score_{name}"]
        adata.obs[f"pass_{name}"] = s > s.mean() + threshold_z * s.std()


def score_profile_markers(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    use_raw: bool = True,
) -> None:
    """Score every positive_markers panel in the profile."""
    for name, panel in profile.positive_markers.items():
        score_panel(adata, name, panel.genes, panel.threshold_z, use_raw=use_raw)


def score_negative_panels(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    use_raw: bool = True,
) -> None:
    """Score negative_markers panels and write is_contaminant flag.

    A cell is_contaminant if ANY negative panel exceeds its exclusion_threshold_z.
    """
    flags = np.zeros(adata.n_obs, dtype=bool)
    for name, panel in profile.negative_markers.items():
        score_panel(adata, name, panel.genes, panel.exclusion_threshold_z, use_raw=use_raw)
        flags |= adata.obs[f"pass_{name}"].to_numpy()
    adata.obs["is_contaminant"] = flags
