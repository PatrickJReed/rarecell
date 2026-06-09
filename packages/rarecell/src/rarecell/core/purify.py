"""Surgical purification of suspect clusters via high-resolution subclustering.

When Gate 1 flags one or more clusters as ``purify`` (a mix of target and
contaminant cells), :func:`subcluster_and_purify` re-clusters just those cells
at a higher Leiden resolution, scores each sub-cluster against the profile's
positive marker panels (and, when available, a contaminant fraction), and drops
the sub-clusters that fail the purity bar. The result is a new AnnData holding
every non-suspect cell plus only the surviving suspect cells — the gene space
is preserved and no cells are ever added.

The subclustering pipeline (PCA → neighbors → Leiden) is standard scanpy with a
pinned ``random_state=0`` so identical inputs reproduce identical sub-labels.
Marker scoring is delegated to :mod:`rarecell.core.markers` so the purity bar
uses the same ``pass_<panel>`` semantics as the rest of the pipeline.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.core import markers
from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile

logger = get_logger("rarecell.purify")

# Obs column that holds the high-resolution sub-cluster labels.
PURIFY_CLUSTER_KEY = "leiden_purify"

# Deterministic seed shared across PCA / neighbors / Leiden so that identical
# sub-populations always produce identical sub-cluster labels.
SEED = 0


def _subcluster(adata: ad.AnnData, resolution: float) -> ad.AnnData:
    """Re-cluster ``adata`` in place at ``resolution`` using standard scanpy.

    Runs PCA → neighbors → Leiden (all seeded with ``SEED``) and writes the
    sub-labels to ``obs[PURIFY_CLUSTER_KEY]``. Returns the same object.
    """
    if adata.n_obs < 3:
        adata.obs[PURIFY_CLUSTER_KEY] = ["0"] * adata.n_obs
        return adata

    # PCA needs at least one component fewer than the smaller matrix dimension.
    n_comps = max(1, min(50, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.pca(adata, n_comps=n_comps, random_state=SEED)
    sc.pp.neighbors(adata, random_state=SEED)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=PURIFY_CLUSTER_KEY,
        random_state=SEED,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return adata


def _positive_pass_fraction(sub: ad.AnnData, profile: TargetCellProfile) -> np.ndarray:
    """Per-cell mean pass-fraction across the profile's positive panels.

    Scores every ``positive_markers`` panel via :func:`markers.score_panel`,
    which writes a boolean ``pass_<panel>`` column, then averages those booleans
    across panels to a per-cell fraction in ``[0, 1]``.
    """
    pass_cols = []
    for name, panel in profile.positive_markers.items():
        markers.score_panel(sub, name, panel.genes, panel.threshold_z, use_raw=False)
        pass_cols.append(sub.obs[f"pass_{name}"].to_numpy().astype(float))
    if not pass_cols:
        return np.ones(sub.n_obs, dtype=float)
    return np.asarray(np.mean(np.column_stack(pass_cols), axis=1))


def _contaminant_fraction(sub: ad.AnnData, profile: TargetCellProfile) -> np.ndarray | None:
    """Per-cell contaminant flag (as float) or ``None`` if unavailable.

    Returns ``None`` when the profile declares no negative panels, so callers
    can skip the contaminant half of the purity test entirely.
    """
    if not profile.negative_markers:
        return None
    markers.score_negative_panels(sub, profile, use_raw=False)
    return np.asarray(sub.obs["is_contaminant"].to_numpy().astype(float))


def subcluster_and_purify(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    suspect_clusters: list[str],
    *,
    cluster_key: str = "leiden",
) -> ad.AnnData:
    """Subcluster suspect clusters and drop the impure sub-clusters.

    Parameters
    ----------
    adata
        Log-normalized AnnData with cluster labels in ``obs[cluster_key]``.
    profile
        Frozen :class:`TargetCellProfile` supplying the purify parameters and
        marker panels used to score each sub-cluster.
    suspect_clusters
        Cluster IDs (matched against ``obs[cluster_key]`` as strings) to
        re-cluster and purify.
    cluster_key
        Obs column holding the existing cluster labels (default ``"leiden"``).

    Returns
    -------
    A new AnnData containing all non-suspect cells plus the cells from the
    suspect sub-clusters that passed the purity bar. The gene space
    (``var_names``) is preserved and ``out.n_obs <= adata.n_obs``. If purify is
    disabled or there are no suspect clusters, ``adata`` is returned unchanged.
    """
    if not profile.purify.enabled or not suspect_clusters:
        return adata

    suspect = {str(c) for c in suspect_clusters}
    labels = adata.obs[cluster_key].astype(str)
    suspect_mask = labels.isin(suspect).to_numpy()

    if not suspect_mask.any():
        logger.info("purify.no_suspect_cells", suspect_clusters=sorted(suspect))
        return adata

    # Cells kept unconditionally (everything outside the suspect clusters).
    keep_mask = ~suspect_mask

    # Re-cluster the suspect sub-population at high resolution.
    sub = adata[suspect_mask].copy()
    _subcluster(sub, profile.purify.high_resolution)

    pos_frac = _positive_pass_fraction(sub, profile)
    contam = _contaminant_fraction(sub, profile)
    sub_labels = sub.obs[PURIFY_CLUSTER_KEY].astype(str)

    min_purity = profile.purify.min_cluster_purity
    max_contam = 1.0 - min_purity

    kept_sub_ids: list[str] = []
    # Boolean over suspect cells: True where the cell's sub-cluster is kept.
    sub_keep = np.zeros(sub.n_obs, dtype=bool)
    for sub_id in sorted(sub_labels.unique()):
        member = (sub_labels == sub_id).to_numpy()
        mean_pos = float(pos_frac[member].mean())
        passes = mean_pos >= min_purity
        contam_frac = None
        if contam is not None:
            contam_frac = float(contam[member].mean())
            passes = passes and contam_frac <= max_contam
        logger.info(
            "purify.subcluster",
            sub_cluster=sub_id,
            n_cells=int(member.sum()),
            mean_positive_pass_fraction=mean_pos,
            is_contaminant_frac=contam_frac,
            kept=passes,
        )
        if passes:
            kept_sub_ids.append(sub_id)
            sub_keep |= member

    # Map the surviving suspect cells back into the parent index space.
    suspect_idx = np.flatnonzero(suspect_mask)
    final_keep = keep_mask.copy()
    final_keep[suspect_idx[sub_keep]] = True

    out = adata[final_keep].copy()
    logger.info(
        "purify.done",
        n_input=int(adata.n_obs),
        n_suspect=int(suspect_mask.sum()),
        n_sub_clusters=int(sub_labels.nunique()),
        n_kept_sub_clusters=len(kept_sub_ids),
        n_output=int(out.n_obs),
    )
    return out
