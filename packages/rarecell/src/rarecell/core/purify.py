"""Profile-driven surgical purification of suspect clusters.

Ported from als_utils.py:2167-2386 with the following generalizations:

  - Signature is profile-driven rather than T-cell-specific. No ``stage``
    parameter; suspect clusters are recomputed at ``profile.purify.high_resolution``
    and judged against ``profile.purify.min_cluster_purity``.
  - No PDF / figure side effects; ``structlog`` is used in place of prints.
  - Returns a new AnnData that excludes sub-clusters whose pooled positive
    marker pass-fraction falls below ``min_cluster_purity`` (or whose
    contaminant fraction exceeds ``1 - min_cluster_purity``).
  - Disabled / empty-input fast paths return the input AnnData unchanged.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.core.annotate import annotate_celltypist
from rarecell.core.evidence import score_biccn_evidence, score_evidence
from rarecell.core.markers import score_negative_panels, score_profile_markers
from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile

log = get_logger("rarecell.purify")

PURIFY_CLUSTER_KEY = "leiden_purify"


def _has_enabled_celltypist(profile: TargetCellProfile) -> bool:
    return any(ref.enabled for ref in profile.reference_labels.celltypist_models)


def _recluster_high_res(sub_adata: ad.AnnData, resolution: float) -> None:
    """Re-cluster ``sub_adata`` at the given resolution, writing
    ``adata.obs[PURIFY_CLUSTER_KEY]``.

    Reuses existing ``X_pca_harmony`` / ``X_pca`` if available; otherwise runs
    a lightweight PCA on the current matrix. Neighbor count is clamped to
    ``n_obs - 1`` for small subsets.
    """
    n_cells = sub_adata.n_obs
    if n_cells < 3:
        sub_adata.obs[PURIFY_CLUSTER_KEY] = ["0"] * n_cells
        return

    if "X_pca_harmony" in sub_adata.obsm:
        use_rep = "X_pca_harmony"
    elif "X_pca" in sub_adata.obsm:
        use_rep = "X_pca"
    else:
        n_pcs = min(30, sub_adata.n_vars - 1, n_cells - 1)
        n_pcs = max(n_pcs, 2)
        sc.tl.pca(sub_adata, n_comps=n_pcs, svd_solver="arpack", random_state=0)
        use_rep = "X_pca"

    n_neighbors = min(30, n_cells - 1)
    n_neighbors = max(n_neighbors, 2)
    sc.pp.neighbors(sub_adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=0)
    sc.tl.leiden(
        sub_adata,
        resolution=float(resolution),
        flavor="igraph",
        n_iterations=2,
        key_added=PURIFY_CLUSTER_KEY,
        random_state=0,
    )


def _keep_subclusters(
    table,
    profile: TargetCellProfile,
) -> list[str]:
    """Choose sub-clusters to keep based on the profile thresholds.

    A sub-cluster is kept when the mean ``pass_{panel}_frac`` across all
    positive panels (defaulting to 0 for missing columns) is at least
    ``profile.purify.min_cluster_purity`` AND, when ``is_contaminant_frac`` is
    populated, that fraction does not exceed ``1 - min_cluster_purity``.
    """
    min_purity = float(profile.purify.min_cluster_purity)
    max_contam = 1.0 - min_purity

    pass_cols = [
        f"pass_{name}_frac"
        for name in profile.positive_markers
        if f"pass_{name}_frac" in table.columns
    ]

    keep: list[str] = []
    for _, row in table.iterrows():
        if pass_cols:
            vals = [float(row[c]) for c in pass_cols if not _isnan(row[c])]
            mean_pass = float(np.mean(vals)) if vals else 0.0
        else:
            mean_pass = 1.0  # nothing to fail on

        contam = row.get("is_contaminant_frac", 0.0)
        contam = 0.0 if _isnan(contam) else float(contam)

        if mean_pass >= min_purity and contam <= max_contam:
            keep.append(str(row["cluster"]))

    return keep


def _isnan(val) -> bool:
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


def subcluster_and_purify(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    suspect_clusters: list,
    *,
    cluster_key: str = "leiden",
) -> ad.AnnData:
    """Surgically remove contaminant sub-populations from suspect clusters.

    Algorithm:
      1. Extract cells in ``suspect_clusters``.
      2. Re-cluster the sub-population at ``profile.purify.high_resolution``,
         writing ``adata.obs["leiden_purify"]``.
      3. Run the full annotation pipeline on the sub-population
         (positive panels, negative panels, optional CellTypist, optional
         BICCN, evidence table).
      4. For each sub-cluster, keep it iff the mean positive-panel
         pass-fraction is at least ``profile.purify.min_cluster_purity`` and
         (when present) ``is_contaminant_frac`` is at most
         ``1 - min_cluster_purity``.
      5. Return a new AnnData containing all non-suspect cells plus the cells
         from kept sub-clusters.

    Parameters
    ----------
    adata
        Full AnnData with ``cluster_key`` populated in ``.obs``.
    profile
        TargetCellProfile driving the resolution / threshold / pipeline choices.
    suspect_clusters
        Cluster labels (str or int; coerced to str) to subcluster and purify.
    cluster_key
        Obs column holding the existing cluster labels (default ``"leiden"``).

    Returns
    -------
    AnnData
        A new AnnData containing only the cells that survived purification.
        When ``profile.purify.enabled`` is False or ``suspect_clusters`` is
        empty, returns ``adata`` unchanged (same identity).
    """
    if not profile.purify.enabled:
        log.info("purify.disabled_skip", n_obs=int(adata.n_obs))
        return adata
    if not suspect_clusters:
        log.info("purify.no_suspects_skip", n_obs=int(adata.n_obs))
        return adata
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"cluster_key '{cluster_key}' not in adata.obs. Run clustering before purification."
        )

    suspect_ids = {str(c) for c in suspect_clusters}
    suspect_mask = adata.obs[cluster_key].astype(str).isin(suspect_ids)
    n_suspect = int(suspect_mask.sum())
    if n_suspect == 0:
        log.info("purify.suspect_clusters_absent", suspect_clusters=sorted(suspect_ids))
        return adata

    log.info(
        "purify.start",
        n_obs=int(adata.n_obs),
        n_suspect=n_suspect,
        suspect_clusters=sorted(suspect_ids),
        high_resolution=float(profile.purify.high_resolution),
        min_cluster_purity=float(profile.purify.min_cluster_purity),
    )

    sub_adata = adata[suspect_mask].copy()

    # ── Re-cluster at high resolution ──
    _recluster_high_res(sub_adata, profile.purify.high_resolution)
    n_subclusters = int(sub_adata.obs[PURIFY_CLUSTER_KEY].nunique())
    log.info("purify.reclustered", n_subclusters=n_subclusters, n_obs=int(sub_adata.n_obs))

    # ── Full annotation pipeline on the sub-population ──
    use_raw = sub_adata.raw is not None
    score_profile_markers(sub_adata, profile, use_raw=use_raw)
    score_negative_panels(sub_adata, profile, use_raw=use_raw)

    if _has_enabled_celltypist(profile):
        try:
            annotate_celltypist(sub_adata, profile)
        except Exception as e:  # network / model load / etc.
            log.warning(
                "purify.celltypist_failed",
                error_type=type(e).__name__,
                error=str(e),
            )

    if profile.biccn_rules.enabled:
        try:
            score_biccn_evidence(sub_adata, profile, cluster_key=PURIFY_CLUSTER_KEY)
        except Exception as e:
            log.warning(
                "purify.biccn_failed",
                error_type=type(e).__name__,
                error=str(e),
            )

    table = score_evidence(sub_adata, profile, cluster_key=PURIFY_CLUSTER_KEY)

    # ── Decide which sub-clusters to keep ──
    keep_ids = _keep_subclusters(table, profile)
    drop_ids = [
        str(c) for c in table["cluster"].astype(str).tolist() if str(c) not in set(keep_ids)
    ]

    log.info(
        "purify.subcluster_decisions",
        n_subclusters=n_subclusters,
        n_keep=len(keep_ids),
        n_drop=len(drop_ids),
        kept=keep_ids,
        dropped=drop_ids,
    )

    keep_subcluster_mask = sub_adata.obs[PURIFY_CLUSTER_KEY].astype(str).isin(set(keep_ids))
    sub_kept_obs_names = set(sub_adata.obs_names[keep_subcluster_mask.to_numpy()].tolist())

    # ── Build the final keep mask on the original adata ──
    non_suspect_mask = ~suspect_mask.to_numpy()
    keep_mask = non_suspect_mask.copy()
    obs_names_arr = np.asarray(adata.obs_names)
    if sub_kept_obs_names:
        suspect_kept = np.array(
            [name in sub_kept_obs_names for name in obs_names_arr],
            dtype=bool,
        )
        keep_mask = keep_mask | suspect_kept

    out = adata[keep_mask].copy()
    log.info(
        "purify.done",
        n_in=int(adata.n_obs),
        n_out=int(out.n_obs),
        n_dropped=int(adata.n_obs - out.n_obs),
    )
    return out
