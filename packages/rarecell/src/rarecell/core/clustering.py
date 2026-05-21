"""Profile-driven taxonomy clustering: HVG → PCA → (Harmony) → silhouette-guided Leiden.

Ported from als_utils.py:1111-1794 with the following generalizations:
  - ``stage`` is a free-form string (no T-cell / class/subclass/subtype semantics).
  - Batch correction is driven by ``profile.batch_correction``:
      * ``in_dataset == "harmony"`` runs Harmony on PCA and stores
        ``X_pca_harmony`` for neighbors/Leiden.
      * ``in_dataset == "none"`` skips Harmony entirely; only PCA is stored.
  - No PDF / figure side effects; structlog is used in place of prints.
  - Random seeds (PCA, neighbors, Leiden, Harmony, silhouette subsampling)
    are pinned to 0 for replay determinism.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from rarecell.core.ingest import get_protein_coding_autosomal_genes
from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile

log = get_logger("rarecell.clustering")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Per-stage defaults; unknown stages fall back to the "class" preset.
TAXONOMY_PARAMS: dict[str, dict[str, float | int]] = {
    "class": {"n_hvgs": 3000, "leiden_resolution": 0.1, "n_pcs": 30},
    "subclass": {"n_hvgs": 2000, "leiden_resolution": 0.2, "n_pcs": 30},
    "subtype": {"n_hvgs": 1000, "leiden_resolution": 0.3, "n_pcs": 30},
}

# Resolution candidates for silhouette-guided scan (same range for all stages).
RESOLUTION_CANDIDATES: dict[str, list[float]] = {
    "class": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "subclass": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "subtype": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
}

# Cell-cycle gene lists (Tirosh et al. 2016).
# fmt: off
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]
G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]
# fmt: on


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (must be defined before taxonomy_cluster references them)
# ─────────────────────────────────────────────────────────────────────────────


def _stage_params(stage: str) -> dict[str, Any]:
    return dict(TAXONOMY_PARAMS.get(stage, TAXONOMY_PARAMS["class"]))


def _resolution_candidates(stage: str, default: float) -> list[float]:
    return RESOLUTION_CANDIDATES.get(stage, [default])


def _safe_protein_coding_autosomal(adata: ad.AnnData) -> list[str]:
    """Return protein-coding autosomal genes present in ``adata.var_names``.

    Wraps :func:`rarecell.core.ingest.get_protein_coding_autosomal_genes` and
    returns an empty list on any failure (e.g. no network, no BioMart cache,
    synthetic gene names). Callers should fall back to all genes when the
    returned list is too short.
    """
    try:
        return list(get_protein_coding_autosomal_genes(adata))
    except Exception as e:  # network / cache / parse errors
        log.warning(
            "clustering.pc_autosomal_lookup_failed",
            error_type=type(e).__name__,
            error=str(e),
        )
        return []


def _smooth_metric(values: pd.Series | np.ndarray) -> np.ndarray:
    """3-point weighted average smoothing (interior [0.25, 0.50, 0.25]).

    Endpoints get heavier self-weight (0.67) and a single neighbor (0.33).
    Dampens anomalous spikes at edge resolutions.
    """
    arr = values.values if hasattr(values, "values") else np.asarray(values)
    n = len(arr)
    if n <= 2:
        return arr.copy().astype(float)

    smoothed = np.empty(n, dtype=float)
    smoothed[0] = 0.67 * arr[0] + 0.33 * arr[1]
    smoothed[-1] = 0.67 * arr[-1] + 0.33 * arr[-2]
    for i in range(1, n - 1):
        smoothed[i] = 0.25 * arr[i - 1] + 0.50 * arr[i] + 0.25 * arr[i + 1]
    return smoothed


def _normalize_01(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns 0.5 for constant arrays."""
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _select_best_resolution(df: pd.DataFrame, default_res: float) -> float:
    """Select best resolution via smoothed geometric mean of focused metrics.

    Uses ``silhouette_mean`` (higher is better) and, when available,
    ``marker_overlap_score`` (lower is better — inverted). Each metric is
    3-point smoothed, then min-max normalized, then combined as a geometric
    mean. Tie-break: prefer the resolution closest to ``default_res``.
    """
    valid = df.dropna(subset=["silhouette_mean"]).copy()
    if len(valid) == 0:
        return default_res

    valid = valid.sort_values("resolution").reset_index(drop=True)

    metric_specs: list[tuple[str, bool, str]] = [
        ("silhouette_mean", False, "silhouette"),
    ]
    if "marker_overlap_score" in valid.columns and valid["marker_overlap_score"].notna().any():
        metric_specs.append(("marker_overlap_score", True, "overlap_inv"))

    norm_cols: list[str] = []
    for col, invert, name in metric_specs:
        raw = valid[col].values.astype(float)
        if invert:
            raw = -raw
        smoothed = _smooth_metric(pd.Series(raw))
        normed = _normalize_01(smoothed)
        normed = np.clip(normed, 0.01, 1.0)
        col_name = f"_norm_{name}"
        valid[col_name] = normed
        norm_cols.append(col_name)

    geo_product = np.ones(len(valid), dtype=float)
    for col_name in norm_cols:
        geo_product *= valid[col_name].values
    valid["_composite_score"] = geo_product ** (1.0 / len(norm_cols))

    max_score = valid["_composite_score"].max()
    ties = valid[np.abs(valid["_composite_score"] - max_score) < 1e-9]
    best_res = ties.loc[(ties["resolution"] - default_res).abs().idxmin(), "resolution"]
    return float(best_res)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-quality and marker-purity helpers
# ─────────────────────────────────────────────────────────────────────────────


def compute_cluster_quality(
    adata: ad.AnnData,
    stage: str,
    leiden_key: str | None = None,
    use_rep: str | None = None,
    n_pcs: int | None = None,
) -> pd.DataFrame:
    """Per-cluster silhouette + flag clusters with mean silhouette < 0.

    Stores the result in ``adata.uns[f"cluster_quality_{stage}"]``.
    """
    from sklearn.metrics import silhouette_samples

    if leiden_key is None:
        leiden_key = _default_leiden_key(stage)
    if use_rep is None:
        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    if n_pcs is None:
        n_pcs = int(_stage_params(stage)["n_pcs"])

    embedding = adata.obsm[use_rep][:, :n_pcs]
    labels = adata.obs[leiden_key].values
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        log.info("clustering.quality_skip", n_clusters=int(n_unique), stage=stage)
        return pd.DataFrame()

    n_cells = adata.n_obs
    max_sil_cells = 20_000
    if n_cells > max_sil_cells:
        rng = np.random.RandomState(0)
        sil_idx = rng.choice(n_cells, max_sil_cells, replace=False)
        sil_idx.sort()
    else:
        sil_idx = np.arange(n_cells)

    embedding_sub = embedding[sil_idx]
    labels_sub = labels[sil_idx]
    sil_samples = silhouette_samples(embedding_sub, labels_sub, metric="cosine")

    def _cl_key(x: Any) -> int:
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    clusters = sorted(np.unique(labels_sub), key=_cl_key)
    rows = []
    for cl in clusters:
        cl_mask = labels_sub == cl
        cl_sil = sil_samples[cl_mask]
        full_count = int((labels == cl).sum())
        rows.append(
            {
                "cluster": cl,
                "n_cells": full_count,
                "silhouette_mean": round(float(np.mean(cl_sil)), 4),
                "silhouette_min": round(float(np.min(cl_sil)), 4),
                "poorly_separated": bool(np.mean(cl_sil) < 0),
            }
        )

    df = pd.DataFrame(rows)
    adata.uns[f"cluster_quality_{stage}"] = df

    n_poor = int(df["poorly_separated"].sum()) if len(df) else 0
    if n_poor > 0:
        log.warning("clustering.poorly_separated_clusters", n=n_poor, stage=stage)
    return df


def compute_marker_purity(
    adata: ad.AnnData,
    labels_per_res: dict[float, np.ndarray],
    stage: str,
    max_de_cells: int = 10_000,
    n_top_genes: int = 10,
) -> pd.DataFrame:
    """Marker gene purity per candidate resolution (Wilcoxon DE on subsample).

    Returns a DataFrame with columns ``resolution``, ``mean_top_lfc``,
    ``frac_sig_clusters``, ``marker_overlap_score``.
    """
    use_raw = adata.raw is not None
    rows: list[dict[str, float]] = []

    for res, labels in labels_per_res.items():
        clusters = np.unique(labels)
        n_clust = len(clusters)
        rng = np.random.RandomState(0)
        per_cluster_cap = max(50, max_de_cells // max(n_clust, 1))

        sub_chunks: list[np.ndarray] = []
        for cl in clusters:
            cl_idx = np.where(labels == cl)[0]
            n_take = min(len(cl_idx), per_cluster_cap)
            sub_chunks.append(rng.choice(cl_idx, n_take, replace=False))
        sub_idx = np.concatenate(sub_chunks) if sub_chunks else np.array([], dtype=int)
        if len(sub_idx) > max_de_cells:
            sub_idx = rng.choice(sub_idx, max_de_cells, replace=False)
        sub_idx.sort()

        adata_sub = adata[sub_idx].copy()
        adata_sub.obs["_tmp_cluster"] = labels[sub_idx]

        if use_raw:
            adata_de = adata_sub.raw.to_adata()
            adata_de.obs["_tmp_cluster"] = adata_sub.obs["_tmp_cluster"].values
        else:
            adata_de = adata_sub

        try:
            sc.tl.rank_genes_groups(
                adata_de,
                groupby="_tmp_cluster",
                method="wilcoxon",
                n_genes=n_top_genes,
                use_raw=False,
            )
        except Exception as e:
            log.warning(
                "clustering.marker_purity_de_failed",
                resolution=float(res),
                error=str(e),
            )
            rows.append(
                {
                    "resolution": float(res),
                    "mean_top_lfc": np.nan,
                    "frac_sig_clusters": np.nan,
                    "marker_overlap_score": np.nan,
                }
            )
            continue

        result = adata_de.uns["rank_genes_groups"]
        group_names = list(result["names"].dtype.names)

        top_genes_per_cluster: dict[str, set[str]] = {}
        lfcs: list[float] = []
        sig_count = 0
        for grp in group_names:
            names = result["names"][grp][:n_top_genes]
            logfcs = result["logfoldchanges"][grp][:n_top_genes]
            pvals_adj = result["pvals_adj"][grp][:n_top_genes]
            top_genes_per_cluster[grp] = set(names)
            lfcs.extend(np.asarray(logfcs).tolist())
            if int(np.sum(pvals_adj < 0.05)) >= n_top_genes / 2:
                sig_count += 1

        mean_lfc = float(np.mean(lfcs)) if lfcs else 0.0
        frac_sig = sig_count / len(group_names) if group_names else 0.0

        if len(group_names) >= 2:
            jaccards: list[float] = []
            for g1, g2 in combinations(group_names, 2):
                s1, s2 = top_genes_per_cluster[g1], top_genes_per_cluster[g2]
                union = len(s1 | s2)
                jaccards.append(len(s1 & s2) / union if union > 0 else 0.0)
            overlap = float(np.mean(jaccards))
        else:
            overlap = 0.0

        rows.append(
            {
                "resolution": float(res),
                "mean_top_lfc": round(mean_lfc, 4),
                "frac_sig_clusters": round(frac_sig, 4),
                "marker_overlap_score": round(overlap, 4),
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette-guided Leiden resolution scan
# ─────────────────────────────────────────────────────────────────────────────


def scan_leiden_resolution(
    adata: ad.AnnData,
    resolutions: list[float],
    use_rep: str,
    n_neighbors: int,
    n_pcs: int,
    stage: str,
    leiden_key: str,
) -> tuple[pd.DataFrame, float]:
    """Run Leiden at each candidate resolution and score with silhouette + purity.

    Returns ``(scan_df, best_resolution)``.
    """
    from sklearn.metrics import silhouette_samples

    default_res = float(_stage_params(stage)["leiden_resolution"])
    embedding = adata.obsm[use_rep][:, :n_pcs]

    n_cells = adata.n_obs
    max_sil_cells = 20_000
    if n_cells > max_sil_cells:
        rng = np.random.RandomState(0)
        sil_idx = rng.choice(n_cells, max_sil_cells, replace=False)
        sil_idx.sort()
    else:
        sil_idx = np.arange(n_cells)
    embedding_sub = embedding[sil_idx]

    rows: list[dict[str, float]] = []
    labels_per_res: dict[float, np.ndarray] = {}

    for res in resolutions:
        tmp_key = f"_leiden_scan_{res}"
        sc.tl.leiden(
            adata,
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            key_added=tmp_key,
            random_state=0,
        )
        labels = adata.obs[tmp_key].values
        n_clust = len(np.unique(labels))

        if n_clust < 2:
            rows.append(
                {
                    "resolution": float(res),
                    "n_clusters": n_clust,
                    "silhouette_mean": np.nan,
                    "silhouette_median": np.nan,
                    "wc_dispersion": np.nan,
                }
            )
            del adata.obs[tmp_key]
            continue

        labels_per_res[float(res)] = labels.copy()

        labels_sub = labels[sil_idx]
        sil_samples = silhouette_samples(embedding_sub, labels_sub, metric="cosine")
        sil_mean = float(np.mean(sil_samples))
        sil_median = float(np.median(sil_samples))

        wc_disp = 0.0
        unique_sub = np.unique(labels_sub)
        for cl in unique_sub:
            cl_mask = labels_sub == cl
            if cl_mask.sum() > 1:
                cl_embed = embedding_sub[cl_mask]
                cl_center = cl_embed.mean(axis=0)
                wc_disp += float(np.mean(np.sum((cl_embed - cl_center) ** 2, axis=1)))
        wc_disp /= max(len(unique_sub), 1)

        rows.append(
            {
                "resolution": float(res),
                "n_clusters": n_clust,
                "silhouette_mean": round(sil_mean, 4),
                "silhouette_median": round(sil_median, 4),
                "wc_dispersion": round(wc_disp, 4),
            }
        )

        del adata.obs[tmp_key]

    df = pd.DataFrame(rows)

    if labels_per_res:
        try:
            purity_df = compute_marker_purity(adata, labels_per_res, stage)
            df = df.merge(purity_df, on="resolution", how="left")
        except Exception as e:
            log.warning("clustering.marker_purity_failed", error=str(e))

    best_res = _select_best_resolution(df, default_res)
    df["selected"] = df["resolution"] == best_res

    adata.uns[f"_labels_per_res_{stage}"] = {
        str(res): labels for res, labels in labels_per_res.items()
    }
    return df, best_res


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def _default_leiden_key(stage: str) -> str:
    """Canonical leiden key for a stage. ``stage="class"`` uses bare "leiden"."""
    return "leiden" if stage == "class" else f"leiden_{stage}"


def taxonomy_cluster(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    stage: str = "class",
    *,
    regress_vars: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Profile-driven iterative clustering for one taxonomy stage.

    Performs HVG selection (restricted to protein-coding autosomal genes when
    annotations are available) → cell-cycle scoring → regression of
    confounders → PCA → optional Harmony batch correction → neighbors →
    silhouette-guided Leiden resolution scan → final Leiden + UMAP.

    Mutates ``adata`` in place:

      - ``adata.raw`` is set to the pre-HVG log-normalized matrix.
      - ``adata.obsm["X_pca"]`` is always written.
      - ``adata.obsm["X_pca_harmony"]`` is written iff
        ``profile.batch_correction.in_dataset == "harmony"`` and the batch
        column is present with >1 unique value.
      - ``adata.obsm["X_umap"]`` is written.
      - ``adata.obs["leiden"]`` (and ``adata.obs[f"leiden_{stage}"]`` when
        ``stage != "class"``) holds the final cluster assignments.
      - ``adata.uns`` is populated with ``recommended_resolution_{stage}``,
        ``resolution_scan_{stage}``, ``_taxonomy_params_{stage}``, and
        ``cluster_quality_{stage}``.

    Parameters
    ----------
    adata
        Log-normalized AnnData (full gene set, raw counts in
        ``.layers['counts']`` is fine but not required).
    profile
        TargetCellProfile; drives batch correction settings.
    stage
        Free-form stage label. Selects defaults from ``TAXONOMY_PARAMS`` /
        ``RESOLUTION_CANDIDATES`` when the stage matches a known key.
    regress_vars
        Variables to regress out before PCA. Default: ``total_counts``,
        ``pct_counts_mt`` (when present in ``adata.obs``) plus
        ``cycle_diff`` (added after cell-cycle scoring, when possible).
    params
        Overrides for ``n_hvgs``, ``leiden_resolution``, ``n_pcs``.
    """
    p = {**_stage_params(stage), **(params or {})}
    n_hvgs = int(p["n_hvgs"])
    n_pcs = int(p["n_pcs"])
    resolution = float(p["leiden_resolution"])

    batch_key = profile.batch_correction.batch_key
    in_dataset = profile.batch_correction.in_dataset

    if regress_vars is None:
        regress_vars = []
        if "total_counts" in adata.obs.columns:
            regress_vars.append("total_counts")
        if "pct_counts_mt" in adata.obs.columns:
            regress_vars.append("pct_counts_mt")
        regress_vars.append("cycle_diff")  # may be removed below if not scored

    n_neighbors = 100 if adata.n_obs >= 10_000 else 30
    leiden_key = _default_leiden_key(stage)

    if adata.n_obs < 500:
        n_pcs = min(n_pcs, max(10, adata.n_obs // 5))
        n_hvgs = min(n_hvgs, adata.n_vars)
        n_neighbors = min(n_neighbors, max(5, adata.n_obs // 10))

    log.info(
        "clustering.start",
        stage=stage,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        leiden_resolution=resolution,
        batch_key=batch_key,
        in_dataset=in_dataset,
        regress_vars=list(regress_vars),
        n_neighbors=n_neighbors,
    )

    # ── Cell-cycle scoring (before subsetting). Graceful skip if too few
    #    genes from S/G2M lists are present.
    s_present = [g for g in S_GENES if g in adata.var_names]
    g_present = [g for g in G2M_GENES if g in adata.var_names]
    if len(s_present) >= 5 and len(g_present) >= 5:
        try:
            sc.tl.score_genes_cell_cycle(
                adata,
                s_genes=s_present,
                g2m_genes=g_present,
            )
            adata.obs["cycle_diff"] = adata.obs["S_score"] - adata.obs["G2M_score"]
            log.info(
                "clustering.cell_cycle_scored",
                n_s=len(s_present),
                n_g2m=len(g_present),
            )
        except Exception as e:
            log.warning("clustering.cell_cycle_score_failed", error=str(e))
            regress_vars = [v for v in regress_vars if v != "cycle_diff"]
    else:
        log.info(
            "clustering.cell_cycle_skipped",
            n_s=len(s_present),
            n_g2m=len(g_present),
        )
        regress_vars = [v for v in regress_vars if v != "cycle_diff"]

    # Stash full gene set in .raw before HVG subsetting.
    adata.raw = adata

    # ── Protein-coding autosomal restriction (with graceful fallback) ──
    pc_auto = _safe_protein_coding_autosomal(adata)
    if len(pc_auto) >= n_hvgs:
        adata_hvg = adata[:, pc_auto].copy()
        log.info(
            "clustering.hvg_candidates_restricted",
            n_pc_autosomal=len(pc_auto),
        )
    else:
        adata_hvg = adata.copy()
        log.warning(
            "clustering.hvg_pc_autosomal_fallback",
            n_pc_autosomal=len(pc_auto),
            n_total=int(adata.n_vars),
        )

    # ── HVG selection ──
    n_hvgs = min(n_hvgs, adata_hvg.n_vars)
    use_batch = (
        batch_key
        if (batch_key in adata_hvg.obs.columns and adata_hvg.obs[batch_key].nunique() > 1)
        else None
    )
    if use_batch is not None:
        sizes = adata_hvg.obs[use_batch].value_counts()
        if sizes.min() < 2:
            log.warning(
                "clustering.batched_hvg_tiny_batch",
                smallest_batch=str(sizes.idxmin()),
                size=int(sizes.min()),
            )
            use_batch = None
    try:
        sc.pp.highly_variable_genes(
            adata_hvg,
            n_top_genes=n_hvgs,
            batch_key=use_batch,
        )
    except (IndexError, ValueError) as e:
        if use_batch is not None:
            log.warning("clustering.batched_hvg_retry", error=str(e))
            sc.pp.highly_variable_genes(
                adata_hvg,
                n_top_genes=n_hvgs,
                batch_key=None,
            )
        else:
            raise

    hvg_names = adata_hvg.var_names[adata_hvg.var.highly_variable]
    log.info("clustering.hvg_selected", n=len(hvg_names))
    del adata_hvg

    # Subset original adata to selected HVGs (this rewrites adata.X / shape).
    # We do this in place by replacing X / var / obs with the sliced copy.
    adata_sub = adata[:, hvg_names].copy()
    _replace_inplace(adata, adata_sub)
    sc.pp.scale(adata, max_value=10)

    # ── Regress confounders ──
    if regress_vars:
        available = [v for v in regress_vars if v in adata.obs.columns]
        if available:
            sc.pp.regress_out(adata, available)
            log.info("clustering.regressed_out", vars=available)

    # ── PCA ──
    n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=0)

    # ── Harmony (optional, profile-driven) ──
    if in_dataset == "harmony":
        run_harmony = batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1
        if run_harmony:
            import harmonypy

            harmony_out = harmonypy.run_harmony(
                adata.obsm["X_pca"],
                adata.obs,
                batch_key,
                max_iter_harmony=20,
                random_state=0,
            )
            Z = harmony_out.Z_corr
            # harmonypy versions differ on orientation.
            if Z.shape[0] == n_pcs and Z.shape[1] == adata.n_obs:
                Z = Z.T
            adata.obsm["X_pca_harmony"] = np.asarray(Z)
            use_rep = "X_pca_harmony"
            log.info(
                "clustering.harmony_done",
                batch_key=batch_key,
                n_batches=int(adata.obs[batch_key].nunique()),
            )
        else:
            use_rep = "X_pca"
            log.info(
                "clustering.harmony_skipped_single_batch",
                batch_key=batch_key,
            )
    else:
        # in_dataset == "none": never run Harmony, never store X_pca_harmony.
        use_rep = "X_pca"
        log.info("clustering.harmony_disabled_by_profile")

    # ── Neighbors + UMAP ──
    sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=0,
    )
    sc.tl.umap(adata, random_state=0)

    # ── Resolution scan (skipped if user provided explicit resolution) ──
    user_overrode = (params or {}).get("leiden_resolution") is not None
    if not user_overrode and adata.n_obs >= 100:
        candidates = _resolution_candidates(stage, resolution)
        scan_df, best_res = scan_leiden_resolution(
            adata,
            candidates,
            use_rep,
            n_neighbors,
            n_pcs,
            stage,
            leiden_key,
        )
        adata.uns[f"resolution_scan_{stage}"] = scan_df
        resolution = best_res
        log.info("clustering.resolution_chosen", resolution=resolution)

    adata.uns[f"recommended_resolution_{stage}"] = resolution
    adata.uns[f"_taxonomy_params_{stage}"] = {
        "use_rep": use_rep,
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "leiden_key": leiden_key,
    }

    finalize_taxonomy_cluster(adata, stage, resolution)


def finalize_taxonomy_cluster(
    adata: ad.AnnData,
    stage: str,
    resolution: float | None = None,
) -> None:
    """Run final Leiden + per-cluster quality at ``resolution`` for ``stage``.

    Requires that :func:`taxonomy_cluster` has already populated
    ``adata.uns[f"_taxonomy_params_{stage}"]``.
    """
    params = adata.uns.get(f"_taxonomy_params_{stage}")
    if params is None:
        raise ValueError(
            f"No taxonomy params found for stage '{stage}'. Run taxonomy_cluster() first."
        )

    leiden_key = params["leiden_key"]
    use_rep = params["use_rep"]
    n_pcs = int(params["n_pcs"])

    if resolution is None:
        resolution = adata.uns.get(f"recommended_resolution_{stage}")
        if resolution is None:
            raise ValueError(f"No recommended resolution for stage '{stage}'.")

    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        flavor="igraph",
        n_iterations=2,
        key_added=leiden_key,
        random_state=0,
    )

    # When the canonical key is stage-suffixed, also mirror to "leiden" for
    # downstream code that expects the scanpy-default obs column.
    if leiden_key != "leiden":
        adata.obs["leiden"] = adata.obs[leiden_key]

    compute_cluster_quality(
        adata,
        stage,
        leiden_key=leiden_key,
        use_rep=use_rep,
        n_pcs=n_pcs,
    )

    n_clusters = int(adata.obs[leiden_key].nunique())
    adata.uns[f"recommended_resolution_{stage}"] = float(resolution)
    log.info(
        "clustering.finalize",
        stage=stage,
        n_clusters=n_clusters,
        resolution=float(resolution),
    )


# ─────────────────────────────────────────────────────────────────────────────
# In-place AnnData replacement (HVG subset)
# ─────────────────────────────────────────────────────────────────────────────


def _replace_inplace(dst: ad.AnnData, src: ad.AnnData) -> None:
    """Replace ``dst``'s data with ``src``'s data, preserving identity.

    AnnData lacks a clean "rewrite from another AnnData" API, so we swap
    the underlying attributes via :meth:`AnnData._init_as_actual` while
    preserving ``dst.raw`` (the full gene set).
    """
    raw = dst.raw
    dst._init_as_actual(
        X=src.X,
        obs=src.obs,
        var=src.var,
        uns=src.uns,
        obsm=src.obsm,
        varm=src.varm,
        obsp=src.obsp,
        varp=src.varp,
        layers=src.layers,
        raw=raw,
    )
