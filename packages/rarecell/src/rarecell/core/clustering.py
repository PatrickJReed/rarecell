"""Taxonomy clustering built on scanpy / scikit-learn primitives.

This module turns a log-normalized expression matrix into a stable Leiden
clustering, choosing the resolution by a silhouette (+ optional marker-purity)
scan. Everything is seeded with ``random_state=0`` so that identical inputs
produce identical cluster labels.

Public surface:

- :func:`taxonomy_cluster` — full HVG → (cell cycle) → PCA → (harmony) →
  neighbors → resolution scan → Leiden → UMAP pipeline, mutating ``adata``
  in place.
- :func:`finalize_taxonomy_cluster` — re-run Leiden at a chosen resolution and
  recompute per-cluster quality, using parameters stashed by
  ``taxonomy_cluster``.
- :func:`scan_leiden_resolution` — score a grid of Leiden resolutions and pick
  the best.
- :func:`compute_cluster_quality` — per-cluster mean silhouette + a low-quality
  flag.
- :func:`compute_marker_purity` — marker-gene purity per candidate resolution
  via Wilcoxon differential expression on a subsample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_samples, silhouette_score

from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anndata as ad

logger = get_logger("rarecell.clustering")

# Deterministic seed shared across PCA / neighbors / Leiden / Harmony / UMAP and
# all silhouette subsampling. Pinning a single value keeps results reproducible.
SEED = 0

# Cap on the number of cells used for silhouette scoring; larger inputs are
# subsampled (with ``SEED``) to keep the O(n^2) silhouette tractable.
_MAX_SILHOUETTE_CELLS = 5_000

# Cell-cycle gene lists from Tirosh et al. 2016 (Science 352:189-196,
# "Dissecting the multicellular ecosystem of metastatic melanoma by
# single-cell RNA-seq"), the canonical public S and G2/M marker sets that
# scanpy's tutorials use for cell-cycle scoring.
_TIROSH_S_GENES = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]
_TIROSH_G2M_GENES = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]

# Default resolution grid scanned by ``taxonomy_cluster`` when the caller does
# not override it via ``params``.
_DEFAULT_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _default_leiden_key(stage: str) -> str:
    """Return the ``adata.obs`` key holding Leiden labels for ``stage``.

    ``stage="class"`` uses the bare ``"leiden"`` key (the contract the
    IsolateRunner and tests rely on); any other stage gets a suffixed key so
    multiple stages can coexist on the same AnnData without clobbering.
    """
    return "leiden" if stage == "class" else f"leiden_{stage}"


def _params_key(stage: str) -> str:
    """``adata.uns`` key under which finalize parameters are stashed."""
    return f"_taxonomy_params_{stage}"


def _subsample_index(n: int, max_n: int, seed: int = SEED) -> np.ndarray:
    """Return a deterministic index array of ``min(n, max_n)`` positions."""
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _leiden(adata: ad.AnnData, *, resolution: float, key_added: str) -> None:
    """Deterministic Leiden wrapper with the modern igraph flavor."""
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=SEED,
        key_added=key_added,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )


def compute_cluster_quality(
    adata: ad.AnnData,
    stage: str,
    leiden_key: str | None = None,
    use_rep: str | None = None,
    n_pcs: int | None = None,
) -> pd.DataFrame:
    """Per-cluster mean silhouette, flagging clusters with mean < 0.

    Silhouette is computed over ``use_rep`` (defaulting to the representation
    stored for ``stage`` by :func:`taxonomy_cluster`, else ``"X_pca"``). The
    returned DataFrame has one row per cluster with columns ``cluster``,
    ``n_cells``, ``mean_silhouette`` and a boolean ``low_quality`` flag.
    """
    params = adata.uns.get(_params_key(stage), {})
    if leiden_key is None:
        leiden_key = params.get("leiden_key") or _default_leiden_key(stage)
    if use_rep is None:
        use_rep = params.get("use_rep") or (
            "X_pca_harmony"
            if "X_pca_harmony" in adata.obsm
            else "X_pca"
            if "X_pca" in adata.obsm
            else None
        )
    if n_pcs is None:
        n_pcs = params.get("n_pcs")

    if leiden_key not in adata.obs:
        raise ValueError(f"compute_cluster_quality: missing adata.obs[{leiden_key!r}].")
    if use_rep is None or use_rep not in adata.obsm:
        raise ValueError(
            f"compute_cluster_quality: representation {use_rep!r} not found in adata.obsm."
        )

    rep = np.asarray(adata.obsm[use_rep])
    if n_pcs is not None:
        rep = rep[:, :n_pcs]
    labels = adata.obs[leiden_key].to_numpy()

    unique_labels = pd.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        # Silhouette is undefined for a single cluster; report a neutral score.
        single_rows = [
            {
                "cluster": str(cl),
                "n_cells": int(np.sum(labels == cl)),
                "mean_silhouette": float("nan"),
                "low_quality": False,
            }
            for cl in unique_labels
        ]
        logger.info("cluster_quality.single_cluster", stage=stage, leiden_key=leiden_key)
        return pd.DataFrame(single_rows)

    idx = _subsample_index(rep.shape[0], _MAX_SILHOUETTE_CELLS)
    sub_rep = rep[idx]
    sub_labels = labels[idx]
    # Guard against the subsample collapsing to a single cluster.
    if len(pd.unique(sub_labels)) < 2:
        sub_rep, sub_labels = rep, labels

    sil = silhouette_samples(sub_rep, sub_labels)
    sil_series = pd.Series(sil, index=pd.Index(sub_labels))
    rows: list[dict[str, object]] = []
    for cl in unique_labels:
        present = sil_series.index == cl
        mean_sil = float(sil_series[present].mean()) if present.any() else float("nan")
        rows.append(
            {
                "cluster": str(cl),
                "n_cells": int(np.sum(labels == cl)),
                "mean_silhouette": mean_sil,
                "low_quality": bool(present.any() and mean_sil < 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    logger.info(
        "cluster_quality.computed",
        stage=stage,
        n_clusters=n_clusters,
        n_low_quality=int(df["low_quality"].sum()),
    )
    return df


def compute_marker_purity(
    adata: ad.AnnData,
    labels_per_res: dict[float, np.ndarray],
    stage: str,
    max_de_cells: int = 10_000,
    n_top_genes: int = 10,
) -> pd.DataFrame:
    """Marker-gene purity per candidate resolution.

    For each resolution's labels, run a Wilcoxon rank-genes test on a fixed
    subsample (≤ ``max_de_cells``), take the top ``n_top_genes`` markers per
    cluster, and score purity as the mean fraction of expressing cells that fall
    in the marker's own cluster. Returns a DataFrame with columns
    ``resolution``, ``n_clusters`` and ``marker_purity``.
    """
    rows: list[dict[str, object]] = []
    n = adata.n_obs
    idx = _subsample_index(n, max_de_cells)
    sub = adata[idx].copy()

    for res, labels in labels_per_res.items():
        labels = np.asarray(labels)
        sub_labels = pd.Categorical([str(x) for x in labels[idx]])
        n_clusters = len(sub_labels.categories)
        if n_clusters < 2:
            rows.append(
                {"resolution": float(res), "n_clusters": n_clusters, "marker_purity": float("nan")}
            )
            continue

        key = f"_purity_groups_{res}"
        sub.obs[key] = sub_labels
        try:
            sc.tl.rank_genes_groups(
                sub, groupby=key, method="wilcoxon", n_genes=n_top_genes, use_raw=False
            )
        except Exception as exc:  # pragma: no cover - degenerate DE inputs
            logger.warning("marker_purity.de_failed", resolution=float(res), error=str(exc))
            rows.append(
                {"resolution": float(res), "n_clusters": n_clusters, "marker_purity": float("nan")}
            )
            continue

        names = sub.uns["rank_genes_groups"]["names"]
        groups = list(names.dtype.names)
        purities: list[float] = []
        for g in groups:
            top = [str(names[g][i]) for i in range(min(n_top_genes, len(names[g])))]
            in_mask = (sub.obs[key] == g).to_numpy()
            for gene in top:
                if gene not in sub.var_names:
                    continue
                col = sub[:, gene].X
                expr = (
                    np.asarray(col.todense()).ravel()
                    if hasattr(col, "todense")
                    else np.asarray(col).ravel()
                )
                expressing = expr > 0
                total_expr = int(expressing.sum())
                if total_expr == 0:
                    continue
                in_cluster_expr = int((expressing & in_mask).sum())
                purities.append(in_cluster_expr / total_expr)
        marker_purity = float(np.mean(purities)) if purities else float("nan")
        rows.append(
            {
                "resolution": float(res),
                "n_clusters": n_clusters,
                "marker_purity": marker_purity,
            }
        )

    logger.info("marker_purity.computed", stage=stage, n_resolutions=len(rows))
    return pd.DataFrame(rows)


def scan_leiden_resolution(
    adata: ad.AnnData,
    resolutions: list[float],
    use_rep: str,
    n_neighbors: int,
    n_pcs: int | None,
    stage: str,
    leiden_key: str,
) -> tuple[pd.DataFrame, float]:
    """Score a grid of Leiden resolutions and return ``(scan_df, best_resolution)``.

    Each resolution is clustered with ``sc.tl.leiden(random_state=0)`` on the
    existing neighbor graph and scored by the (subsampled) silhouette score,
    blended with marker purity. Resolutions that yield fewer than two clusters
    are penalized. ``scan_df`` has one row per resolution; the chosen resolution
    maximizes the combined score, falling back to the middle of the grid if none
    qualify.
    """
    if not resolutions:
        raise ValueError("scan_leiden_resolution: resolutions list must be non-empty.")
    rep = np.asarray(adata.obsm[use_rep])
    if n_pcs is not None and n_pcs <= rep.shape[1]:
        rep = rep[:, :n_pcs]

    sil_idx = _subsample_index(rep.shape[0], _MAX_SILHOUETTE_CELLS)
    sil_rep = rep[sil_idx]

    labels_per_res: dict[float, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for res in resolutions:
        scratch_key = f"{leiden_key}_scan_{res}"
        _leiden(adata, resolution=res, key_added=scratch_key)
        labels = adata.obs[scratch_key].to_numpy()
        labels_per_res[res] = labels
        n_clusters = len(pd.unique(labels))

        if n_clusters < 2:
            sil = -1.0
        else:
            sub_labels = labels[sil_idx]
            if len(pd.unique(sub_labels)) < 2:
                sil = float(silhouette_score(rep, labels))
            else:
                sil = float(silhouette_score(sil_rep, sub_labels))
        rows.append({"resolution": float(res), "n_clusters": n_clusters, "silhouette": sil})

    scan_df = pd.DataFrame(rows)

    # Blend in marker purity when more than one resolution is viable.
    viable = {
        r: labels_per_res[r]
        for r, n in zip(resolutions, scan_df["n_clusters"], strict=True)
        if n >= 2
    }
    if viable:
        purity_df = compute_marker_purity(adata, viable, stage=stage)
        scan_df = scan_df.merge(
            purity_df[["resolution", "marker_purity"]], on="resolution", how="left"
        )
    else:
        scan_df["marker_purity"] = float("nan")

    # Combined score: silhouette dominates, marker purity breaks ties. Penalize
    # degenerate (<2 cluster) resolutions so they never win.
    sil_component = scan_df["silhouette"].fillna(-1.0)
    purity_component = scan_df["marker_purity"].fillna(0.0)
    scan_df["score"] = np.where(
        scan_df["n_clusters"] >= 2,
        sil_component + 0.1 * purity_component,
        -1e9,
    )

    qualified = scan_df[scan_df["n_clusters"] >= 2]
    if qualified.empty:
        best_resolution = float(resolutions[len(resolutions) // 2])
        logger.warning("scan_leiden.no_qualified", stage=stage, fallback=best_resolution)
    else:
        best_idx = qualified["score"].idxmax()
        best_resolution = float(scan_df.loc[best_idx, "resolution"])

    # Clean up scratch label columns so they don't leak into the AnnData.
    for res in resolutions:
        scratch_key = f"{leiden_key}_scan_{res}"
        if scratch_key in adata.obs:
            del adata.obs[scratch_key]

    logger.info(
        "scan_leiden.selected",
        stage=stage,
        best_resolution=best_resolution,
        n_resolutions=len(resolutions),
    )
    return scan_df, best_resolution


def _harmony_enabled(profile: TargetCellProfile) -> bool:
    return profile.batch_correction.in_dataset == "harmony"


def _harmony_integrate(adata: ad.AnnData, *, batch_key: str) -> None:
    """Run Harmony on ``X_pca`` and write the corrected basis to ``X_pca_harmony``.

    We call ``harmonypy.run_harmony`` directly rather than scanpy's
    ``sce.pp.harmony_integrate`` wrapper: that wrapper hard-codes a transpose of
    ``Z_corr`` which is wrong for harmonypy >= 2.0 (whose ``Z_corr`` is already
    cells-by-PCs), producing a mis-shaped obsm. We orient the result by matching
    the cell-axis to ``adata.n_obs`` so it works across harmonypy versions.
    """
    import harmonypy

    z = np.asarray(
        harmonypy.run_harmony(
            adata.obsm["X_pca"],
            adata.obs,
            [batch_key],
            random_state=SEED,
        ).Z_corr
    )
    # Orient to (n_obs, n_pcs) regardless of harmonypy's internal convention.
    if z.shape[0] != adata.n_obs and z.shape[1] == adata.n_obs:
        z = z.T
    # Precondition: n_obs != n_pcs (guaranteed by the n_pcs cap in taxonomy_cluster).
    # If they were equal the orient guard above could not detect a mis-oriented matrix.
    assert (
        z.shape[0] != z.shape[1] or z.shape[0] == adata.n_obs
    ), "Ambiguous harmony output shape: n_obs must not equal n_pcs."
    adata.obsm["X_pca_harmony"] = z


def taxonomy_cluster(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    stage: str = "class",
    *,
    regress_vars: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Cluster ``adata`` in place: HVG → PCA → (Harmony) → Leiden → UMAP.

    Pipeline (all seeded with ``random_state=0``):

    1. ``sc.pp.highly_variable_genes`` (n_top / flavor from ``params``).
    2. Optional cell-cycle scoring via ``sc.tl.score_genes_cell_cycle`` using
       the public Tirosh et al. 2016 S / G2/M gene lists.
    3. Optional ``sc.pp.regress_out`` when ``regress_vars`` is given.
    4. ``sc.pp.pca`` → optional ``harmony_integrate`` (writes ``X_pca_harmony``).
    5. ``sc.pp.neighbors`` on the chosen representation.
    6. ``scan_leiden_resolution`` to pick the resolution, then ``sc.tl.leiden``.
    7. ``sc.tl.umap`` and a stash of the params needed by
       :func:`finalize_taxonomy_cluster` under ``adata.uns``.
    """
    params = dict(params or {})
    leiden_key = _default_leiden_key(stage)

    n_top_genes = params.get("n_top_genes", 2000)
    hvg_flavor = params.get("hvg_flavor", "seurat")
    n_comps = params.get("n_comps", 50)
    n_neighbors = params.get("n_neighbors", 15)
    resolutions = params.get("resolutions", _DEFAULT_RESOLUTIONS)
    run_cell_cycle = params.get("score_cell_cycle", False)

    # 1. Highly variable genes. Cap n_top below the gene count for tiny inputs.
    n_top = min(n_top_genes, max(adata.n_vars - 1, 1))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=hvg_flavor)

    # 2. Optional cell-cycle scoring (Tirosh et al. 2016 gene lists).
    if run_cell_cycle:
        s_genes = [g for g in _TIROSH_S_GENES if g in adata.var_names]
        g2m_genes = [g for g in _TIROSH_G2M_GENES if g in adata.var_names]
        if s_genes and g2m_genes:
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        else:
            logger.info("taxonomy.cell_cycle_skipped", stage=stage, reason="no_cc_genes_present")

    # 3. Optional regression of unwanted variation.
    if regress_vars:
        present = [v for v in regress_vars if v in adata.obs]
        if present:
            sc.pp.regress_out(adata, present)

    # 4. PCA, then optional Harmony batch integration.
    n_pcs = min(n_comps, max(min(adata.n_obs, adata.n_vars) - 1, 1))
    sc.pp.pca(adata, n_comps=n_pcs, random_state=SEED)

    harmony = _harmony_enabled(profile)
    if harmony:
        _harmony_integrate(adata, batch_key=profile.batch_correction.batch_key)
        use_rep = "X_pca_harmony"
    else:
        use_rep = "X_pca"

    # 5. Neighbor graph on the chosen representation.
    n_neighbors = min(n_neighbors, max(adata.n_obs - 1, 2))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=SEED,
    )

    # 6. Resolution scan → Leiden at the chosen resolution.
    _scan_df, best_resolution = scan_leiden_resolution(
        adata,
        resolutions=list(resolutions),
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        stage=stage,
        leiden_key=leiden_key,
    )
    _leiden(adata, resolution=best_resolution, key_added=leiden_key)

    # 7. UMAP embedding + stash finalize parameters.
    sc.tl.umap(adata, random_state=SEED)

    adata.uns[_params_key(stage)] = {
        "use_rep": use_rep,
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "leiden_key": leiden_key,
        "resolution": float(best_resolution),
        "harmony": bool(harmony),
    }
    logger.info(
        "taxonomy.clustered",
        stage=stage,
        leiden_key=leiden_key,
        resolution=best_resolution,
        n_clusters=int(adata.obs[leiden_key].nunique()),
        harmony=harmony,
    )


def finalize_taxonomy_cluster(
    adata: ad.AnnData,
    stage: str,
    resolution: float | None = None,
) -> None:
    """Re-run Leiden at ``resolution`` and recompute per-cluster quality.

    Reads the parameters stashed by :func:`taxonomy_cluster` under
    ``adata.uns[f"_taxonomy_params_{stage}"]``. Raises ``ValueError`` if those
    parameters are absent. When ``resolution`` is None the stored resolution is
    reused. The recomputed quality table is stored under
    ``adata.uns[f"_taxonomy_quality_{stage}"]``.
    """
    key = _params_key(stage)
    if key not in adata.uns:
        raise ValueError(
            f"finalize_taxonomy_cluster: missing adata.uns[{key!r}]. "
            "Call taxonomy_cluster(...) for this stage first."
        )
    params = adata.uns[key]
    leiden_key = params["leiden_key"]
    use_rep = params["use_rep"]
    n_pcs = params.get("n_pcs")
    res = float(resolution) if resolution is not None else float(params["resolution"])

    _leiden(adata, resolution=res, key_added=leiden_key)
    params["resolution"] = res
    adata.uns[key] = params

    quality = compute_cluster_quality(
        adata,
        stage=stage,
        leiden_key=leiden_key,
        use_rep=use_rep,
        n_pcs=n_pcs,
    )
    adata.uns[f"_taxonomy_quality_{stage}"] = quality
    logger.info(
        "taxonomy.finalized",
        stage=stage,
        resolution=res,
        n_clusters=int(adata.obs[leiden_key].nunique()),
        n_low_quality=int(quality["low_quality"].sum()) if len(quality) else 0,
    )
