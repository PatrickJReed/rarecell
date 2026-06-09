"""Profile-driven QC metrics, cell/gene filtering, and per-sample Scrublet."""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from rarecell.logging import get_logger
from rarecell.profile.schema import QCParams

log = get_logger(__name__)


def run_qc(adata: ad.AnnData, params: QCParams) -> ad.AnnData:
    """Compute QC metrics and filter cells per profile params.

    Side effects on returned adata.obs:
      - n_genes_by_counts, total_counts, pct_counts_mt
      - cells failing filters are removed (not flagged).
    Genes appearing in fewer than params.min_cells_per_gene cells are also removed.
    """
    n_start = adata.n_obs
    g_start = adata.n_vars

    # Stash raw integer counts so downstream stages can recover them
    # after normalization.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Mitochondrial gene flag (human MT- prefix convention).
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Filter 1: min genes per cell
    sc.pp.filter_cells(adata, min_genes=params.min_genes_per_cell)

    # Filter 2: min cells per gene
    sc.pp.filter_genes(adata, min_cells=params.min_cells_per_gene)

    # Filter 3: max % mitochondrial
    adata = adata[adata.obs["pct_counts_mt"] <= params.max_pct_mt, :].copy()

    # Filter 4: max genes per cell
    adata = adata[adata.obs["n_genes_by_counts"] <= params.max_genes_per_cell, :].copy()

    log.info(
        "qc.complete",
        n_start=n_start,
        g_start=g_start,
        n_end=adata.n_obs,
        g_end=adata.n_vars,
        min_genes_per_cell=params.min_genes_per_cell,
        max_pct_mt=params.max_pct_mt,
        max_genes_per_cell=params.max_genes_per_cell,
        min_cells_per_gene=params.min_cells_per_gene,
    )
    return adata


def run_scrublet(
    adata: ad.AnnData,
    *,
    batch_key: str = "sample_id",
    expected_doublet_rate: float = 0.05,
) -> ad.AnnData:
    """Per-batch Scrublet doublet detection.

    Writes adata.obs:
      - doublet_score (float)
      - predicted_doublet (bool)
    Batches with <50 cells skip Scrublet and are marked as non-doublets.
    Predicted doublets are REMOVED before return (not just flagged).
    """
    min_cells_per_batch = 50
    n_before = adata.n_obs

    # Exclude tiny batches that would crash Scrublet's internal PCA
    batch_sizes = adata.obs[batch_key].value_counts()
    small_batches = batch_sizes[batch_sizes < min_cells_per_batch].index
    if len(small_batches) > 0:
        small_mask = adata.obs[batch_key].isin(small_batches)
        adata_large = adata[~small_mask].copy()
        adata_small = adata[small_mask].copy()
        adata_small.obs["predicted_doublet"] = False
        adata_small.obs["doublet_score"] = 0.0
    else:
        adata_large = adata
        adata_small = None

    sc.pp.scrublet(
        adata_large,
        batch_key=batch_key,
        expected_doublet_rate=expected_doublet_rate,
        random_state=0,
    )

    if adata_small is not None:
        adata = ad.concat([adata_large, adata_small])
        adata.obs_names_make_unique()
    else:
        adata = adata_large

    n_doublets = int(adata.obs["predicted_doublet"].sum())

    # Remove predicted doublets in-place before downstream steps.
    adata = adata[~adata.obs["predicted_doublet"]].copy()

    log.info(
        "scrublet.complete",
        n_before=n_before,
        n_doublets=n_doublets,
        n_after=adata.n_obs,
        batch_key=batch_key,
        expected_doublet_rate=expected_doublet_rate,
    )
    return adata
