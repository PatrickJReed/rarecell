"""Ingest: count validation, symbol conversion, gene filtering, obs-name dedup."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from rarecell.errors import MissingRawCountsError

CountsLocation = Literal["X", "raw", "counts"] | str

AUTOSOMAL_CHROMOSOMES = {str(i) for i in range(1, 23)}

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rarecell"
_DEFAULT_BIOMART_CACHE = _DEFAULT_CACHE_DIR / "biomart_gene_map.tsv"
_DEFAULT_GENE_ANN_CACHE = _DEFAULT_CACHE_DIR / "gene_annotations.tsv"


# ─────────────────────────────────────────────────────────────────────────────
# Count validation
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_counts(matrix) -> bool:
    """Heuristic: integer-valued (or near-integer-valued) and non-negative."""
    sample = matrix[:100].toarray() if hasattr(matrix, "toarray") else np.asarray(matrix[:100])
    sample = np.asarray(sample)
    if sample.size == 0:
        return False
    if (sample < 0).any():
        return False
    return bool(np.allclose(sample, np.round(sample)))


def validate_counts(adata: ad.AnnData) -> CountsLocation:
    """Locate raw integer counts on the AnnData.

    Returns the location label: "X", "raw", or the layer name.
    Raises MissingRawCountsError if not found.
    """
    if _looks_like_counts(adata.X):
        return "X"
    for layer_name in ("counts", "raw_counts", "spliced"):
        if layer_name in adata.layers and _looks_like_counts(adata.layers[layer_name]):
            return layer_name
    if adata.raw is not None and _looks_like_counts(adata.raw.X):
        return "raw"
    raise MissingRawCountsError(
        "No raw integer counts found in .X, .layers['counts'], or .raw. "
        "rarecell requires integer counts for QC, Scrublet, and normalization. "
        "If counts are stored elsewhere, copy them into adata.layers['counts']."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Obs-name deduplication across samples
# ─────────────────────────────────────────────────────────────────────────────


def make_obs_names_unique_across_samples(
    adata_list: list[ad.AnnData],
    sample_ids: list[str],
) -> list[ad.AnnData]:
    """Prefix barcodes with sample_id to ensure uniqueness before concatenation.

    Returns a list of new AnnData copies with obs_names rewritten as
    ``f"{sample_id}_{barcode}"``.
    """
    out = []
    for a, sid in zip(adata_list, sample_ids, strict=True):
        a_copy = a.copy()
        a_copy.obs_names = [f"{sid}_{bc}" for bc in a_copy.obs_names]
        out.append(a_copy)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BioMart helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────


def _load_or_query_biomart(cache_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load cached Ensembl-to-symbol mapping, or query BioMart and cache it.

    Returns a DataFrame with columns ``['ensembl_gene_id', 'external_gene_name']``.
    """
    if cache_path is None:
        cache_path = _DEFAULT_BIOMART_CACHE
    cache_path = Path(cache_path)

    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    from pybiomart import Server

    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    mapping = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])

    mapping = mapping.dropna(subset=["Gene name"])
    mapping = mapping[mapping["Gene name"].str.strip() != ""]
    mapping.columns = ["ensembl_gene_id", "external_gene_name"]
    mapping = mapping.drop_duplicates(subset="ensembl_gene_id", keep="first")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(cache_path, sep="\t", index=False)
    return mapping


def _load_or_query_gene_annotations(
    cache_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Load cached gene annotation table (symbol → biotype + chromosome).

    Returns a DataFrame with columns
    ``['gene_name', 'gene_biotype', 'chromosome_name']``.
    """
    if cache_path is None:
        cache_path = _DEFAULT_GENE_ANN_CACHE
    cache_path = Path(cache_path)

    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    from pybiomart import Server

    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    df = dataset.query(
        attributes=[
            "external_gene_name",
            "gene_biotype",
            "chromosome_name",
        ]
    )
    df.columns = ["gene_name", "gene_biotype", "chromosome_name"]
    df = df.dropna(subset=["gene_name"])
    df = df[df["gene_name"].str.strip() != ""]
    df = df.drop_duplicates(subset="gene_name", keep="first")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, sep="\t", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Gene filtering
# ─────────────────────────────────────────────────────────────────────────────


def get_protein_coding_autosomal_genes(adata: ad.AnnData) -> list[str]:
    """Return the list of gene symbols in ``adata.var_names`` that are
    protein-coding and on autosomes (chromosomes 1-22).

    Genes not present in the BioMart annotation table are excluded
    (conservative).
    """
    annotations = _load_or_query_gene_annotations()
    pc_auto = annotations[
        (annotations["gene_biotype"] == "protein_coding")
        & (annotations["chromosome_name"].astype(str).isin(AUTOSOMAL_CHROMOSOMES))
    ]
    pc_auto_set = set(pc_auto["gene_name"].astype(str).values)
    keep = [g for g in adata.var_names if g in pc_auto_set]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Ensembl → symbol conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_ensembl_to_symbols(
    adata: ad.AnnData,
    cache_path: str | os.PathLike | None = None,
) -> ad.AnnData:
    """Detect Ensembl gene IDs in ``var_names`` and convert to gene symbols.

    No-op if <50% of ``var_names`` match the ENSG pattern.

    Priority for new names:
      1. Existing gene-symbol column in ``adata.var`` (if found)
      2. BioMart mapping
      3. Keep original ENSG ID

    Stores original IDs in ``adata.var['ensembl_id']`` and calls
    ``var_names_make_unique()`` after renaming.
    """
    ensg_mask = adata.var_names.str.match(r"^ENSG\d{11}")
    ensg_frac = float(ensg_mask.mean()) if len(ensg_mask) else 0.0
    if ensg_frac < 0.50:
        # Ensure var_names are a plain string Index (h5ad can store as
        # Categorical, which breaks downstream var_names_make_unique calls).
        if hasattr(adata.var_names, "categories"):
            adata.var_names = adata.var_names.astype(str)
        return adata

    # --- Version stripping: ENSG00000000003.15 → ENSG00000000003 ---
    original_ids = adata.var_names.to_series()
    stripped_ids = original_ids.str.replace(r"\.\d+$", "", regex=True)
    adata.var["ensembl_id"] = original_ids.values
    adata.var_names = stripped_ids.values

    # --- Look for an existing symbol column in adata.var ---
    existing_symbol_col = None
    for col in adata.var.columns:
        if col == "ensembl_id":
            continue
        vals = adata.var[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        ensg_in_col = vals.str.match(r"^ENSG\d{11}").mean()
        numeric_frac = vals.str.match(r"^[\d.]+$").mean()
        median_len = vals.str.len().median()
        n_unique = vals.nunique()
        if ensg_in_col < 0.1 and numeric_frac < 0.5 and median_len < 20 and n_unique > 100:
            existing_symbol_col = col
            break

    # --- BioMart lookup ---
    biomart_map = _load_or_query_biomart(cache_path)
    ensg_to_symbol = dict(
        zip(
            biomart_map["ensembl_gene_id"],
            biomart_map["external_gene_name"],
            strict=False,
        )
    )

    # --- Build new names ---
    new_names = []
    for idx in range(adata.n_vars):
        ensg = adata.var_names[idx]

        if existing_symbol_col is not None:
            sym = str(adata.var[existing_symbol_col].iloc[idx]).strip()
            if sym and sym != "nan" and not re.match(r"^ENSG\d{11}", sym):
                new_names.append(sym)
                continue

        if ensg in ensg_to_symbol:
            new_names.append(ensg_to_symbol[ensg])
            continue

        new_names.append(ensg)

    adata.var_names = pd.Index(new_names, dtype="object")
    adata.var_names_make_unique()
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Restore full gene set after HVG subsetting
# ─────────────────────────────────────────────────────────────────────────────


def _check_normalization(X, context: str = "") -> None:
    """Raise a RuntimeError if X does not look like log1p(CP10K) data.

    log1p(CP10K) has a hard theoretical maximum of log1p(10000) ≈ 9.21.
    Values above 15 indicate raw counts, CP10K (non-log), or CPM.
    """
    x = X.toarray() if issparse(X) else np.asarray(X, dtype=float)
    x_max = float(x.max())
    x_mean = float(x.mean())
    if x_max > 15:
        raise RuntimeError(
            f"[{context}] Data scale ERROR: X.max()={x_max:.1f} — data is NOT "
            f"log1p(CP10K). Expected max ≤ 9.21."
        )
    if x_mean < -1.0:
        raise RuntimeError(
            f"[{context}] Data scale ERROR: X.mean()={x_mean:.3f} — data appears "
            f"to be scaled (not log1p-normalized)."
        )


def restore_full_genes(adata: ad.AnnData) -> ad.AnnData:
    """Restore the full gene set from ``.raw``, preserving computed metadata.

    After HVG selection and scaling, ``adata.X`` holds only scaled HVGs while
    ``.raw`` holds the full log-normalized gene set. This restores the full
    gene set for downstream saving, keeping ``obs``, ``obsm``, ``obsp``, and
    ``uns`` intact. Reconstructs approximate raw integer counts via
    ``expm1`` and stores them in ``layers['counts']``.

    No-op if ``.raw`` is None.
    """
    if adata.raw is None:
        return adata

    n_hvg = adata.n_vars
    full = adata.raw.to_adata()
    full.obs = adata.obs
    full.obsm = adata.obsm
    full.obsp = adata.obsp
    full.uns = adata.uns

    _check_normalization(full.X, context=f"restore_full_genes [{n_hvg} HVGs]")

    if issparse(full.X):
        counts = full.X.copy()
        counts.data = np.expm1(counts.data)
        counts.data = np.rint(counts.data)
        counts = counts.astype(int)
    else:
        counts = np.rint(np.expm1(full.X)).astype(int)
    full.layers["counts"] = counts

    return full
