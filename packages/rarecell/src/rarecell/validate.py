"""Pre-flight validation: does a profile match an input AnnData?

A fast (no clustering) check that for each marker panel reports:
  - how many genes overlap with adata.var_names
  - mean per-gene expression prevalence (fraction of cells with > 0 counts)
  - mean ± std of the panel score across all cells

Returns a structured dict; overall_status is "pass" if all positive panels
clear a 50% gene-overlap bar, else "fail".
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.profile.schema import MarkerPanel, NegativePanel, TargetCellProfile

_OVERLAP_THRESHOLD = 0.5


def _per_panel_stats(
    adata: ad.AnnData,
    panel_name: str,
    panel: MarkerPanel | NegativePanel,
) -> dict[str, Any]:
    var_names = set(adata.var_names)
    found = [g for g in panel.genes if g in var_names]
    missing = [g for g in panel.genes if g not in var_names]
    overlap_n = len(found)
    overlap_total = len(panel.genes)
    overlap_frac = overlap_n / max(overlap_total, 1)

    # Per-gene prevalence + panel score (only if any gene present)
    if found:
        idx = [adata.var_names.get_loc(g) for g in found]
        X = adata.X[:, idx]
        if hasattr(X, "toarray"):
            X = X.toarray()
        prevalences = (np.asarray(X) > 0).mean(axis=0)
        mean_prevalence = float(prevalences.mean())

        # Score the panel (in-place; we drop the obs col after computing stats)
        score_name = f"_validate_score_{panel_name}"
        sc.tl.score_genes(adata, gene_list=found, score_name=score_name, use_raw=False)
        scores = adata.obs[score_name].to_numpy()
        score_mean = float(np.nanmean(scores))
        score_std = float(np.nanstd(scores))
        del adata.obs[score_name]
    else:
        mean_prevalence = 0.0
        score_mean = 0.0
        score_std = 0.0

    return {
        "panel_name": panel_name,
        "gene_overlap_count": overlap_n,
        "gene_overlap_total": overlap_total,
        "gene_overlap_fraction": overlap_frac,
        "genes_found": found,
        "genes_missing": missing,
        "mean_prevalence": mean_prevalence,
        "score_mean": score_mean,
        "score_std": score_std,
    }


def validate_profile_against_adata(
    adata: ad.AnnData,
    profile: TargetCellProfile,
) -> dict[str, Any]:
    """Run the pre-flight check.

    Returns a structured report:
      {
        "dataset": {n_obs, n_vars, samples},
        "expected_abundance": {min_fraction, max_fraction},
        "positive_markers": {panel_name: {gene_overlap_*, prevalence, score, ...}},
        "negative_markers": {...},
        "overall_status": "pass" | "fail",
      }
    """
    positive = {
        name: _per_panel_stats(adata, name, panel)
        for name, panel in profile.positive_markers.items()
    }
    negative = {
        name: _per_panel_stats(adata, name, panel)
        for name, panel in profile.negative_markers.items()
    }

    all_positive_pass = (
        all(p["gene_overlap_fraction"] >= _OVERLAP_THRESHOLD for p in positive.values())
        if positive
        else False
    )

    samples = sorted(set(map(str, adata.obs.get("sample_id", ["_"]))))

    return {
        "dataset": {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "samples": samples,
        },
        "expected_abundance": {
            "min_fraction": profile.expected_abundance.min_fraction,
            "max_fraction": profile.expected_abundance.max_fraction,
        },
        "positive_markers": positive,
        "negative_markers": negative,
        "overall_status": "pass" if all_positive_pass else "fail",
    }
