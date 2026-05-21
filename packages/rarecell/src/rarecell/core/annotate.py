"""Profile-driven CellTypist annotation, Enrichr enrichment, consensus labels.

CellTypist models, Enrichr libraries, and consensus-label rules are all
driven by the TargetCellProfile / caller — nothing here is tissue- or
lineage-specific.
"""

from __future__ import annotations

import os
import time

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from rarecell.profile.schema import TargetCellProfile


def _model_label(model_filename: str) -> str:
    """Strip directory and .pkl extension from a CellTypist model filename.

    e.g. 'Immune_All_Low.pkl' -> 'Immune_All_Low'
         '/path/to/Brain.pkl' -> 'Brain'
    """
    base = os.path.basename(model_filename)
    if base.lower().endswith(".pkl"):
        base = base[:-4]
    return base


def _run_one_celltypist_model(
    adata: ad.AnnData,
    model_name: str,
    majority_voting: bool = True,
) -> pd.DataFrame:
    """Run a single CellTypist model. Returns predicted_labels DataFrame.

    The returned DataFrame is indexed by adata.obs_names with columns
    ``predicted_labels``, ``majority_voting``, ``conf_score``.
    """
    import celltypist
    from celltypist import models as ct_models

    model = ct_models.Model.load(model=model_name)
    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=majority_voting,
    )
    return predictions.predicted_labels


def annotate_celltypist(adata: ad.AnnData, profile: TargetCellProfile) -> None:
    """Run every enabled CellTypist model in the profile.

    For each enabled ``CellTypistRef`` in ``profile.reference_labels.celltypist_models``,
    writes to ``adata.obs``:

      - ``celltypist_{label}_label``           (per-cell predicted label)
      - ``celltypist_{label}_label_majority``  (majority-voting label)
      - ``celltypist_{label}_conf``            (confidence score)

    where ``label`` is the model filename stripped of directory and ``.pkl``.
    """
    for ref in profile.reference_labels.celltypist_models:
        if not ref.enabled:
            continue
        label = _model_label(ref.model)
        result = _run_one_celltypist_model(adata, ref.model)
        adata.obs[f"celltypist_{label}_label"] = result["predicted_labels"].values
        adata.obs[f"celltypist_{label}_label_majority"] = result["majority_voting"].values
        adata.obs[f"celltypist_{label}_conf"] = result["conf_score"].values


def enrichr_cell_type_enrichment(
    gene_lists: dict[str, list[str]],
    gene_sets: list[str],
    organism: str = "human",
    enrichr_pval_cutoff: float = 0.05,
    top_terms_per_group: int = 5,
    min_genes: int = 5,
    sleep_seconds: float = 0.5,
    max_consecutive_failures: int = 2,
) -> dict[str, pd.DataFrame]:
    """Run Enrichr enrichment for one or more gene lists.

    Generic wrapper over ``gseapy.enrichr`` — no built-in defaults for
    ``gene_sets``; caller supplies the Enrichr library names explicitly.

    Parameters
    ----------
    gene_lists
        Mapping of group_id (e.g. cluster ID or panel name) to list of genes.
    gene_sets
        Enrichr library names (e.g. ``["CellMarker_2024", "PanglaoDB_Augmented_2021"]``).
    organism
        Organism name passed to ``gseapy.enrichr`` (default ``"human"``).
    enrichr_pval_cutoff
        Adjusted-p-value cutoff applied to Enrichr results.
    top_terms_per_group
        Top N terms (by adjusted p-value) kept per group.
    min_genes
        Skip groups with fewer than this many genes.
    sleep_seconds
        Delay between Enrichr API calls.
    max_consecutive_failures
        Abort remaining groups after this many consecutive request failures.

    Returns
    -------
    dict[group_id, DataFrame] of enrichment results (one row per term,
    columns from ``gseapy.enrichr``). Empty DataFrame if no significant terms
    or the group was skipped.
    """
    try:
        import gseapy
    except ImportError:
        print("  gseapy not installed. Install with: pip install gseapy")
        return {}

    results: dict[str, pd.DataFrame] = {}
    consecutive_failures = 0

    for group_id, genes in gene_lists.items():
        if len(genes) < min_genes:
            print(f"    Group {group_id}: <{min_genes} genes — skipping enrichment")
            results[group_id] = pd.DataFrame()
            continue

        try:
            enr = gseapy.enrichr(
                gene_list=list(genes),
                gene_sets=gene_sets,
                organism=organism,
                outdir=None,
                no_plot=True,
            )
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"    Group {group_id}: Enrichr query failed — {e}")
            results[group_id] = pd.DataFrame()
            if consecutive_failures >= max_consecutive_failures:
                print(
                    f"    {max_consecutive_failures} consecutive failures "
                    "— aborting remaining groups."
                )
                break
            time.sleep(sleep_seconds)
            continue

        res = enr.results
        res = res[res["Adjusted P-value"] < enrichr_pval_cutoff]
        res = res.sort_values("Adjusted P-value").head(top_terms_per_group)
        results[group_id] = res
        time.sleep(sleep_seconds)

    return results


def plot_enrichr_bubble(
    enrichr_results: dict[str, pd.DataFrame],
    title: str = "Enrichr enrichment",
    top_n: int = 3,
) -> Figure:
    """Bubble plot of top Enrichr enrichment terms per group.

    Parameters
    ----------
    enrichr_results
        Mapping of group_id to enrichment DataFrame (as returned by
        :func:`enrichr_cell_type_enrichment`).
    title
        Plot title.
    top_n
        Top terms per group to show.

    Returns
    -------
    matplotlib Figure. Caller is responsible for saving or showing.
    """
    import matplotlib.pyplot as plt

    plot_rows = []
    for group_id, df in enrichr_results.items():
        if df is None or len(df) == 0:
            continue
        top = df.sort_values("Adjusted P-value").head(top_n)
        for _, row in top.iterrows():
            plot_rows.append(
                {
                    "group": str(group_id),
                    "term": str(row["Term"])[:40],
                    "neg_log10_pval": -np.log10(max(row["Adjusted P-value"], 1e-300)),
                    "library": row.get("Gene_set", ""),
                }
            )

    if not plot_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No significant enrichment terms",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return fig

    plot_df = pd.DataFrame(plot_rows)
    libs = plot_df["library"].unique()
    cmap = plt.get_cmap("tab10", max(len(libs), 1))
    lib_colors = {lib: cmap(i) for i, lib in enumerate(libs)}

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(plot_df["group"].unique()) * 1.2),
            max(6, len(plot_df["term"].unique()) * 0.35),
        )
    )

    for lib in libs:
        sub = plot_df[plot_df["library"] == lib]
        ax.scatter(
            sub["group"],
            sub["term"],
            s=sub["neg_log10_pval"] * 20,
            c=[lib_colors[lib]] * len(sub),
            label=lib,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )

    ax.set_xlabel("Group")
    ax.set_ylabel("Enrichment term")
    ax.set_title(title)
    ax.legend(
        title="Library", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, title_fontsize=8
    )
    fig.tight_layout()
    return fig


def build_consensus_labels(
    adata: ad.AnnData,
    celltypist_key: str,
    original_key: str | None = None,
    output_key: str | None = None,
) -> None:
    """Build a unified consensus label column from CellTypist + optional originals.

    Writes a new column to ``adata.obs`` as a pandas ``Categorical``.
    Unknown / empty values become the string ``"Unknown"``.

    Parameters
    ----------
    adata
        AnnData with ``celltypist_key`` in ``.obs``.
    celltypist_key
        Exact obs column name for CellTypist predictions
        (e.g. ``"celltypist_Immune_All_Low_label_majority"``).
    original_key
        Optional obs column with dataset-provided labels. When given and the
        column exists, original labels override CellTypist where non-null.
    output_key
        Name for the new obs column. If None, auto-generated from inputs.
    """
    if celltypist_key not in adata.obs.columns:
        raise ValueError(
            f"CellTypist column '{celltypist_key}' not found in obs. "
            "Run CellTypist annotation first."
        )

    if output_key is None:
        suffix = celltypist_key.replace("celltypist_", "").replace("_label_majority", "")
        output_key = f"consensus_{suffix}_with_original" if original_key else f"consensus_{suffix}"

    labels = adata.obs[celltypist_key].astype(str).copy()

    if original_key is not None:
        if original_key in adata.obs.columns:
            orig = adata.obs[original_key].astype(str)
            has_original = orig.notna() & (orig != "") & (orig != "nan") & (orig != "None")
            n_original = int(has_original.sum())
            labels[has_original] = orig[has_original]
            print(
                f"  Consensus labels: {n_original:,} from '{original_key}', "
                f"rest from '{celltypist_key}'"
            )
        else:
            print(f"  WARNING: '{original_key}' not in obs — using CellTypist only")

    bad_mask = labels.isna() | (labels == "") | (labels == "nan") | (labels == "None")
    labels[bad_mask] = "Unknown"

    adata.obs[output_key] = pd.Categorical(labels)

    n_labeled = int((labels != "Unknown").sum())
    n_unknown = int((labels == "Unknown").sum())
    n_categories = int(labels.nunique())
    print(
        f"  Created '{output_key}': {n_labeled:,} labeled, "
        f"{n_unknown:,} Unknown, {n_categories} categories"
    )
