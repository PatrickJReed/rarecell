"""Cluster-level evidence scoring.

This module turns per-cell marker scores (written upstream by
:mod:`rarecell.core.markers`) and per-cell reference annotations into a
per-cluster *consensus table* used by the recommender to decide keep/drop/
purify. It also implements beta-binomial *trinarization* and BICCN-style
lineage rules that assign a coarse cell-class label to each cluster.

The trinarization model and the lineage-rule scoring follow the
Cytograph / Siletti-lab approach (Siletti et al., *Science* 2023,
"Transcriptomic diversity of cell types across the adult human brain").
Canonical lineage markers are drawn from that work plus standard
immunology references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import betainc

from rarecell.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anndata as ad
    import matplotlib.figure

    from rarecell.profile.schema import TargetCellProfile

log = get_logger(__name__)

# Minimum winning class score for a BICCN label to be assigned.
# Clusters whose best class score falls below this threshold receive "N/A"/0.0
# so that low-signal clusters do not produce confident labels read as real evidence.
MIN_BICCN_SCORE = 0.5


def _sorted_cluster_ids(clusters: pd.Series) -> list[str]:
    """Return cluster IDs as strings in numeric order (falling back to lexicographic)."""
    cats = [str(c) for c in pd.Categorical(clusters).categories]
    try:
        return sorted(cats, key=lambda x: int(x))
    except (ValueError, TypeError):
        return sorted(cats)


# Canonical public lineage markers. Brain lineages (NEURON, ASTROCYTE, OLIGO,
# MGL) follow Siletti et al., Science 2023 (transcriptomic atlas of the adult
# human brain); immune/myeloid lineages (TCELL, BCELL, NK, MONO, MAC) follow
# standard immunology marker panels. Each entry lists genes expected to be
# expressed ("present") and genes expected to be absent ("absent").
BICCN_MARKER_RULES: dict[str, dict[str, list[str]]] = {
    "TCELL": {"present": ["CD3D", "CD3E", "TRAC"], "absent": []},
    "BCELL": {"present": ["MS4A1", "CD79A"], "absent": []},
    "NK": {"present": ["NKG7", "GNLY", "KLRD1"], "absent": ["CD3D"]},
    "MGL": {"present": ["CSF1R", "P2RY12", "TMEM119"], "absent": ["CD3D"]},
    "MONO": {"present": ["CD14", "VCAN"], "absent": ["CD3D"]},
    "MAC": {"present": ["CD68", "CD163"], "absent": ["CD3D"]},
    "NEURON": {"present": ["RBFOX3", "SYT1", "SNAP25"], "absent": []},
    "ASTROCYTE": {"present": ["AQP4", "GFAP"], "absent": []},
    "OLIGO": {"present": ["MBP", "MOG", "PLP1"], "absent": []},
}


def _trinarize(k: int, n: int, f: float = 0.2, a: float = 1.5, b: float = 2.0) -> float:
    """Beta-binomial trinarization (Siletti 2023 / Cytograph).

    Returns the posterior probability ``P(theta >= f | k, n)`` that the true
    expressing fraction ``theta`` is at least ``f``, given ``k`` cells express
    the gene out of ``n`` cells, under a ``Beta(a, b)`` prior.

    The posterior of ``theta`` is ``Beta(a + k, b + n - k)``. ``betainc`` is the
    *regularized* incomplete beta function, i.e. the CDF of that Beta evaluated
    at ``f`` (= ``P(theta <= f)``), so the answer is ``1 - betainc(...)``.

    Returns a float in ``[0, 1]``. Deterministic; no randomness.
    """
    if n <= 0:
        return 0.0
    cdf_at_f = betainc(a + k, b + n - k, f)  # P(theta <= f)
    prob = 1.0 - float(cdf_at_f)
    # Guard against tiny floating-point excursions outside [0, 1].
    return min(1.0, max(0.0, prob))


def _gene_expressing_count(adata: ad.AnnData, gene: str, mask: np.ndarray) -> int:
    """Number of cells (within ``mask``) with expression > 0 for ``gene``."""
    col = adata[mask, gene].X
    # Densify a single-gene column robustly for sparse or dense X.
    arr = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
    return int(np.count_nonzero(arr > 0))


def _candidate_classes(profile: TargetCellProfile) -> dict[str, dict[str, list[str]]]:
    """Filter BICCN_MARKER_RULES by the profile's class/subclass filters.

    Both filters union; an empty pair of filters selects all classes.
    """
    rules = profile.biccn_rules
    wanted: set[str] = set(rules.class_filter) | set(rules.subclass_filter)
    if not wanted:
        return dict(BICCN_MARKER_RULES)
    return {k: v for k, v in BICCN_MARKER_RULES.items() if k in wanted}


def score_biccn_evidence(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
    f: float = 0.20,
) -> None:
    """Assign a BICCN lineage label to each cluster via trinarized rules.

    No-op when ``profile.biccn_rules.enabled`` is False.

    Writes:
      * ``adata.obs["biccn_label"]`` — per-cell categorical (the label of the
        cell's cluster).
      * ``adata.uns["biccn_evidence"]`` — dict with keys ``"trinaries"``
        (``{cluster_id: {gene: P(theta>=f)}}``) and ``"rules"`` (a DataFrame
        with one row per cluster and columns ``cluster``, ``biccn_label``,
        ``biccn_score``, ``biccn_details``).
    """
    if not profile.biccn_rules.enabled:
        return

    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key {cluster_key!r} not found in adata.obs")

    candidates = _candidate_classes(profile)
    var_set = set(adata.var_names)
    # Genes we actually need to trinarize: union over candidate rules, present in var.
    needed_genes: set[str] = set()
    for spec in candidates.values():
        for g in spec["present"] + spec["absent"]:
            if g in var_set:
                needed_genes.add(g)

    clusters = adata.obs[cluster_key].astype(str)
    cluster_ids = _sorted_cluster_ids(clusters)

    trinaries: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    per_cell_label = pd.Series(index=adata.obs_names, dtype=object)

    for cid in cluster_ids:
        mask = (clusters == cid).to_numpy()
        n = int(mask.sum())
        gene_probs: dict[str, float] = {}
        for gene in needed_genes:
            k = _gene_expressing_count(adata, gene, mask)
            gene_probs[gene] = _trinarize(k, n, f)
        trinaries[cid] = gene_probs

        class_scores: dict[str, float] = {}
        for cls, spec in candidates.items():
            present = [g for g in spec["present"] if g in var_set]
            absent = [g for g in spec["absent"] if g in var_set]
            if not present:
                # No measurable positive evidence for this class in this dataset.
                class_scores[cls] = 0.0
                continue
            present_factor = float(np.mean([gene_probs[g] for g in present]))
            absent_factor = float(np.mean([1.0 - gene_probs[g] for g in absent])) if absent else 1.0
            class_scores[cls] = present_factor * absent_factor

        if class_scores:
            best_label = max(class_scores, key=lambda c: class_scores[c])
            best_score = class_scores[best_label]
            if best_score < MIN_BICCN_SCORE:
                best_label, best_score = "N/A", 0.0
        else:
            best_label, best_score = "N/A", 0.0

        details = ";".join(f"{c}={class_scores[c]:.3f}" for c in sorted(class_scores))
        rows.append(
            {
                "cluster": cid,
                "biccn_label": best_label,
                "biccn_score": round(float(best_score), 4),
                "biccn_details": details,
            }
        )
        per_cell_label[mask] = best_label

    adata.obs["biccn_label"] = pd.Categorical(per_cell_label)
    rules_df = pd.DataFrame(
        rows, columns=["cluster", "biccn_label", "biccn_score", "biccn_details"]
    )
    adata.uns["biccn_evidence"] = {"trinaries": trinaries, "rules": rules_df}
    log.info(
        "biccn_evidence_scored",
        n_clusters=len(cluster_ids),
        n_genes=len(needed_genes),
        n_classes=len(candidates),
    )


def _celltypist_top(labels: pd.Series) -> tuple[str, float]:
    """Return the most common label in a per-cell label series and its fraction."""
    vc = labels.astype(str).value_counts(normalize=True)
    if vc.empty:
        return "N/A", 0.0
    return str(vc.index[0]), float(vc.iloc[0])


def score_evidence(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> pd.DataFrame:
    """Build the per-cluster consensus table consumed by the recommender.

    One row per cluster. Columns, in order:
      * ``cluster`` (str), ``n_cells`` (int)
      * ``score_{panel}_mean`` per positive panel (mean of ``score_{panel}``;
        nan if the obs column is absent)
      * ``pass_{panel}_frac`` per positive panel (fraction truthy in
        ``pass_{panel}``; nan if absent)
      * ``negative_{panel}_frac`` per negative panel (fraction truthy in
        ``pass_{panel}``; nan if absent)
      * ``is_contaminant_frac`` (fraction truthy in ``is_contaminant``; nan if
        absent)
      * ``celltypist_{label}_top_label`` + ``..._top_label_frac`` per enabled
        CellTypist model
      * ``biccn_top_label`` + ``biccn_top_prob`` when biccn is enabled and
        ``uns["biccn_evidence"]`` is present (``"N/A"`` / ``0.0`` if no match).

    The per-cell ``score_*``/``pass_*``/``is_contaminant`` columns are written
    upstream by :func:`rarecell.core.markers.score_profile_markers`; this
    function only aggregates them.
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key {cluster_key!r} not found in adata.obs")

    obs = adata.obs
    clusters = obs[cluster_key].astype(str)
    cluster_ids = _sorted_cluster_ids(clusters)

    positive_names = list(profile.positive_markers.keys())
    negative_names = list(profile.negative_markers.keys())
    ct_refs = [r for r in profile.reference_labels.celltypist_models if r.enabled]

    biccn_on = profile.biccn_rules.enabled and "biccn_evidence" in adata.uns
    biccn_rules: pd.DataFrame | None = None
    if biccn_on:
        biccn_rules = adata.uns["biccn_evidence"]["rules"]

    rows: list[dict[str, object]] = []
    for cid in cluster_ids:
        mask = (clusters == cid).to_numpy()
        row: dict[str, object] = {"cluster": cid, "n_cells": int(mask.sum())}

        for name in positive_names:
            score_col = f"score_{name}"
            pass_col = f"pass_{name}"
            if score_col in obs.columns:
                row[f"score_{name}_mean"] = round(float(obs.loc[mask, score_col].mean()), 4)
            else:
                row[f"score_{name}_mean"] = np.nan
            if pass_col in obs.columns:
                row[f"pass_{name}_frac"] = round(
                    float(obs.loc[mask, pass_col].astype(bool).mean()), 4
                )
            else:
                row[f"pass_{name}_frac"] = np.nan

        for name in negative_names:
            pass_col = f"pass_{name}"
            if pass_col in obs.columns:
                row[f"negative_{name}_frac"] = round(
                    float(obs.loc[mask, pass_col].astype(bool).mean()), 4
                )
            else:
                row[f"negative_{name}_frac"] = np.nan

        if "is_contaminant" in obs.columns:
            row["is_contaminant_frac"] = round(
                float(obs.loc[mask, "is_contaminant"].astype(bool).mean()), 4
            )
        else:
            row["is_contaminant_frac"] = np.nan

        for ref in ct_refs:
            label = ref.model.split("/")[-1].replace(".pkl", "")
            col = f"celltypist_{label}_label"
            if col in obs.columns:
                top_label, top_frac = _celltypist_top(obs.loc[mask, col])
            else:
                top_label, top_frac = "N/A", 0.0
            row[f"celltypist_{label}_top_label"] = top_label
            row[f"celltypist_{label}_top_label_frac"] = round(top_frac, 4)

        if biccn_on and biccn_rules is not None:
            sub = biccn_rules[biccn_rules["cluster"].astype(str) == cid]
            if not sub.empty:
                row["biccn_top_label"] = str(sub.iloc[0]["biccn_label"])
                row["biccn_top_prob"] = float(sub.iloc[0]["biccn_score"])
            else:
                row["biccn_top_label"] = "N/A"
                row["biccn_top_prob"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def select_clusters(table: pd.DataFrame, recommendation: str) -> list[str]:
    """Return cluster IDs (as strings) whose ``recommendation`` equals the arg."""
    if "recommendation" not in table.columns:
        raise ValueError(
            "table has no 'recommendation' column; "
            "attach recommendations before calling select_clusters."
        )
    sel = table[table["recommendation"] == recommendation]
    return [str(c) for c in sel["cluster"].tolist()]


def render_consensus_table(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """Score evidence and render a simple matplotlib table figure.

    Returns ``(table, figure)``. Long string cells are truncated for display.
    No files are written. Matplotlib is imported lazily.
    """
    import matplotlib.pyplot as plt

    table = score_evidence(adata, profile, cluster_key=cluster_key)

    display = table.copy()
    for col in display.columns:
        if display[col].dtype == object:
            display[col] = (
                display[col].astype(str).map(lambda s: s if len(s) <= 24 else s[:21] + "...")
            )

    n_rows = max(len(display), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(display.columns)), 0.4 * n_rows + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    fig.tight_layout()
    return table, fig
