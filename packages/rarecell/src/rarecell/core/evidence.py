"""BICCN trinarization + multi-evidence consensus scoring.

Public surface:

- :func:`score_biccn_evidence` — Bayesian beta-binomial trinarization of BICCN
  marker rules, writing per-cell labels and per-cluster joint probabilities.
- :func:`score_evidence` — orchestrate per-cluster summary across positive
  panels, negative panels, contaminant flag, CellTypist top labels, and BICCN.
- :func:`render_consensus_table` — wrap :func:`score_evidence` with a
  color-coded matplotlib table figure.
- :func:`select_clusters` — filter clusters by recommendation column.

Plus plot helpers (return matplotlib Figures, no PDF saves).
"""

from __future__ import annotations

from math import exp, lgamma, log

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.sparse import issparse

from rarecell.profile.schema import TargetCellProfile

# ── BICCN reference rules ──────────────────────────────────────────────────
# Minimal subset ported from als_utils to support the public API. Profiles
# select which classes to use via biccn_rules.class_filter.
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


# ── Bayesian beta-binomial trinarization ───────────────────────────────────


def _trinarize(k, n, f: float = 0.2, a: float = 1.5, b: float = 2.0):
    """Beta-binomial trinarization score (Siletti et al. 2023 / Cytograph).

    Computes the posterior probability that at least a fraction ``f`` of cells
    in a cluster truly express a gene, given ``k`` expressing cells out of
    ``n``.

    Uses a Beta(a, b) prior on the true expression fraction theta, updated with
    the observed (k, n) to give posterior Beta(a+k, b+n-k). Returns
    P(theta >= f | k, n).
    """
    from scipy.special import betainc, betaln

    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    scalar = k_arr.ndim == 0
    k_arr = np.atleast_1d(k_arr)
    n_arr = np.atleast_1d(n_arr)

    result = np.zeros_like(k_arr, dtype=float)
    for i in range(len(k_arr)):
        ki, ni = k_arr[i], n_arr[i]
        incb = betainc(a + ki, b - ki + ni, f)
        if incb == 0:
            result[i] = 1.0
        else:
            result[i] = 1.0 - exp(
                log(incb)
                + betaln(a + ki, b - ki + ni)
                + lgamma(a + b + ni)
                - lgamma(a + ki)
                - lgamma(b - ki + ni)
            )
    return float(result[0]) if scalar else result


# ── Helpers ────────────────────────────────────────────────────────────────


def _sorted_clusters(values: pd.Series) -> list:
    """Sort cluster labels numerically if possible, else lexically."""
    try:
        return sorted(values.unique(), key=lambda x: int(x))
    except (ValueError, TypeError):
        return sorted(values.unique())


def _filtered_rules(profile: TargetCellProfile) -> dict[str, dict[str, list[str]]]:
    """Return BICCN rules filtered by profile.biccn_rules.class_filter.

    If class_filter is empty, use all known rules.
    """
    cf = profile.biccn_rules.class_filter
    if not cf:
        return dict(BICCN_MARKER_RULES)
    return {k: v for k, v in BICCN_MARKER_RULES.items() if k in cf}


# ── score_biccn_evidence ───────────────────────────────────────────────────


def score_biccn_evidence(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
    f: float = 0.20,
) -> None:
    """BICCN trinarization + presence/absence rule scoring.

    No-op if ``profile.biccn_rules.enabled is False``.

    Writes:
      - ``adata.obs["biccn_label"]`` — per-cell soft assignment (categorical)
      - ``adata.uns["biccn_evidence"]`` — dict with keys ``trinaries`` (per-
        cluster per-gene trinarization), ``rules`` (per-cluster top label +
        score + details DataFrame).
    """
    if not profile.biccn_rules.enabled:
        return

    rules = _filtered_rules(profile)
    if not rules:
        adata.uns["biccn_evidence"] = {
            "trinaries": {},
            "rules": pd.DataFrame(
                columns=["cluster", "biccn_label", "biccn_score", "biccn_details"]
            ),
        }
        return

    data_source = adata.raw if adata.raw is not None else adata
    var_names = list(data_source.var_names)
    var_name_set = set(var_names)

    all_rule_genes: set[str] = set()
    for rule in rules.values():
        all_rule_genes.update(rule["present"])
        all_rule_genes.update(rule["absent"])
    available_genes = sorted(all_rule_genes & var_name_set)

    if len(available_genes) == 0:
        adata.uns["biccn_evidence"] = {
            "trinaries": {},
            "rules": pd.DataFrame(
                columns=["cluster", "biccn_label", "biccn_score", "biccn_details"]
            ),
        }
        return

    gene_to_idx = {g: var_names.index(g) for g in available_genes}
    gene_indices = [gene_to_idx[g] for g in available_genes]

    X_src = adata.raw.X if adata.raw is not None else adata.X
    X_rules = X_src[:, gene_indices]
    if issparse(X_rules):
        X_rules = X_rules.toarray()
    X_rules = np.asarray(X_rules)

    rules_gene_col = {g: i for i, g in enumerate(available_genes)}
    cell_expressed = X_rules > 0  # (n_cells, n_rule_genes)

    # ── Per-cell soft scoring ──
    cell_type_names = list(rules.keys())
    cell_scores = np.zeros((adata.n_obs, len(cell_type_names)), dtype=np.float32)

    for ct_idx, (_ct_name, rule) in enumerate(rules.items()):
        present_genes = [g for g in rule["present"] if g in rules_gene_col]
        absent_genes = [g for g in rule["absent"] if g in rules_gene_col]
        n_total = len(present_genes) + len(absent_genes)
        if n_total == 0:
            continue

        present_score = np.zeros(adata.n_obs, dtype=np.float32)
        for g in present_genes:
            present_score += cell_expressed[:, rules_gene_col[g]].astype(np.float32)

        absent_score = np.zeros(adata.n_obs, dtype=np.float32)
        for g in absent_genes:
            absent_score += (~cell_expressed[:, rules_gene_col[g]]).astype(np.float32)

        combined = (present_score + absent_score) / n_total

        # Gate: require at least one present gene expressed.
        if len(present_genes) > 0:
            has_any_present = present_score > 0
            combined[~has_any_present] = 0.0

        cell_scores[:, ct_idx] = combined

    best_ct_idx = np.argmax(cell_scores, axis=1)
    best_ct_score = np.max(cell_scores, axis=1)
    cell_labels = np.array([cell_type_names[i] for i in best_ct_idx])
    cell_labels[best_ct_score < 0.5] = "Unassigned"
    adata.obs["biccn_label"] = pd.Categorical(cell_labels)

    # ── Per-cluster Bayesian beta-binomial scoring ──
    clusters = _sorted_clusters(adata.obs[cluster_key])
    trinaries: dict[str, dict[str, float]] = {}
    rows = []

    for clust in clusters:
        mask = (adata.obs[cluster_key] == clust).to_numpy()
        n_cells = int(mask.sum())
        X_cluster = X_rules[mask]

        k_per_gene = (X_cluster > 0).sum(axis=0)
        clust_trinaries: dict[str, float] = {}
        for gi, g in enumerate(available_genes):
            clust_trinaries[g] = float(_trinarize(int(k_per_gene[gi]), n_cells, f=f))
        trinaries[str(clust)] = clust_trinaries

        best_label = "Unassigned"
        best_score = 0.0
        details_parts = []

        for ct_name, rule in rules.items():
            present_genes = [g for g in rule["present"] if g in rules_gene_col]
            absent_genes = [g for g in rule["absent"] if g in rules_gene_col]
            if len(present_genes) + len(absent_genes) == 0:
                continue

            p = 1.0
            for g in present_genes:
                t = clust_trinaries[g]
                p *= t
            for g in absent_genes:
                t = clust_trinaries[g]
                p *= 1.0 - t

            details_parts.append(f"{ct_name}={p:.3f}")

            if p > best_score:
                best_score = p
                best_label = ct_name

        if best_score < 0.5:
            best_label = "Unassigned"

        rows.append(
            {
                "cluster": str(clust),
                "biccn_label": best_label,
                "biccn_score": round(best_score, 3),
                "biccn_details": "; ".join(details_parts),
            }
        )

    df = pd.DataFrame(rows)
    adata.uns["biccn_evidence"] = {"trinaries": trinaries, "rules": df}


# ── score_evidence ─────────────────────────────────────────────────────────


def _celltypist_top(series: pd.Series) -> tuple[str, float]:
    """Most common categorical label + its fraction in the cluster."""
    if len(series) == 0:
        return ("N/A", 0.0)
    counts = series.astype(str).value_counts()
    if len(counts) == 0:
        return ("N/A", 0.0)
    top = counts.index[0]
    return (str(top), float(counts.iloc[0]) / float(len(series)))


def _celltypist_obs_columns(adata: ad.AnnData, profile: TargetCellProfile) -> dict[str, str]:
    """Return {label: obs_column} for enabled celltypist models present in obs.

    Prefers majority-vote columns, falls back to per-cell label columns.
    """
    from rarecell.core.annotate import _model_label

    found: dict[str, str] = {}
    for ref in profile.reference_labels.celltypist_models:
        if not ref.enabled:
            continue
        label = _model_label(ref.model)
        majority_col = f"celltypist_{label}_label_majority"
        single_col = f"celltypist_{label}_label"
        if majority_col in adata.obs.columns:
            found[label] = majority_col
        elif single_col in adata.obs.columns:
            found[label] = single_col
    return found


def score_evidence(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> pd.DataFrame:
    """Per-cluster multi-evidence summary table.

    Columns:
      - ``cluster``, ``n_cells``
      - ``score_{panel}_mean`` per positive panel in profile.positive_markers
      - ``pass_{panel}_frac`` per positive panel
      - ``negative_{panel}_frac`` per negative panel
      - ``is_contaminant_frac``
      - ``celltypist_{label}_top_label`` and ``celltypist_{label}_top_label_frac``
        per enabled celltypist model whose obs columns exist
      - ``biccn_top_label`` and ``biccn_top_prob`` if biccn_rules.enabled and
        ``adata.uns["biccn_evidence"]`` is populated
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"cluster_key '{cluster_key}' not found in adata.obs. Run clustering first."
        )

    clusters = _sorted_clusters(adata.obs[cluster_key])
    ct_columns = _celltypist_obs_columns(adata, profile)

    biccn_rules_df: pd.DataFrame | None = None
    if profile.biccn_rules.enabled:
        ev = adata.uns.get("biccn_evidence")
        if isinstance(ev, dict) and isinstance(ev.get("rules"), pd.DataFrame):
            biccn_rules_df = ev["rules"]

    rows = []
    for clust in clusters:
        mask = adata.obs[cluster_key] == clust
        n_cells = int(mask.sum())
        row: dict[str, object] = {"cluster": str(clust), "n_cells": n_cells}

        # Positive panels
        for name in profile.positive_markers:
            score_col = f"score_{name}"
            pass_col = f"pass_{name}"
            if score_col in adata.obs.columns:
                row[f"score_{name}_mean"] = round(float(adata.obs.loc[mask, score_col].mean()), 4)
            else:
                row[f"score_{name}_mean"] = np.nan
            if pass_col in adata.obs.columns:
                row[f"pass_{name}_frac"] = round(
                    float(adata.obs.loc[mask, pass_col].astype(bool).mean()), 4
                )
            else:
                row[f"pass_{name}_frac"] = np.nan

        # Negative panels
        for name in profile.negative_markers:
            pass_col = f"pass_{name}"
            if pass_col in adata.obs.columns:
                row[f"negative_{name}_frac"] = round(
                    float(adata.obs.loc[mask, pass_col].astype(bool).mean()), 4
                )
            else:
                row[f"negative_{name}_frac"] = np.nan

        # Contaminant flag
        if "is_contaminant" in adata.obs.columns:
            row["is_contaminant_frac"] = round(
                float(adata.obs.loc[mask, "is_contaminant"].astype(bool).mean()), 4
            )
        else:
            row["is_contaminant_frac"] = np.nan

        # CellTypist top label per enabled model
        for label, col in ct_columns.items():
            top_label, top_frac = _celltypist_top(adata.obs.loc[mask, col])
            row[f"celltypist_{label}_top_label"] = top_label
            row[f"celltypist_{label}_top_label_frac"] = round(top_frac, 4)

        # BICCN
        if biccn_rules_df is not None and len(biccn_rules_df) > 0:
            sub = biccn_rules_df[biccn_rules_df["cluster"].astype(str) == str(clust)]
            if len(sub) > 0:
                row["biccn_top_label"] = str(sub.iloc[0]["biccn_label"])
                row["biccn_top_prob"] = float(sub.iloc[0]["biccn_score"])
            else:
                row["biccn_top_label"] = "N/A"
                row["biccn_top_prob"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# ── render_consensus_table ────────────────────────────────────────────────


def render_consensus_table(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> tuple[pd.DataFrame, Figure]:
    """Run :func:`score_evidence` and render a color-coded matplotlib table."""
    import matplotlib.pyplot as plt

    df = score_evidence(adata, profile, cluster_key=cluster_key)

    text_cols = {"cluster"}
    # Any column ending with _top_label is also a text column
    for c in df.columns:
        if c.endswith("_top_label") or c == "biccn_top_label":
            text_cols.add(c)

    numeric_cols = [c for c in df.columns if c not in text_cols]

    norm_df = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    for col in numeric_cols:
        vals = pd.to_numeric(norm_df[col], errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
            norm_df[col] = (vals - vmin) / (vmax - vmin)
        else:
            norm_df[col] = 0.5

    n_rows = len(df)
    n_cols_table = len(df.columns)
    fig_width = max(12, n_cols_table * 1.5)
    fig_height = max(3, n_rows * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    display_values: list[list[str]] = []
    for _, row_data in df.iterrows():
        display_row: list[str] = []
        for col in df.columns:
            val = row_data[col]
            if col == "n_cells":
                display_row.append(f"{int(val):,}" if pd.notna(val) else "N/A")
            elif isinstance(val, float) and pd.notna(val):
                display_row.append(f"{val:.3f}" if abs(val) <= 10 else f"{val:.1f}")
            else:
                display_row.append(str(val) if pd.notna(val) else "N/A")
        display_values.append(display_row)

    table = ax.table(
        cellText=display_values, colLabels=df.columns.tolist(), cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    if n_cols_table > 0:
        table.auto_set_column_width(list(range(n_cols_table)))

    cmap = plt.cm.RdYlGn
    for i in range(n_rows):
        for j, col in enumerate(df.columns):
            cell = table[i + 1, j]
            if col in text_cols:
                cell.set_facecolor("white")
            else:
                val = norm_df.iloc[i][col] if col in norm_df.columns else np.nan
                if pd.isna(val):
                    cell.set_facecolor("#f0f0f0")
                else:
                    cell.set_facecolor(cmap(float(val)))
                    if float(val) < 0.3 or float(val) > 0.7:
                        cell.set_text_props(color="white")

    for j in range(n_cols_table):
        header = table[0, j]
        header.set_facecolor("#333333")
        header.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Consensus evidence table — {profile.profile_id}", fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return df, fig


# ── select_clusters ────────────────────────────────────────────────────────


def select_clusters(table: pd.DataFrame, recommendation: str) -> list[str]:
    """Return cluster IDs (as strings) where table['recommendation'] == ``recommendation``."""
    if "recommendation" not in table.columns:
        raise ValueError(
            "select_clusters requires a 'recommendation' column. "
            "Run the recommender first or set it manually."
        )
    mask = table["recommendation"] == recommendation
    return [str(c) for c in table.loc[mask, "cluster"].tolist()]


# ── Plot helpers ──────────────────────────────────────────────────────────
#
# These return matplotlib Figures (no PDF saves). Helpers that need
# als_utils-internal state (resolution scan dict, specific obs columns) are
# stubbed for now and will be ported alongside the Jupyter widget work —
# the public API is retained so callers compile.


def plot_stage_evidence(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Violin panels for positive-marker scores + a bar of contaminant fraction.

    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    panel_names = list(profile.positive_markers.keys())
    score_cols = [f"score_{n}" for n in panel_names if f"score_{n}" in adata.obs.columns]

    n_panels = len(score_cols) + (1 if "is_contaminant" in adata.obs.columns else 0)
    n_panels = max(n_panels, 1)

    clusters = _sorted_clusters(adata.obs[cluster_key])
    cluster_strs = [str(c) for c in clusters]

    fig_width = max(8, len(clusters) * 0.6 + 2)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width * n_panels, 4), squeeze=False)
    axes_flat = axes[0]

    ax_idx = 0
    for col in score_cols:
        ax = axes_flat[ax_idx]
        data = [adata.obs.loc[adata.obs[cluster_key] == c, col].to_numpy() for c in clusters]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(clusters) + 1))
        ax.set_xticklabels(cluster_strs, rotation=45, ha="right")
        ax.set_title(col)
        ax.set_xlabel(cluster_key)
        ax_idx += 1

    if "is_contaminant" in adata.obs.columns:
        ax = axes_flat[ax_idx]
        fracs = [
            float(adata.obs.loc[adata.obs[cluster_key] == c, "is_contaminant"].astype(bool).mean())
            for c in clusters
        ]
        ax.bar(cluster_strs, fracs, color="crimson", edgecolor="white")
        ax.set_ylabel("Contaminant fraction")
        ax.set_xlabel(cluster_key)
        ax.set_title("is_contaminant by cluster")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"{profile.profile_id} — stage evidence", fontsize=11)
    fig.tight_layout()
    return fig


def plot_biccn_composition(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Stacked-bar of per-cell BICCN labels per cluster.

    Returns a matplotlib Figure. If BICCN evidence has not been computed,
    returns an empty Figure with an explanatory title.
    """
    import matplotlib.pyplot as plt

    if "biccn_label" not in adata.obs.columns or not profile.biccn_rules.enabled:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No BICCN labels — run score_biccn_evidence",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    rules = _filtered_rules(profile)
    biccn_types = [*rules.keys(), "Unassigned"]
    clusters = _sorted_clusters(adata.obs[cluster_key])
    cluster_strs = [str(c) for c in clusters]

    frac_df = pd.DataFrame(0.0, index=cluster_strs, columns=biccn_types)
    for clust in clusters:
        mask = adata.obs[cluster_key] == clust
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        counts = adata.obs.loc[mask, "biccn_label"].astype(str).value_counts()
        for ct in biccn_types:
            frac_df.loc[str(clust), ct] = float(counts.get(ct, 0)) / n_cells

    fig_width = max(8, len(clusters) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    cmap = plt.get_cmap("tab20", max(len(biccn_types), 1))

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters))
    for i, ct in enumerate(biccn_types):
        vals = frac_df[ct].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=ct, color=cmap(i), edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cluster_strs, rotation=45, ha="right")
    ax.set_xlabel(cluster_key)
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{profile.profile_id} — BICCN composition")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_biccn_dotplot(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Dotplot of BICCN rule-gene trinarization scores per cluster.

    Dot size = detection rate, color = trinarization score (Bayesian posterior).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ev = adata.uns.get("biccn_evidence", {})
    trinaries = ev.get("trinaries", {}) if isinstance(ev, dict) else {}
    rules_df = ev.get("rules") if isinstance(ev, dict) else None

    if not trinaries or rules_df is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No BICCN trinarization — run score_biccn_evidence",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    rules = _filtered_rules(profile)
    data_source = adata.raw if adata.raw is not None else adata
    var_name_set = set(data_source.var_names)

    gene_order: list[tuple[str, str, str]] = []
    seen_genes: set[str] = set()
    for ct_name, rule in rules.items():
        for g in rule["present"]:
            if g in var_name_set and g not in seen_genes:
                gene_order.append((g, ct_name, "present"))
                seen_genes.add(g)
        for g in rule["absent"]:
            if g in var_name_set and g not in seen_genes:
                gene_order.append((g, ct_name, "absent"))
                seen_genes.add(g)

    if not gene_order:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No BICCN rule genes available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    gene_names = [g[0] for g in gene_order]
    var_names = list(data_source.var_names)
    gene_indices = [var_names.index(g) for g in gene_names]

    X_src = adata.raw.X if adata.raw is not None else adata.X
    X_rules = X_src[:, gene_indices]
    if issparse(X_rules):
        X_rules = X_rules.toarray()
    X_rules = np.asarray(X_rules)

    clusters = _sorted_clusters(adata.obs[cluster_key])
    n_genes = len(gene_names)
    n_clusters = len(clusters)
    det_rates = np.zeros((n_clusters, n_genes))

    for ci, clust in enumerate(clusters):
        mask = (adata.obs[cluster_key] == clust).to_numpy()
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        X_cl = X_rules[mask]
        det_rates[ci] = (X_cl > 0).sum(axis=0) / n_cells

    biccn_map = dict(zip(rules_df["cluster"].astype(str), rules_df["biccn_label"], strict=False))

    fig_height = max(4, n_clusters * 0.4 + 2)
    fig_width = max(8, n_genes * 0.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    max_dot, min_dot = 200, 10
    for ci in range(n_clusters):
        clust_str = str(clusters[ci])
        clust_trin = trinaries.get(clust_str, {})
        for gi in range(n_genes):
            dr = det_rates[ci, gi]
            t_score = clust_trin.get(gene_names[gi], 0.0)
            size = min_dot + dr * (max_dot - min_dot)
            if t_score >= 0.8:
                edge = "#2ca02c"
            elif t_score <= 0.2:
                edge = "#d62728"
            else:
                edge = "#bbbbbb"
            ax.scatter(
                gi,
                ci,
                s=size,
                c=[[t_score]],
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                edgecolors=edge,
                linewidths=1.5,
                zorder=3,
            )

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=9)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{c} ({biccn_map.get(str(c), '?')})" for c in clusters], fontsize=9)
    ax.set_xlim(-0.5, n_genes - 0.5)
    ax.set_ylim(-0.5, n_clusters - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="#2ca02c",
            markerfacecolor="#a8d5a2",
            markersize=10,
            markeredgewidth=2,
            label="Confidently present (P>=0.8)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="#d62728",
            markerfacecolor="#f4a4a0",
            markersize=10,
            markeredgewidth=2,
            label="Confidently absent (P<=0.2)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="#bbbbbb",
            markerfacecolor="#eeee99",
            markersize=10,
            markeredgewidth=2,
            label="Uncertain",
        ),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(f"{profile.profile_id} — BICCN trinarization")
    fig.tight_layout()
    return fig


def plot_biccn_probability_table(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Heatmap of BICCN joint probabilities per cluster x cell type."""
    import matplotlib.pyplot as plt

    ev = adata.uns.get("biccn_evidence", {})
    trinaries = ev.get("trinaries", {}) if isinstance(ev, dict) else {}
    rules_df = ev.get("rules") if isinstance(ev, dict) else None

    if not trinaries or rules_df is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No BICCN trinarization data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    rules = _filtered_rules(profile)
    clusters = sorted(trinaries.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    cell_types = list(rules.keys())

    prob_matrix = np.zeros((len(clusters), len(cell_types)))
    for i, clust in enumerate(clusters):
        clust_trin = trinaries[clust]
        for j, (_ct_name, rule) in enumerate(rules.items()):
            present_genes = [g for g in rule["present"] if g in clust_trin]
            absent_genes = [g for g in rule["absent"] if g in clust_trin]
            if not present_genes and not absent_genes:
                continue
            p = 1.0
            for g in present_genes:
                p *= clust_trin[g]
            for g in absent_genes:
                p *= 1.0 - clust_trin[g]
            prob_matrix[i, j] = p

    biccn_map = dict(zip(rules_df["cluster"].astype(str), rules_df["biccn_label"], strict=False))
    row_labels = [f"C{c} ({biccn_map.get(str(c), '?')})" for c in clusters]

    fig_height = max(4, len(clusters) * 0.4 + 1)
    fig_width = max(8, len(cell_types) * 0.7 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(prob_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(clusters)):
        for j in range(len(cell_types)):
            val = prob_matrix[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f"{profile.profile_id} — BICCN trinarization probabilities")
    fig.colorbar(im, ax=ax, label="Joint probability", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_resolution_scan(adata: ad.AnnData, cluster_key: str = "leiden") -> Figure:
    """Placeholder — will consume the resolution-scan results from clustering.

    Returns an empty placeholder Figure for now; the full implementation lands
    alongside the Jupyter widget work in a future release.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(
        0.5,
        0.5,
        "plot_resolution_scan — not yet implemented",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    return fig


def plot_resolution_umap_comparison(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
) -> Figure:
    """Placeholder — see plot_resolution_scan."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(
        0.5,
        0.5,
        "plot_resolution_umap_comparison — not yet implemented",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    return fig


def plot_annotation_confusion(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Confusion-matrix heatmap: clusters vs. ``cell_type_original`` obs column.

    Returns an empty Figure if ``cell_type_original`` is not present.
    """
    import matplotlib.pyplot as plt

    if "cell_type_original" not in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No 'cell_type_original' obs column",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    ct = pd.crosstab(adata.obs[cluster_key], adata.obs["cell_type_original"])
    try:
        ct = ct.loc[sorted(ct.index, key=lambda x: int(x))]
    except (ValueError, TypeError):
        ct = ct.sort_index()

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.loc[:, ct_pct.sum(axis=0) > 0]
    col_order = ct.sum(axis=0).sort_values(ascending=False).index
    col_order = [c for c in col_order if c in ct_pct.columns]
    ct_pct = ct_pct[col_order]

    n_rows, n_cols = len(ct_pct), len(ct_pct.columns)
    fig_w = max(8, n_cols * 0.6 + 3)
    fig_h = max(5, n_rows * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(ct_pct.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(ct_pct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ct_pct.index, fontsize=8)
    ax.set_xlabel("Original cell type")
    ax.set_ylabel(cluster_key)
    ax.set_title(f"{profile.profile_id} — clusters vs original annotations")

    for i in range(n_rows):
        for j in range(n_cols):
            pct = ct_pct.values[i, j]
            if pct >= 1:
                text_color = "white" if pct > 60 else "black"
                ax.text(j, i, f"{pct:.0f}%", ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of cluster")
    fig.tight_layout()
    return fig


def plot_all_markers_dotplot(
    adata: ad.AnnData,
    profile: TargetCellProfile,
    cluster_key: str = "leiden",
) -> Figure:
    """Dotplot of every gene from every positive + negative panel in the profile.

    Returns the matplotlib Figure produced by scanpy's dotplot.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    if adata.raw is not None:
        available = set(adata.raw.var_names)
        use_raw = True
    else:
        available = set(adata.var_names)
        use_raw = False

    seen: set[str] = set()
    marker_groups: dict[str, list[str]] = {}
    for name, panel in profile.positive_markers.items():
        genes = [g for g in panel.genes if g in available and g not in seen]
        if genes:
            marker_groups[f"+ {name}"] = genes
            seen.update(genes)
    for name, panel in profile.negative_markers.items():
        genes = [g for g in panel.genes if g in available and g not in seen]
        if genes:
            marker_groups[f"- {name}"] = genes
            seen.update(genes)

    if not marker_groups:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No profile marker genes present in adata",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    try:
        dp = sc.pl.dotplot(
            adata,
            var_names=marker_groups,
            groupby=cluster_key,
            use_raw=use_raw,
            show=False,
            title=f"{profile.profile_id} — all profile markers",
        )
        dp.add_totals()
        dp.make_figure()
        fig = plt.gcf()
    except Exception:
        flat_genes = [g for genes in marker_groups.values() for g in genes]
        sc.pl.dotplot(
            adata,
            var_names=flat_genes,
            groupby=cluster_key,
            use_raw=use_raw,
            show=False,
            title=f"{profile.profile_id} — all profile markers",
        )
        fig = plt.gcf()

    return fig
