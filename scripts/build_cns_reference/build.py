"""Orchestrate building a CNS reference bundle from a labeled, normalized atlas."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
from rarecell.cns import format as fmt
from rarecell.logging import get_logger

from scripts.build_cns_reference import labels as labelmod
from scripts.build_cns_reference import sample as samplemod
from scripts.build_cns_reference import train as trainmod

log = get_logger("rarecell.build_cns")


def _build_one(
    adata: ad.AnnData,
    *,
    bundle_dir: Path,
    level: fmt.DecisionLevel,
    parent: str | None,
    label_key: str,
    donor_key: str,
    cells_per_class: int,
    min_donors: int,
    top_genes: int,
    seed: int,
    check_expression: bool,
) -> fmt.DecisionArtifact | None:
    sub, stats = samplemod.balanced_subsample(
        adata,
        label_key,
        donor_key=donor_key,
        cells_per_class=cells_per_class,
        min_donors=min_donors,
        seed=seed,
    )
    included = [c for c, s in stats.items() if s.included]
    if len(included) < 2:
        log.info("build.skip_single_class", level=level, parent=parent, included=included)
        return None
    model, metrics, panels = trainmod.train_decision(
        sub,
        label_key,
        donor_key=donor_key,
        top_genes=top_genes,
        seed=seed,
        check_expression=check_expression,
    )
    log.info("build.trained", level=level, parent=parent, **metrics)
    return fmt.write_decision(
        bundle_dir,
        level=level,
        parent=parent,
        model=model,
        marker_panels=panels,
        per_class=stats,
        metrics=metrics,
    )


def build_bundle(
    atlas: ad.AnnData,
    *,
    out_dir: Path,
    biccn_release: str,
    cells_per_class: int = 5000,
    min_donors: int = 10,
    top_genes: int = 300,
    max_genes: int = 5000,
    seed: int = 0,
    check_expression: bool = True,
) -> fmt.BundleManifest:
    """Build supercluster (31-way) + per-supercluster cluster decisions into a bundle.

    `atlas` must be log1p-CP10K normalized with native obs columns present
    (resolved via labels.SUPERCLUSTER_CANDIDATES / CLUSTER_CANDIDATES /
    DONOR_CANDIDATES).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sc_key = labelmod.resolve_label_column(atlas.obs, labelmod.SUPERCLUSTER_CANDIDATES)
    cl_key = labelmod.resolve_label_column(atlas.obs, labelmod.CLUSTER_CANDIDATES)
    donor_key = labelmod.resolve_label_column(atlas.obs, labelmod.DONOR_CANDIDATES)

    # Bound the gene dimension before training. CellTypist's feature-selection
    # pass densifies the training matrix, so the full ~58k-gene atlas needs
    # ~36 GB for the supercluster model alone and OOMs even a High-RAM runtime.
    # Pre-select highly variable genes once (HVG selection is sparse-safe) so
    # every decision trains within a bounded set; CellTypist's own top_genes
    # selection then runs inside it.
    if max_genes and atlas.n_vars > max_genes:
        import scanpy as sc

        sc.pp.highly_variable_genes(atlas, n_top_genes=max_genes, flavor="seurat")
        atlas = atlas[:, atlas.var["highly_variable"].to_numpy()].copy()
        log.info("build.hvg_reduced", n_genes=int(atlas.n_vars))

    decisions: list[fmt.DecisionArtifact] = []

    # Supercluster (root) decision.
    sc_art = _build_one(
        atlas,
        bundle_dir=out_dir,
        level="supercluster",
        parent=None,
        label_key=sc_key,
        donor_key=donor_key,
        cells_per_class=cells_per_class,
        min_donors=min_donors,
        top_genes=top_genes,
        seed=seed,
        check_expression=check_expression,
    )
    if sc_art is not None:
        decisions.append(sc_art)

    # Per-supercluster cluster decisions + taxonomy tree.
    tree: dict[str, list[str]] = {}
    for sc_name, grp in atlas.obs.groupby(sc_key, observed=True):
        sc_str = str(sc_name)
        children = sorted(str(c) for c in grp[cl_key].unique())
        tree[sc_str] = children
        if len(children) < 2:
            continue
        sub = atlas[grp.index]
        cl_art = _build_one(
            sub,
            bundle_dir=out_dir,
            level="cluster",
            parent=sc_str,
            label_key=cl_key,
            donor_key=donor_key,
            cells_per_class=cells_per_class,
            min_donors=min_donors,
            top_genes=top_genes,
            seed=seed,
            check_expression=check_expression,
        )
        if cl_art is not None:
            decisions.append(cl_art)

    fmt.write_taxonomy(out_dir, tree)
    manifest = fmt.BundleManifest(
        biccn_release=biccn_release, created_with="rarecell-build", decisions=decisions
    )
    fmt.write_manifest(out_dir, manifest)
    return manifest
