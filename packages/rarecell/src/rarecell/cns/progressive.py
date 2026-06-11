"""Apply a CNS reference model chain to a query AnnData as a progressive gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.cns.format import DecisionArtifact, load_markers, load_model
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.progressive")


@dataclass
class ProgressiveResult:
    mask: np.ndarray  # bool over the input cells: True == on-path to target
    provenance: dict[str, Any]


def _as_log1p_cp10k(sub: ad.AnnData) -> ad.AnnData:
    """Return ``sub`` as log1p counts-per-10k, normalizing a copy only if needed.

    The reference CellTypist models require log1p-CP10K input and reject any
    matrix with ``max > log1p(1e4) ≈ 9.22``. The query reaching the gate may still
    be raw counts (the gate runs before the pipeline normalizes), so normalize a
    copy in that case; if it is already log-normalized, pass it through. The
    caller's matrix is never mutated, so the gate can narrow the raw input.
    """
    if sub.n_obs == 0 or float(sub.X.max()) <= 9.22:
        return sub
    out = sub.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def _predict_with_model(
    sub: ad.AnnData, bundle_dir: Path, artifact: DecisionArtifact
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_label, confidence) arrays over ``sub`` using the model."""
    import celltypist

    model = load_model(bundle_dir, artifact)
    pred = celltypist.annotate(_as_log1p_cp10k(sub), model=model)
    labels = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    conf: np.ndarray = pred.probability_matrix.max(axis=1).to_numpy()
    return labels, conf


def _predict_with_markers(
    sub: ad.AnnData, bundle_dir: Path, artifact: DecisionArtifact, keep_class: str
) -> tuple[np.ndarray, np.ndarray]:
    """Marker fallback: score the on-path class's panel; label keep_class where the
    score exceeds mean + 1 std, else "__other__". Confidence is 1.0/0.0.

    When the target class has no dedicated markers, fall back to an inverse
    strategy: collect all other-class markers and select cells whose other-class
    score falls below mean - 1 std (i.e. cells that do not express other-class genes).
    A class can have an empty panel when the build step's positive-coefficient marker
    extraction (scripts/build_cns_reference/train.py:extract_markers) found no gene
    with a positive coefficient for it (e.g. a low-expression baseline class).
    """
    panels = load_markers(bundle_dir, artifact.markers_file)
    genes = [g for g in panels.get(keep_class, []) if g in sub.var_names]
    if genes:
        sc.tl.score_genes(sub, gene_list=genes, score_name="_cns_fallback")
        s: np.ndarray = sub.obs["_cns_fallback"].to_numpy()
        passed = s > (s.mean() + s.std())
    else:
        # No direct markers for the target class; invert other-class panels.
        other_genes_all: list[str] = []
        for cls, panel in panels.items():
            if cls != keep_class:
                other_genes_all.extend(g for g in panel if g in sub.var_names)
        other_genes = list(dict.fromkeys(other_genes_all))  # deduplicate, preserve order
        if not other_genes:
            raise ReferenceBuildError(
                f"Marker fallback has no usable genes for class {keep_class!r} "
                f"or any other class in the panel"
            )
        log.info(
            "cns_gate.marker_fallback.inverse",
            keep_class=keep_class,
            n_other_genes=len(other_genes),
        )
        sc.tl.score_genes(sub, gene_list=other_genes, score_name="_cns_fallback")
        s = sub.obs["_cns_fallback"].to_numpy()
        # Keep cells that are LOW in other-class expression (below mean - 1 std).
        passed = s < (s.mean() - s.std())
    labels: np.ndarray = np.where(passed, keep_class, "__other__")
    conf: np.ndarray = passed.astype(float)
    return labels, conf


def apply_progressive(
    adata: ad.AnnData,
    bundle_dir: Path,
    path: list[tuple[DecisionArtifact, str]],
    *,
    min_confidence: float = 0.5,
    marker_fallback: bool = True,
) -> ProgressiveResult:
    """Apply the decision chain to ``adata`` (post-QC, log1p-CP10K).

    At each level the level's model predicts a label per surviving cell; cells
    predicted as the on-path class with confidence >= ``min_confidence`` are kept,
    and all other survivors are dropped (a conservative hard gate). If the level's
    model is unavailable (missing file / sha mismatch) and ``marker_fallback`` is
    True, the node's marker panel is scored instead. (Per-cell marker rescue for
    low-confidence model calls is intentionally not done in v1.)

    Writes ``obs["taxonomy_<level>"]`` for surviving cells and returns the
    keep-mask (cells on the path toward the target) plus provenance.
    """
    n = adata.n_obs
    mask = np.ones(n, dtype=bool)
    obs_names = adata.obs_names.to_numpy()
    levels: list[dict[str, Any]] = []

    for artifact, keep_class in path:
        sub = adata[mask].copy()
        method = "model"
        try:
            labels, conf = _predict_with_model(sub, bundle_dir, artifact)
        except ReferenceBuildError:
            if not marker_fallback:
                raise
            method = "marker_fallback"
            labels, conf = _predict_with_markers(sub, bundle_dir, artifact, keep_class)

        sub_keep = (labels == keep_class) & (conf >= min_confidence)

        col = f"taxonomy_{artifact.level}"
        if col not in adata.obs:
            adata.obs[col] = ""
        adata.obs.loc[sub.obs_names, col] = labels

        # Drop, from the global mask, the surviving cells that failed this level.
        drop_names = sub.obs_names.to_numpy()[~sub_keep]
        if len(drop_names):
            mask &= ~np.isin(obs_names, drop_names)

        levels.append(
            {
                "level": artifact.level,
                "keep_class": keep_class,
                "method": method,
                "n_in": int(sub.n_obs),
                "n_kept": int(sub_keep.sum()),
            }
        )
        log.info("cns_gate.level", **levels[-1])
        if int(sub_keep.sum()) == 0:
            log.warning("cns_gate.level_dropped_all", level=artifact.level, keep_class=keep_class)

    return ProgressiveResult(mask=mask, provenance={"levels": levels})
