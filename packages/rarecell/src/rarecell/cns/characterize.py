"""Characterize an isolated population against reference clusters + annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import pandas as pd

from rarecell.cns.format import load_annotations, load_model
from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.characterize")


@dataclass
class CharacterizationResult:
    per_cell_labels: pd.Series[str]
    summary: list[dict[str, Any]]


def characterize(
    isolated: ad.AnnData,
    bundle_dir: Path,
    *,
    level: Literal["cluster", "subcluster"],
    parent_node: str,
) -> CharacterizationResult:
    """Classify isolated cells into the parent_node's child clusters and summarize.

    ``level="subcluster"`` falls back to ``"cluster"`` (no subcluster models in v1).
    """
    import celltypist

    tax = TaxonomyTree.load(bundle_dir)
    if level == "subcluster":
        log.info("characterize.subcluster_fallback", parent_node=parent_node)
    # v1 has supercluster + cluster models only.
    artifact = tax._decision("cluster", parent_node)
    model = load_model(bundle_dir, artifact)
    pred = celltypist.annotate(isolated, model=model)
    labels = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    per_cell: pd.Series[str] = pd.Series(labels, index=isolated.obs_names, name="reference_cluster")

    ann = load_annotations(bundle_dir)
    n = len(per_cell)
    summary: list[dict[str, Any]] = []
    for cl, count in per_cell.value_counts().items():
        a = ann.get(str(cl), {})
        summary.append(
            {
                "cluster": str(cl),
                "n": int(count),
                "fraction": float(count) / n,
                "class": a.get("class", ""),
                "neurotransmitter": a.get("neurotransmitter", ""),
                "regions": a.get("regions", []),
                "markers": a.get("markers", []),
            }
        )
    summary.sort(key=lambda d: -d["fraction"])
    return CharacterizationResult(per_cell_labels=per_cell, summary=summary)
