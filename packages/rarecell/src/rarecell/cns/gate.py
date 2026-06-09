"""Profile-driven CNS class gate: resolve bundle -> path -> apply -> subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad

from rarecell.cns.bundle import ReferenceBundle
from rarecell.cns.progressive import apply_progressive
from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.errors import RareCellError, ReferenceBuildError
from rarecell.logging import get_logger
from rarecell.profile.schema import CNSTaxonomyConfig

log = get_logger("rarecell.cns.gate")


def apply_cns_class_gate(
    adata: ad.AnnData, cfg: CNSTaxonomyConfig, *, cache_dir: Path
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Return (narrowed_adata, provenance). No-op when disabled."""
    if not cfg.enabled:
        return adata, {"enabled": False}
    if not cfg.target_node or not cfg.reference_release:
        raise ReferenceBuildError("cns_taxonomy.enabled requires target_node and reference_release")

    try:
        bundle = ReferenceBundle.resolve(cfg.reference_release, cache_dir=Path(cache_dir))
        tax = TaxonomyTree.load(bundle.path)
        path = tax.path_to(cfg.target_node, cfg.target_level)
        result = apply_progressive(
            adata,
            bundle.path,
            path,
            min_confidence=cfg.min_confidence,
            marker_fallback=(cfg.on_missing == "marker_fallback"),
        )
    except RareCellError as e:  # ReferenceBuildError is a RareCellError subclass
        if cfg.on_missing == "skip":
            log.warning("cns_gate.skipped", error=str(e))
            return adata, {"enabled": True, "skipped": True, "error": str(e)}
        raise

    narrowed = adata[result.mask].copy()
    prov: dict[str, Any] = {
        "enabled": True,
        "target_node": cfg.target_node,
        "target_level": cfg.target_level,
        "n_in": int(adata.n_obs),
        "n_out": int(narrowed.n_obs),
        **result.provenance,
    }
    log.info("cns_gate.done", n_in=prov["n_in"], n_out=prov["n_out"])
    return narrowed, prov
