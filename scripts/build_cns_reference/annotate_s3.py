"""Attach Siletti et al. (Science 2023) Table S3 per-cluster annotations.

Table S3 (vendored as a trimmed CSV) gives, per ``Cluster ID`` (== CELLxGENE
``cluster_id``), the paper's automated **class** call (e.g. ``ASTRO``, ``MGL``,
``NEUR`` — a coarse canonical cell class derived by matching each cluster's
enriched genes to canonical class marker panels, corroborated by Allen-MTG
label transfer), a **subtype** call, **neuropeptide** call, and the cluster's
**top enriched genes** (a curated marker panel). We use these to enrich the
bundle's per-cluster annotation sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

TABLE_S3_CSV = Path(__file__).parent / "data" / "siletti_table_s3.csv"


def load_table_s3(path: Path | None = None) -> pd.DataFrame:
    """Load the vendored, trimmed Table S3 CSV."""
    return pd.read_csv(path or TABLE_S3_CSV)


def _clean(v: object) -> str:
    return "" if pd.isna(v) else str(v).strip()


def _parse_regions(raw: object) -> list[str]:
    """Parse 'Cerebral cortex: 40%, Thalamus: 20%' -> ['Cerebral cortex', 'Thalamus']."""
    out: list[str] = []
    for part in _clean(raw).split(","):
        name = part.split(":")[0].strip()
        if name:
            out.append(name)
    return out


def build_s3_map(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build ``cluster_id (str) -> {class, subtype, neuropeptide, markers, regions}``."""
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        genes_raw = _clean(r.get("top_enriched_genes"))
        markers = [g.strip() for g in genes_raw.split(",") if g.strip()]
        out[str(int(r["cluster_id"]))] = {
            "class": _clean(r.get("class_auto")),
            "subtype": _clean(r.get("subtype_auto")),
            "neuropeptide": _clean(r.get("neuropeptide_auto")),
            "markers": markers,
            "regions": _parse_regions(r.get("top_regions", "")),
        }
    return out


def cluster_annotations(
    adata: ad.AnnData,
    s3map: dict[str, dict[str, Any]],
    *,
    cluster_key: str = "cluster_id",
    cluster_name_key: str = "cluster_name",
) -> dict[str, dict[str, Any]]:
    """Map each ``cluster_name`` in the atlas to its Table S3 annotation via the
    cluster's numeric ``cluster_id``."""
    name_key = cluster_name_key if cluster_name_key in adata.obs else cluster_key
    cols = [cluster_key] if name_key == cluster_key else [name_key, cluster_key]
    obs = adata.obs[cols].astype(str).drop_duplicates()
    out: dict[str, dict[str, Any]] = {}
    for _, r in obs.iterrows():
        ann = s3map.get(r[cluster_key])
        if ann is not None:
            out[r[name_key]] = ann  # name_key == cluster_key when no cluster_name
    return out
