"""Attach Allen ABC Atlas biological annotations to the WHB cells.

The CELLxGENE h5ads label cells with a named ``supercluster_term`` but only
numeric ``cluster_id`` / ``subcluster_id``. The Allen ABC Atlas publishes a
small (~1.6 MB) public-S3 metadata table that maps each subcluster
(``cluster_alias`` == CELLxGENE ``subcluster_id``) to lineage-grounded labels
(e.g. cluster ``MGE_259``) and a neurotransmitter type (``GABA``, ``VGLUT1`` …).
We fetch it with the standard library (no special dependency) and use it to
relabel cluster nodes and annotate the bundle.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import anndata as ad
import pandas as pd
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

from scripts.build_cns_reference import labels as labelmod

log = get_logger("rarecell.build_cns.annotate_abc")

ABC_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
# WHB-taxonomy metadata release (small CSV; pinned for reproducibility).
MEMBERSHIP_URL = (
    f"{ABC_BASE}/metadata/WHB-taxonomy/20240330/cluster_to_cluster_annotation_membership.csv"
)

# obs columns this module writes.
CLUSTER_NAME_COL = "cluster_name"
SUBCLUSTER_NAME_COL = "subcluster_name"
NEUROTRANSMITTER_COL = "neurotransmitter"


def load_membership(cache_dir: Path) -> pd.DataFrame:
    """Download (and cache) the ABC WHB membership CSV; return it as a DataFrame."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "abc_whb_membership.csv"
    if not dest.exists():
        log.info("abc.download", url=MEMBERSHIP_URL)
        urllib.request.urlretrieve(MEMBERSHIP_URL, dest)
    return pd.read_csv(dest)


def build_annotation_map(membership: pd.DataFrame) -> dict[str, dict[str, str]]:
    """From the long membership table, build ``subcluster_id -> annotation``.

    Keyed by ``cluster_alias`` as a string (== CELLxGENE ``subcluster_id``), with
    the per-level names and neurotransmitter pivoted into one row per subcluster.
    """
    needed = {
        "cluster_alias",
        "cluster_annotation_term_set_name",
        "cluster_annotation_term_name",
    }
    missing = needed - set(membership.columns)
    if missing:
        raise ReferenceBuildError(f"ABC membership table missing columns: {sorted(missing)}")

    piv = membership.pivot_table(
        index="cluster_alias",
        columns="cluster_annotation_term_set_name",
        values="cluster_annotation_term_name",
        aggfunc="first",
    )

    def _clean(v: object) -> str:
        # Non-neuronal clusters have no neurotransmitter -> NaN; store "" not "nan".
        return "" if pd.isna(v) else str(v)

    out: dict[str, dict[str, str]] = {}
    for alias, row in piv.iterrows():
        out[str(alias)] = {
            level: _clean(row.get(level, ""))
            for level in ("supercluster", "cluster", "subcluster", "neurotransmitter")
        }
    return out


def annotate_atlas(
    adata: ad.AnnData, amap: dict[str, dict[str, str]], *, subcluster_key: str | None = None
) -> None:
    """Add ``cluster_name`` / ``subcluster_name`` / ``neurotransmitter`` obs columns
    by mapping each cell's subcluster id through the ABC annotation map (in place)."""
    if subcluster_key is None:
        subcluster_key = labelmod.resolve_label_column(adata.obs, labelmod.SUBCLUSTER_CANDIDATES)
    sub = adata.obs[subcluster_key].astype(str)
    adata.obs[CLUSTER_NAME_COL] = sub.map(lambda s: amap.get(s, {}).get("cluster", s)).astype(str)
    adata.obs[SUBCLUSTER_NAME_COL] = sub.map(lambda s: amap.get(s, {}).get("subcluster", s)).astype(
        str
    )
    adata.obs[NEUROTRANSMITTER_COL] = sub.map(
        lambda s: amap.get(s, {}).get("neurotransmitter", "")
    ).astype(str)


def cluster_neurotransmitters(adata: ad.AnnData) -> dict[str, str]:
    """Map each ``cluster_name`` to its (single) neurotransmitter for the bundle sidecar."""
    if CLUSTER_NAME_COL not in adata.obs or NEUROTRANSMITTER_COL not in adata.obs:
        return {}
    pairs = adata.obs[[CLUSTER_NAME_COL, NEUROTRANSMITTER_COL]].astype(str).drop_duplicates()
    return dict(zip(pairs[CLUSTER_NAME_COL], pairs[NEUROTRANSMITTER_COL], strict=True))
