"""Memory-bounded streaming subsample of the BICCN WHB atlas for bundle building.

The full atlas is ~3.4M cells across ~105 dissection files; loading it whole
would OOM a Colab runtime. We stream files one at a time and sample at the
**cluster** level (the finest taxonomy level we train), taking up to
``per_file_cap`` cells per ``cluster_id`` from each file and at most
``max_per_cluster`` cells per cluster across all files. Sampling per cluster
guarantees rare clusters are represented for the per-supercluster cluster
models; the supercluster model then trains on the union (clusters aggregate up
to superclusters). Ensembl IDs are remapped to gene symbols so the trained
models match symbol-keyed queries. The bounded subsample is handed to
``build.build_bundle``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from rarecell.logging import get_logger

from scripts.build_cns_reference import discover
from scripts.build_cns_reference import labels as labelmod

log = get_logger("rarecell.build_cns.stream")

# CELLxGENE WHB H5ADs use Ensembl IDs as var_names with gene symbols in this
# var column. Models must be trained on symbols to match symbol-keyed queries.
_SYMBOL_COL = "feature_name"


def _to_symbols(a: ad.AnnData) -> None:
    """Rename var_names from Ensembl IDs to gene symbols (in place), if available."""
    if _SYMBOL_COL in a.var.columns:
        a.var["ensembl_id"] = a.var_names
        a.var_names = a.var[_SYMBOL_COL].astype(str)
        a.var_names_make_unique()


def stream_balanced_atlas(
    datasets: list[discover.DiscoverDataset],
    *,
    cache_dir: Path,
    per_file_cap: int = 100,
    max_per_cluster: int = 1000,
    seed: int = 0,
    normalize: bool = True,
) -> ad.AnnData:
    """Stream the dissection files, sampling per ``cluster_id`` (<= ``per_file_cap``
    per file, <= ``max_per_cluster`` total), and return a normalized,
    symbol-keyed AnnData.

    Memory stays bounded to one file plus the accumulated subsample
    (<= ``max_per_cluster`` x n_clusters cells).
    """
    rng = np.random.default_rng(seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kept_per_cluster: dict[str, int] = defaultdict(int)
    parts: list[ad.AnnData] = []

    for i, ds in enumerate(datasets):
        dest = cache_dir / (ds.title.replace("/", "_").replace(" ", "_") + ".h5ad")
        if not dest.exists():
            log.info("stream.download", title=ds.title, cells=ds.cell_count)
            discover.download(ds.h5ad_url, dest)
        a = ad.read_h5ad(dest)
        cl_key = labelmod.resolve_label_column(a.obs, labelmod.CLUSTER_CANDIDATES)

        keep: set[str] = set()
        for cl, grp in a.obs.groupby(cl_key, observed=True):
            room = max_per_cluster - kept_per_cluster[str(cl)]
            if room <= 0:
                continue
            idx = grp.index.to_numpy()
            n_take = min(per_file_cap, room, len(idx))
            if len(idx) > n_take:
                idx = rng.choice(idx, size=n_take, replace=False)
            keep.update(idx.tolist())
            kept_per_cluster[str(cl)] += n_take
        ordered = [n for n in a.obs_names if n in keep]  # original order = determinism

        sub = a[ordered].copy()
        _to_symbols(sub)
        parts.append(sub)
        log.info(
            "stream.sampled",
            file=i + 1,
            of=len(datasets),
            kept=len(ordered),
            running=sum(p.n_obs for p in parts),
            clusters_seen=len(kept_per_cluster),
        )
        del a

    atlas = ad.concat(parts, join="inner")
    atlas.obs_names_make_unique()
    if normalize:
        sc.pp.normalize_total(atlas, target_sum=1e4)
        sc.pp.log1p(atlas)
    log.info("stream.done", n_obs=int(atlas.n_obs), n_vars=int(atlas.n_vars))
    return atlas
