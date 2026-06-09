from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import stream
from scripts.build_cns_reference.discover import DiscoverDataset


def _write_file(path: Path, n_per_cluster: int, donor: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Two superclusters, each with one distinct cluster.
    sc_cl = [("Astrocyte", "astro-1"), ("Microglia", "micro-1")]
    n = n_per_cluster * len(sc_cl)
    X = rng.integers(0, 5, size=(n, 6)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"ENSG{i:05d}" for i in range(6)]
    a.var["feature_name"] = ["AQP4", "GFAP", "CSF1R", "P2RY12", "FOO", "BAR"]
    a.obs = pd.DataFrame(
        {
            "supercluster_term": [s for s, _ in sc_cl for _ in range(n_per_cluster)],
            "cluster_id": [c for _, c in sc_cl for _ in range(n_per_cluster)],
            "donor_id": [donor] * n,
        },
        index=[f"{donor}_{i}" for i in range(n)],
    )
    a.write_h5ad(path)


def _datasets(cache: Path, titles: list[str]) -> list[DiscoverDataset]:
    return [DiscoverDataset(title=t, cell_count=200, h5ad_url="unused") for t in titles]


def test_stream_caps_per_cluster_per_file_and_remaps_symbols(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    titles = ["Dissection: A", "Dissection: B"]
    for t, donor, seed in zip(titles, ["d0", "d1"], [1, 2], strict=True):
        dest = cache / (t.replace("/", "_").replace(" ", "_") + ".h5ad")
        _write_file(dest, n_per_cluster=100, donor=donor, seed=seed)

    atlas = stream.stream_balanced_atlas(
        _datasets(cache, titles), cache_dir=cache, per_file_cap=40, max_per_cluster=1000, seed=0
    )

    counts = atlas.obs["cluster_id"].value_counts().to_dict()
    # Each cluster: <=40 per file x 2 files = 80 (well under max_per_cluster).
    assert counts["astro-1"] == 80
    assert counts["micro-1"] == 80
    # Both donors represented.
    assert set(atlas.obs["donor_id"]) == {"d0", "d1"}
    # var_names remapped to symbols.
    assert "AQP4" in atlas.var_names and not any(v.startswith("ENSG") for v in atlas.var_names)
    # Normalized.
    assert float(atlas.X.max()) < 9.3


def test_stream_enforces_max_per_cluster_total(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    titles = ["Dissection: A", "Dissection: B", "Dissection: C"]
    for t, donor, seed in zip(titles, ["d0", "d1", "d2"], [1, 2, 3], strict=True):
        dest = cache / (t.replace("/", "_").replace(" ", "_") + ".h5ad")
        _write_file(dest, n_per_cluster=100, donor=donor, seed=seed)

    # per_file_cap=40 x 3 files = 120 available, but capped at 90 total per cluster.
    atlas = stream.stream_balanced_atlas(
        _datasets(cache, titles), cache_dir=cache, per_file_cap=40, max_per_cluster=90, seed=0
    )
    counts = atlas.obs["cluster_id"].value_counts().to_dict()
    assert counts["astro-1"] == 90
    assert counts["micro-1"] == 90
