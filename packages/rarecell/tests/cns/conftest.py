from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.build_cns_reference import build


@pytest.fixture
def minimal_profile_kwargs() -> dict:
    """Minimal valid kwargs for TargetCellProfile (no CNS-specific fields)."""
    return {
        "profile_id": "p1",
        "name": "n",
        "description": "d",
        "target_lineage": "astrocyte",
        "tissue": ["brain"],
        "expected_abundance": {"min_fraction": 0.01, "max_fraction": 0.2},
        "positive_markers": {"astro": {"genes": ["AQP4", "GFAP"], "threshold_z": 1.0}},
        "negative_markers": {},
        "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10.0},
    }


def _atlas(seed: int = 0) -> ad.AnnData:
    """3 superclusters; Astrocyte has 2 clusters. log1p-CP10K-ish, capped <9.22."""
    rng = np.random.default_rng(seed)
    n_genes = 40
    rows, sc, cl, donors = [], [], [], []

    def block(
        shift: float, supercluster: str, cluster: str, n_donors: int = 4, per: int = 25
    ) -> None:
        for d in range(n_donors):
            x = rng.normal(loc=shift, size=(per, n_genes)).clip(min=0, max=9.0)
            rows.append(x)
            sc.extend([supercluster] * per)
            cl.extend([cluster] * per)
            donors.extend([f"{supercluster}_{cluster}_d{d}"] * per)

    block(0.0, "Astrocyte", "Astro-1")
    block(0.6, "Astrocyte", "Astro-2")
    block(3.5, "Oligodendrocyte", "Oligo-1")
    block(2.0, "Microglia", "Micro-1")
    X = np.vstack(rows).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs = pd.DataFrame(
        {"supercluster_term": sc, "cluster_id": cl, "donor_id": donors},
        index=[f"c{i}" for i in range(X.shape[0])],
    )
    return a


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cns_bundle")
    build.build_bundle(
        _atlas(),
        out_dir=out,
        biccn_release="WHB-test",
        cells_per_class=60,
        min_donors=2,
        top_genes=20,
        seed=0,
        check_expression=False,
    )
    return out


@pytest.fixture
def atlas_factory() -> Callable[[int], ad.AnnData]:
    """Returns the synthetic-atlas generator (same distributions the bundle trained on)."""
    return _atlas
