from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from rarecell.cns import format as fmt

from scripts.build_cns_reference import build


def _atlas(seed: int = 0) -> ad.AnnData:
    """Three superclusters; 'Astrocyte' has two clusters, others one each."""
    rng = np.random.default_rng(seed)
    n_genes = 40
    rows, sc, cl, donors = [], [], [], []

    def block(
        shift: float,
        supercluster: str,
        cluster: str,
        n_donors: int = 4,
        per: int = 25,
    ) -> None:
        for d in range(n_donors):
            x = rng.normal(loc=shift, size=(per, n_genes)).clip(min=0, max=9.0)
            rows.append(x)
            sc.extend([supercluster] * per)
            cl.extend([cluster] * per)
            donors.extend([f"{supercluster}_{cluster}_d{d}"] * per)

    block(0.0, "Astrocyte", "Astro-1")
    block(0.6, "Astrocyte", "Astro-2")
    block(2.0, "Oligodendrocyte", "Oligo-1")
    block(3.5, "Microglia", "Micro-1")

    X = np.vstack(rows).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs = pd.DataFrame(
        {"supercluster_term": sc, "cluster_id": cl, "donor_id": donors},
        index=[f"c{i}" for i in range(X.shape[0])],
    )
    return a


def test_build_bundle_writes_manifest_taxonomy_and_decisions(tmp_path: object) -> None:
    atlas = _atlas()
    build.build_bundle(
        atlas,
        out_dir=tmp_path,  # type: ignore[arg-type]
        biccn_release="WHB-test",
        cells_per_class=60,
        min_donors=2,
        top_genes=20,
        seed=0,
        check_expression=False,
    )
    manifest = fmt.load_manifest(tmp_path)  # type: ignore[arg-type]
    assert manifest.biccn_release == "WHB-test"

    # One supercluster decision over the 3 superclusters.
    sc_dec = [d for d in manifest.decisions if d.level == "supercluster"]
    assert len(sc_dec) == 1
    assert set(sc_dec[0].classes) == {"Astrocyte", "Oligodendrocyte", "Microglia"}

    # Cluster decision only for Astrocyte (the only multi-cluster supercluster).
    cl_dec = [d for d in manifest.decisions if d.level == "cluster"]
    assert {d.parent for d in cl_dec} == {"Astrocyte"}
    assert sorted(cl_dec[0].classes) == ["Astro-1", "Astro-2"]

    # Model files exist and hashes verify.
    for d in manifest.decisions:
        mp = tmp_path / d.model_file  # type: ignore[operator]
        assert mp.exists() and fmt.sha256_file(mp) == d.model_sha256

    # Taxonomy tree written: every supercluster present, children sorted.
    tree = fmt.load_taxonomy(tmp_path)  # type: ignore[arg-type]
    assert tree["Astrocyte"] == ["Astro-1", "Astro-2"]
    assert tree["Oligodendrocyte"] == ["Oligo-1"]
    assert tree["Microglia"] == ["Micro-1"]
