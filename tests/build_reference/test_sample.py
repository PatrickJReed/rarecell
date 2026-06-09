from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import sample


def _adata(labels: list[str], donors: list[str]) -> ad.AnnData:
    n = len(labels)
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    a.obs = pd.DataFrame({"label": labels, "donor_id": donors}, index=[f"c{i}" for i in range(n)])
    return a


def test_caps_each_class_and_reports_stats() -> None:
    # class A: 10 cells / 3 donors; class B: 4 cells / 2 donors
    labels = ["A"] * 10 + ["B"] * 4
    donors = (["d0", "d1", "d2"] * 4)[:10] + ["d3", "d4"] * 2
    a = _adata(labels, donors)
    sub, stats = sample.balanced_subsample(
        a, "label", donor_key="donor_id", cells_per_class=5, min_donors=2, seed=0
    )
    counts = sub.obs["label"].value_counts().to_dict()
    assert counts["A"] == 5  # capped
    assert counts["B"] == 4  # under cap, kept whole
    assert stats["A"].included and stats["A"].n_cells == 5 and stats["A"].n_donors == 3
    assert stats["B"].included and stats["B"].n_cells == 4


def test_drops_class_below_min_donors() -> None:
    labels = ["A"] * 6 + ["B"] * 6
    donors = ["d0", "d1", "d2"] * 2 + ["d3"] * 6  # B has only 1 donor
    a = _adata(labels, donors)
    sub, stats = sample.balanced_subsample(
        a, "label", donor_key="donor_id", cells_per_class=10, min_donors=2, seed=0
    )
    assert "B" not in set(sub.obs["label"])
    assert stats["B"].included is False and stats["B"].n_donors == 1
    assert stats["A"].included is True


def test_is_deterministic_for_seed() -> None:
    labels = ["A"] * 20
    donors = [f"d{i % 4}" for i in range(20)]
    a = _adata(labels, donors)
    s1, _ = sample.balanced_subsample(
        a, "label", donor_key="donor_id", cells_per_class=8, min_donors=2, seed=7
    )
    s2, _ = sample.balanced_subsample(
        a, "label", donor_key="donor_id", cells_per_class=8, min_donors=2, seed=7
    )
    assert list(s1.obs_names) == list(s2.obs_names)
