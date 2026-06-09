from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from rarecell.errors import ReferenceBuildError

from scripts.build_cns_reference import labels


def _adata(cols: dict[str, list[str]]) -> ad.AnnData:
    n = len(next(iter(cols.values())))
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obs = pd.DataFrame(cols, index=[f"c{i}" for i in range(n)])
    return a


def test_resolves_first_matching_candidate() -> None:
    a = _adata({"supercluster_term": ["Astrocyte", "Oligodendrocyte"]})
    assert labels.resolve_label_column(a.obs, labels.SUPERCLUSTER_CANDIDATES) == "supercluster_term"


def test_raises_when_no_candidate_present() -> None:
    a = _adata({"unrelated": ["x", "y"]})
    with pytest.raises(ReferenceBuildError):
        labels.resolve_label_column(a.obs, labels.SUPERCLUSTER_CANDIDATES)


def test_donor_column_resolves() -> None:
    a = _adata({"donor_id": ["d1", "d2"]})
    assert labels.resolve_label_column(a.obs, labels.DONOR_CANDIDATES) == "donor_id"
