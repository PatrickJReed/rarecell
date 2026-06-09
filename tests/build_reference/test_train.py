from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from rarecell.errors import ReferenceBuildError

from scripts.build_cns_reference import train


def _separable_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_genes = 40
    rows, labels, donors = [], [], []
    for cls, shift in [("A", 0.0), ("B", 3.0)]:
        for d in range(4):  # 4 donors per class
            x = rng.normal(loc=shift, size=(30, n_genes)).clip(min=0)
            rows.append(x)
            labels += [cls] * 30
            donors += [f"{cls}_d{d}"] * 30
    X = np.vstack(rows).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs = pd.DataFrame(
        {"label": labels, "donor_id": donors}, index=[f"c{i}" for i in range(X.shape[0])]
    )
    return a


def test_train_decision_raises_on_single_class() -> None:
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(20, 10)).clip(min=0).astype(np.float32))
    a.var_names = [f"g{i}" for i in range(10)]
    a.obs = pd.DataFrame(
        {"label": ["A"] * 20, "donor_id": [f"d{i % 3}" for i in range(20)]},
        index=[f"c{i}" for i in range(20)],
    )
    with pytest.raises(ReferenceBuildError):
        train.train_decision(a, "label", donor_key="donor_id", check_expression=False)


def test_train_decision_returns_model_metrics_and_markers() -> None:
    a = _separable_adata()
    model, metrics, panels = train.train_decision(
        a, "label", donor_key="donor_id", top_genes=20, seed=0, check_expression=False
    )
    assert metrics["heldout_accuracy"] >= 0.8
    assert set(panels.keys()) == {"A", "B"}
    # Panels contain only positive-coefficient genes; a class may have an empty
    # panel if no gene has a strictly positive coefficient for it (valid).
    assert all(isinstance(genes, list) for genes in panels.values())
    # At least one class should have markers in a well-separated dataset.
    assert any(len(genes) > 0 for genes in panels.values())
    # Model must be writable to disk (celltypist Model API).
    assert hasattr(model, "write")
