"""Train one multi-class CellTypist model per taxonomy decision."""

from __future__ import annotations

import importlib
import types
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import anndata as ad
import celltypist
import numpy as np
from celltypist.models import Model
from numpy.typing import NDArray
from rarecell.errors import ReferenceBuildError
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# sklearn ≥ 1.6 removed the `multi_class` kwarg from LogisticRegression.
# celltypist 1.x still passes it unconditionally.  We replace the
# LogisticRegression name inside the celltypist.train submodule with a
# subclass that accepts and stores the deprecated kwarg (satisfying sklearn's
# _get_param_names introspection) but does not forward it to super().__init__.
# ---------------------------------------------------------------------------
_ct_train_mod: types.ModuleType = importlib.import_module("celltypist.train")
_ORIG_LR_IN_CELLTYPIST: type = _ct_train_mod.LogisticRegression


class _CompatLR(LogisticRegression):  # type: ignore[misc]
    """LogisticRegression subclass that accepts the removed ``multi_class`` kwarg."""

    def __init__(
        self,
        penalty: Any = "l2",
        C: float = 1.0,
        l1_ratio: Any = None,
        dual: bool = False,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Any = None,
        random_state: Any = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Any = None,
        multi_class: Any = "deprecated",  # kept for celltypist compat; ignored by sklearn ≥ 1.6
    ) -> None:
        # Store so sklearn's get_params() / _get_param_names() introspection works.
        self.multi_class = multi_class
        super().__init__(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )


# NOTE: this swaps a module-global name and is NOT thread-safe. The build
# pipeline is single-threaded, so this is acceptable.
@contextmanager
def _celltypist_compat() -> Generator[None, None, None]:
    """Temporarily replace LogisticRegression in celltypist's train module."""
    _ct_train_mod.LogisticRegression = _CompatLR  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _ct_train_mod.LogisticRegression = _ORIG_LR_IN_CELLTYPIST  # type: ignore[attr-defined]


def _normalize_classifier(model: Model) -> None:
    """Recast a ``_CompatLR`` classifier back to plain ``LogisticRegression``.

    CellTypist pickles ``model.classifier``; under :func:`_celltypist_compat`
    that is our build-only ``_CompatLR`` subclass, whose module (``scripts...``)
    is not installed at runtime — so the saved bundle would fail to unpickle with
    ``ModuleNotFoundError: No module named 'scripts'``. The fitted state is
    identical to a plain ``LogisticRegression``; only the class reference matters.
    """
    clf = model.classifier
    if isinstance(clf, _CompatLR):
        clf.__class__ = LogisticRegression
        clf.__dict__.pop("multi_class", None)


def _heldout_donor_split(
    adata: ad.AnnData, donor_key: str, frac: float, seed: int
) -> NDArray[np.bool_]:
    """Boolean test mask choosing ~`frac` of donors as held-out."""
    donors = np.array(sorted(adata.obs[donor_key].unique()))
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(donors) * frac))
    test_donors = set(rng.choice(donors, size=min(n_test, len(donors) - 1), replace=False))
    result: NDArray[np.bool_] = adata.obs[donor_key].isin(test_donors).to_numpy()
    return result


def extract_markers(model: Model, top_n: int = 20) -> dict[str, list[str]]:
    """Top positive-coefficient genes per class from the trained LR model."""
    clf = model.classifier
    feats = np.asarray(model.features)
    coef = np.asarray(clf.coef_)
    classes = [str(c) for c in clf.classes_]
    # Binary LR gives a single coef row; expand to the (negative, positive) pair.
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    panels: dict[str, list[str]] = {}
    for i, cls in enumerate(classes):
        coefs_i = coef[i]
        order = np.argsort(coefs_i)[::-1]
        positive = [int(j) for j in order if coefs_i[j] > 0][:top_n]
        panels[cls] = feats[positive].tolist()
    return panels


def train_decision(
    adata: ad.AnnData,
    label_key: str,
    *,
    donor_key: str,
    top_genes: int = 300,
    C: float = 1.0,
    heldout_frac: float = 0.25,
    seed: int = 0,
    check_expression: bool = True,
) -> tuple[Model, dict[str, float], dict[str, list[str]]]:
    """Returns (celltypist Model, metrics, per-class marker panels)."""
    if adata.obs[label_key].nunique() < 2:
        raise ReferenceBuildError(
            f"train_decision needs >=2 classes in {label_key!r}, got "
            f"{adata.obs[label_key].nunique()}"
        )

    test_mask = _heldout_donor_split(adata, donor_key, heldout_frac, seed)
    train_ad = adata[~test_mask].copy()
    test_ad = adata[test_mask].copy()

    if train_ad.obs[label_key].nunique() < 2:
        raise ReferenceBuildError(
            f"After the held-out-donor split, the training set has <2 classes in "
            f"{label_key!r} (a class likely has too few donors). Provide more donors "
            f"per class or lower heldout_frac."
        )

    with _celltypist_compat():
        model: Model = celltypist.train(
            train_ad,
            labels=label_key,
            feature_selection=True,
            top_genes=top_genes,
            C=C,
            n_jobs=-1,
            check_expression=check_expression,
        )
    # Strip the build-only _CompatLR class so the saved bundle unpickles at
    # runtime (where the `scripts` package isn't installed).
    _normalize_classifier(model)

    pred = celltypist.annotate(test_ad, model=model)
    y_true = test_ad.obs[label_key].astype(str).to_numpy()
    y_pred = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    metrics = {"heldout_accuracy": float((y_true == y_pred).mean())}

    panels = extract_markers(model, top_n=min(20, top_genes))
    return model, metrics, panels
