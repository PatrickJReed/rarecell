# CNS Reference Build Pipeline — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline, one-time pipeline that distills the BICCN Human Brain Cell Atlas v1.0 into a small, versioned **CNS reference bundle** (per-level CellTypist models + marker panels + manifest) that `rarecell` ships and loads at runtime.

**Architecture:** A shared, shipped **bundle-format module** (`rarecell.cns.format`, pure pydantic/JSON, read+write) defines the on-disk contract. A separate, importable-but-not-shipped **build package** (`scripts/build_cns_reference/`) downloads BICCN H5ADs from the CELLxGENE Discover collection (Curation API), balanced-subsamples the atlas to ~equal cells per class, trains one 31-way supercluster CellTypist model plus a per-supercluster cluster model, extracts marker panels, and writes a bundle. Heavy deps (`httpx`) are lazy-imported and gated behind a `[build-reference]` dev group so they never touch the user runtime.

**Tech Stack:** Python 3.11+, celltypist (core dep), anndata/scanpy, pydantic v2, httpx (build-only), respx (test HTTP mocking), pytest. Repo is a uv workspace; lint via ruff (line-length 100).

**Spec:** `docs/superpowers/specs/2026-06-01-progressive-taxonomy-classification-design.md` §3–§4, §7.

**Scope note:** This plan delivers the **producer** (build pipeline + bundle format). The **consumer** (runtime `rarecell.cns` taxonomy tree, bundle loader, progressive applier, profile config, S2b pipeline integration, Colab demo) is **Plan 2 of 2** and consumes the bundle format created here.

---

## File Structure

Shipped (in the wheel, `packages/rarecell/src/rarecell/`):
- `cns/__init__.py` — new subpackage marker.
- `cns/format.py` — bundle on-disk contract: pydantic manifest/artifact models + read **and** write helpers + sha256. Used by the build package (write) and Plan 2's runtime (read).
- `errors.py` — MODIFIED: add `ReferenceBuildError`.

Not shipped (repo-root, dev tooling; importable via root `pythonpath=["."]`):
- `scripts/__init__.py`, `scripts/build_cns_reference/__init__.py` — package markers.
- `scripts/build_cns_reference/discover.py` — CELLxGENE Discover Curation API client (httpx, lazy).
- `scripts/build_cns_reference/labels.py` — native BICCN obs-column resolution.
- `scripts/build_cns_reference/sample.py` — balanced donor-aware subsampling.
- `scripts/build_cns_reference/train.py` — per-decision CellTypist training + held-out validation + marker extraction.
- `scripts/build_cns_reference/build.py` — orchestrator (supercluster + cluster) → writes bundle.
- `scripts/build_cns_reference/__main__.py` — `python -m scripts.build_cns_reference` CLI entry.

Tests (repo-root `tests/` is a pytest testpath):
- `tests/build_reference/__init__.py`
- `tests/build_reference/test_format.py`
- `tests/build_reference/test_discover.py`
- `tests/build_reference/test_labels.py`
- `tests/build_reference/test_sample.py`
- `tests/build_reference/test_train.py`
- `tests/build_reference/test_build.py`
- `tests/build_reference/test_integration.py` — network-gated (`@pytest.mark.integration`).

Packaging:
- `pyproject.toml` (root) — MODIFIED: add `build-reference` dependency group (`httpx`).

---

## Task 1: Bundle-format models + error

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/__init__.py`
- Create: `packages/rarecell/src/rarecell/cns/format.py`
- Modify: `packages/rarecell/src/rarecell/errors.py`
- Test: `tests/build_reference/__init__.py`, `tests/build_reference/test_format.py`

- [ ] **Step 1: Add the error class**

In `packages/rarecell/src/rarecell/errors.py`, add under the "Runtime errors" section:

```python
class ReferenceBuildError(RareCellError):
    """Raised when building or reading a CNS reference bundle fails."""
```

- [ ] **Step 2: Create the empty subpackage marker**

Create `packages/rarecell/src/rarecell/cns/__init__.py`:

```python
"""CNS reference-bundle format and (Plan 2) runtime application."""
```

- [ ] **Step 3: Write the failing test for manifest round-trip**

Create `tests/build_reference/__init__.py` (empty file), then `tests/build_reference/test_format.py`:

```python
import json
from pathlib import Path

from rarecell.cns import format as fmt


def test_manifest_round_trip(tmp_path: Path):
    manifest = fmt.BundleManifest(
        biccn_release="WHB-2023",
        created_with="rarecell-test",
        decisions=[
            fmt.DecisionArtifact(
                level="supercluster",
                parent=None,
                classes=["Astrocyte", "Oligodendrocyte"],
                model_file="nodes/supercluster/_decision.celltypist.pkl",
                model_sha256="0" * 64,
                markers_file="nodes/supercluster/_markers.json",
                metrics={"heldout_accuracy": 0.9},
                per_class={
                    "Astrocyte": fmt.ClassStat(n_cells=100, n_donors=12, included=True),
                    "Oligodendrocyte": fmt.ClassStat(n_cells=100, n_donors=11, included=True),
                },
            )
        ],
    )
    fmt.write_manifest(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()
    loaded = fmt.load_manifest(tmp_path)
    assert loaded == manifest
    assert json.loads((tmp_path / "manifest.json").read_text())["format_version"] == 1
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/build_reference/test_format.py::test_manifest_round_trip -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.format` (or `AttributeError`).

- [ ] **Step 5: Implement the format models + manifest I/O**

Create `packages/rarecell/src/rarecell/cns/format.py`:

```python
"""On-disk contract for a CNS reference bundle (read + write).

A bundle is a directory:

    <bundle>/
      manifest.json                              # BundleManifest
      taxonomy.json                              # {supercluster: [clusters...]}
      nodes/supercluster/_decision.celltypist.pkl
      nodes/supercluster/_markers.json
      nodes/cluster/<supercluster>/_decision.celltypist.pkl
      nodes/cluster/<supercluster>/_markers.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from rarecell.errors import ReferenceBuildError

FORMAT_VERSION = 1


class ClassStat(BaseModel):
    n_cells: int
    n_donors: int
    included: bool


class DecisionArtifact(BaseModel):
    level: str  # "supercluster" | "cluster"
    parent: str | None  # supercluster name for cluster-level, else None
    classes: list[str]
    model_file: str  # path relative to bundle root
    model_sha256: str
    markers_file: str  # path relative to bundle root
    metrics: dict[str, float]
    per_class: dict[str, ClassStat]


class BundleManifest(BaseModel):
    format_version: int = FORMAT_VERSION
    biccn_release: str
    created_with: str
    decisions: list[DecisionArtifact]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bundle_dir: Path, manifest: BundleManifest) -> None:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def load_manifest(bundle_dir: Path) -> BundleManifest:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        raise ReferenceBuildError(f"No manifest.json in bundle {bundle_dir}")
    return BundleManifest.model_validate_json(path.read_text())


def write_taxonomy(bundle_dir: Path, tree: dict[str, list[str]]) -> None:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "taxonomy.json").write_text(json.dumps(tree, indent=2, sort_keys=True))


def load_taxonomy(bundle_dir: Path) -> dict[str, list[str]]:
    path = Path(bundle_dir) / "taxonomy.json"
    if not path.exists():
        raise ReferenceBuildError(f"No taxonomy.json in bundle {bundle_dir}")
    return json.loads(path.read_text())


def write_markers(bundle_dir: Path, rel_path: str, panels: dict[str, list[str]]) -> None:
    out = Path(bundle_dir) / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panels, indent=2, sort_keys=True))


def load_markers(bundle_dir: Path, rel_path: str) -> dict[str, list[str]]:
    return json.loads((Path(bundle_dir) / rel_path).read_text())
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/build_reference/test_format.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/ packages/rarecell/src/rarecell/errors.py tests/build_reference/__init__.py tests/build_reference/test_format.py
git commit -m "feat(cns): add reference-bundle format models + manifest I/O"
```

---

## Task 2: Decision writer (model + markers + sha → DecisionArtifact)

**Files:**
- Modify: `packages/rarecell/src/rarecell/cns/format.py`
- Test: `tests/build_reference/test_format.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/build_reference/test_format.py`:

```python
class _FakeModel:
    """Stand-in for a celltypist Model: only needs .write(path)."""

    def __init__(self, payload: bytes = b"MODELBYTES"):
        self.payload = payload

    def write(self, path):
        from pathlib import Path as _P

        _P(path).write_bytes(self.payload)


def test_write_decision_emits_files_and_artifact(tmp_path):
    artifact = fmt.write_decision(
        tmp_path,
        level="supercluster",
        parent=None,
        model=_FakeModel(),
        marker_panels={"Astrocyte": ["AQP4", "GFAP"]},
        per_class={"Astrocyte": fmt.ClassStat(n_cells=50, n_donors=10, included=True)},
        metrics={"heldout_accuracy": 0.95},
    )
    model_path = tmp_path / artifact.model_file
    markers_path = tmp_path / artifact.markers_file
    assert model_path.exists() and markers_path.exists()
    assert artifact.level == "supercluster" and artifact.parent is None
    assert artifact.classes == ["Astrocyte"]
    assert artifact.model_sha256 == fmt.sha256_file(model_path)
    assert fmt.load_markers(tmp_path, artifact.markers_file) == {"Astrocyte": ["AQP4", "GFAP"]}
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_format.py::test_write_decision_emits_files_and_artifact -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'write_decision'`.

- [ ] **Step 3: Implement `write_decision`**

Append to `packages/rarecell/src/rarecell/cns/format.py`:

```python
def _decision_dir(level: str, parent: str | None) -> str:
    if level == "supercluster":
        return "nodes/supercluster"
    if level == "cluster":
        if not parent:
            raise ReferenceBuildError("cluster-level decision requires a parent supercluster")
        safe = parent.replace("/", "_").replace(" ", "_")
        return f"nodes/cluster/{safe}"
    raise ReferenceBuildError(f"Unknown decision level: {level!r}")


def write_decision(
    bundle_dir: Path,
    *,
    level: str,
    parent: str | None,
    model,
    marker_panels: dict[str, list[str]],
    per_class: dict[str, ClassStat],
    metrics: dict[str, float],
) -> DecisionArtifact:
    bundle_dir = Path(bundle_dir)
    rel_dir = _decision_dir(level, parent)
    model_rel = f"{rel_dir}/_decision.celltypist.pkl"
    markers_rel = f"{rel_dir}/_markers.json"

    model_path = bundle_dir / model_rel
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(model_path))
    write_markers(bundle_dir, markers_rel, marker_panels)

    return DecisionArtifact(
        level=level,
        parent=parent,
        # Only classes the model actually discriminates (excluded/low-donor
        # classes are dropped from training, but their stats are still recorded).
        classes=sorted(c for c, s in per_class.items() if s.included),
        model_file=model_rel,
        model_sha256=sha256_file(model_path),
        markers_file=markers_rel,
        metrics=metrics,
        per_class=per_class,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_format.py -v`
Expected: PASS (both format tests).

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/format.py tests/build_reference/test_format.py
git commit -m "feat(cns): add write_decision helper to bundle format"
```

---

## Task 3: BICCN obs-label column resolver

**Files:**
- Create: `scripts/__init__.py`, `scripts/build_cns_reference/__init__.py`, `scripts/build_cns_reference/labels.py`
- Test: `tests/build_reference/test_labels.py`

The exact native obs column names in the CELLxGENE H5ADs are not assumed — we resolve from a candidate list and fail loudly if none match.

- [ ] **Step 1: Create package markers**

Create `scripts/__init__.py` (empty) and `scripts/build_cns_reference/__init__.py`:

```python
"""Offline pipeline that distills BICCN into a CNS reference bundle. Dev tooling — not shipped."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/build_reference/test_labels.py`:

```python
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from rarecell.errors import ReferenceBuildError
from scripts.build_cns_reference import labels


def _adata(cols: dict) -> ad.AnnData:
    n = len(next(iter(cols.values())))
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obs = pd.DataFrame(cols, index=[f"c{i}" for i in range(n)])
    return a


def test_resolves_first_matching_candidate():
    a = _adata({"supercluster_term": ["Astrocyte", "Oligodendrocyte"]})
    assert labels.resolve_label_column(a.obs, labels.SUPERCLUSTER_CANDIDATES) == "supercluster_term"


def test_raises_when_no_candidate_present():
    a = _adata({"unrelated": ["x", "y"]})
    with pytest.raises(ReferenceBuildError):
        labels.resolve_label_column(a.obs, labels.SUPERCLUSTER_CANDIDATES)


def test_donor_column_resolves():
    a = _adata({"donor_id": ["d1", "d2"]})
    assert labels.resolve_label_column(a.obs, labels.DONOR_CANDIDATES) == "donor_id"
```

- [ ] **Step 3: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_cns_reference.labels`.

- [ ] **Step 4: Implement the resolver**

Create `scripts/build_cns_reference/labels.py`:

```python
"""Resolve native BICCN obs columns by trying known candidate names."""

from __future__ import annotations

import pandas as pd

from rarecell.errors import ReferenceBuildError

# Candidate column names in CELLxGENE Discover H5ADs for the Siletti WHB atlas.
# Ordered most-specific first; the first present wins.
SUPERCLUSTER_CANDIDATES = ["supercluster_term", "Supercluster", "supercluster"]
CLUSTER_CANDIDATES = ["cluster_id", "Cluster", "cluster"]
DONOR_CANDIDATES = ["donor_id", "donor", "DonorID"]


def resolve_label_column(obs: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in obs.columns:
            return c
    raise ReferenceBuildError(
        f"None of the candidate columns {candidates} are present in obs "
        f"(have: {list(obs.columns)[:20]})"
    )
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_labels.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/build_cns_reference/__init__.py scripts/build_cns_reference/labels.py tests/build_reference/test_labels.py
git commit -m "feat(build): add BICCN obs-label column resolver"
```

---

## Task 4: Balanced donor-aware subsampler

**Files:**
- Create: `scripts/build_cns_reference/sample.py`
- Test: `tests/build_reference/test_sample.py`

- [ ] **Step 1: Write the failing test**

Create `tests/build_reference/test_sample.py`:

```python
import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import sample


def _adata(labels, donors) -> ad.AnnData:
    n = len(labels)
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    a.obs = pd.DataFrame(
        {"label": labels, "donor_id": donors}, index=[f"c{i}" for i in range(n)]
    )
    return a


def test_caps_each_class_and_reports_stats():
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


def test_drops_class_below_min_donors():
    labels = ["A"] * 6 + ["B"] * 6
    donors = ["d0", "d1", "d2"] * 2 + ["d3"] * 6  # B has only 1 donor
    a = _adata(labels, donors)
    sub, stats = sample.balanced_subsample(
        a, "label", donor_key="donor_id", cells_per_class=10, min_donors=2, seed=0
    )
    assert "B" not in set(sub.obs["label"])
    assert stats["B"].included is False and stats["B"].n_donors == 1
    assert stats["A"].included is True


def test_is_deterministic_for_seed():
    labels = ["A"] * 20
    donors = [f"d{i % 4}" for i in range(20)]
    a = _adata(labels, donors)
    s1, _ = sample.balanced_subsample(a, "label", donor_key="donor_id", cells_per_class=8, min_donors=2, seed=7)
    s2, _ = sample.balanced_subsample(a, "label", donor_key="donor_id", cells_per_class=8, min_donors=2, seed=7)
    assert list(s1.obs_names) == list(s2.obs_names)
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_sample.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_cns_reference.sample`.

- [ ] **Step 3: Implement the subsampler**

Create `scripts/build_cns_reference/sample.py`:

```python
"""Balanced, donor-aware subsampling of a labeled AnnData for classifier training."""

from __future__ import annotations

import anndata as ad
import numpy as np

from rarecell.cns.format import ClassStat


def balanced_subsample(
    adata: ad.AnnData,
    label_key: str,
    *,
    donor_key: str,
    cells_per_class: int,
    min_donors: int,
    seed: int = 0,
) -> tuple[ad.AnnData, dict[str, ClassStat]]:
    """Down-sample to ~`cells_per_class` cells per label, dropping classes seen
    in fewer than `min_donors` donors. Returns (subset, per-class stats)."""
    rng = np.random.default_rng(seed)
    stats: dict[str, ClassStat] = {}
    keep: list[str] = []

    for cls, grp in adata.obs.groupby(label_key, observed=True):
        n_donors = int(grp[donor_key].nunique())
        if n_donors < min_donors:
            stats[str(cls)] = ClassStat(n_cells=0, n_donors=n_donors, included=False)
            continue
        names = grp.index.to_numpy()
        if len(names) > cells_per_class:
            names = rng.choice(names, size=cells_per_class, replace=False)
        keep.extend(names.tolist())
        stats[str(cls)] = ClassStat(n_cells=int(len(names)), n_donors=n_donors, included=True)

    # Preserve original ordering for determinism independent of dict order.
    keep_set = set(keep)
    ordered = [n for n in adata.obs_names if n in keep_set]
    return adata[ordered].copy(), stats
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_sample.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cns_reference/sample.py tests/build_reference/test_sample.py
git commit -m "feat(build): add balanced donor-aware subsampler"
```

---

## Task 5: Trainer — CellTypist train + held-out validation + marker extraction

**Files:**
- Create: `scripts/build_cns_reference/train.py`
- Test: `tests/build_reference/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/build_reference/test_train.py`. The synthetic data is two linearly-separable classes across multiple donors so the classifier is learnable and held-out accuracy is high.

```python
import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import train


def _separable_adata(seed=0) -> ad.AnnData:
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


def test_train_decision_returns_model_metrics_and_markers():
    a = _separable_adata()
    model, metrics, panels = train.train_decision(
        a, "label", donor_key="donor_id", top_genes=20, seed=0, check_expression=False
    )
    assert metrics["heldout_accuracy"] >= 0.8
    assert set(panels.keys()) == {"A", "B"}
    assert all(len(genes) > 0 for genes in panels.values())
    # Model must be writable to disk (celltypist Model API).
    assert hasattr(model, "write")
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_cns_reference.train`.

- [ ] **Step 3: Implement the trainer**

Create `scripts/build_cns_reference/train.py`:

```python
"""Train one multi-class CellTypist model per taxonomy decision."""

from __future__ import annotations

import anndata as ad
import celltypist
import numpy as np

from rarecell.errors import ReferenceBuildError


def _heldout_donor_split(
    adata: ad.AnnData, donor_key: str, frac: float, seed: int
) -> np.ndarray:
    """Boolean test mask choosing ~`frac` of donors as held-out."""
    donors = np.array(sorted(adata.obs[donor_key].unique()))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(donors) * frac)))
    test_donors = set(rng.choice(donors, size=min(n_test, len(donors) - 1), replace=False))
    return adata.obs[donor_key].isin(test_donors).to_numpy()


def extract_markers(model, top_n: int = 20) -> dict[str, list[str]]:
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
        order = np.argsort(coef[i])[::-1][:top_n]
        panels[cls] = feats[order].tolist()
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
) -> tuple[object, dict[str, float], dict[str, list[str]]]:
    """Returns (celltypist Model, metrics, per-class marker panels)."""
    if adata.obs[label_key].nunique() < 2:
        raise ReferenceBuildError(
            f"train_decision needs >=2 classes in {label_key!r}, got "
            f"{adata.obs[label_key].nunique()}"
        )

    test_mask = _heldout_donor_split(adata, donor_key, heldout_frac, seed)
    train_ad = adata[~test_mask].copy()
    test_ad = adata[test_mask].copy()

    model = celltypist.train(
        train_ad,
        labels=label_key,
        feature_selection=True,
        top_genes=top_genes,
        C=C,
        n_jobs=-1,
        check_expression=check_expression,
    )

    pred = celltypist.annotate(test_ad, model=model)
    y_true = test_ad.obs[label_key].astype(str).to_numpy()
    y_pred = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    metrics = {"heldout_accuracy": float((y_true == y_pred).mean())}

    panels = extract_markers(model, top_n=min(20, top_genes))
    return model, metrics, panels
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_train.py -v`
Expected: PASS (held-out accuracy ≥ 0.8 on separable data).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cns_reference/train.py tests/build_reference/test_train.py
git commit -m "feat(build): add CellTypist trainer with held-out validation + markers"
```

---

## Task 6: Build orchestrator (supercluster + per-supercluster cluster) → bundle

**Files:**
- Create: `scripts/build_cns_reference/build.py`
- Test: `tests/build_reference/test_build.py`

The orchestrator takes **already-loaded, normalized** AnnData objects (dependency injection), so it is fully testable without network. The CLI (Task 7) wires download → normalize → orchestrator.

- [ ] **Step 1: Write the failing test**

Create `tests/build_reference/test_build.py`:

```python
import anndata as ad
import numpy as np
import pandas as pd

from rarecell.cns import format as fmt
from scripts.build_cns_reference import build


def _atlas(seed=0) -> ad.AnnData:
    """Three superclusters; 'Astrocyte' has two clusters, others one each."""
    rng = np.random.default_rng(seed)
    n_genes = 40
    rows, sc, cl, donors = [], [], [], []

    def block(shift, supercluster, cluster, n_donors=4, per=25):
        for d in range(n_donors):
            x = rng.normal(loc=shift, size=(per, n_genes)).clip(min=0)
            rows.append(x)
            sc.extend([supercluster] * per)
            cl.extend([cluster] * per)
            donors.extend([f"{supercluster}_{cluster}_d{d}"] * per)

    block(0.0, "Astrocyte", "Astro-1")
    block(0.6, "Astrocyte", "Astro-2")
    block(4.0, "Oligodendrocyte", "Oligo-1")
    block(8.0, "Microglia", "Micro-1")

    X = np.vstack(rows).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs = pd.DataFrame(
        {"supercluster_term": sc, "cluster_id": cl, "donor_id": donors},
        index=[f"c{i}" for i in range(X.shape[0])],
    )
    return a


def test_build_bundle_writes_manifest_taxonomy_and_decisions(tmp_path):
    atlas = _atlas()
    build.build_bundle(
        atlas,
        out_dir=tmp_path,
        biccn_release="WHB-test",
        cells_per_class=60,
        min_donors=2,
        top_genes=20,
        seed=0,
        check_expression=False,
    )
    manifest = fmt.load_manifest(tmp_path)
    assert manifest.biccn_release == "WHB-test"

    # One supercluster decision over the 3 superclusters.
    sc_dec = [d for d in manifest.decisions if d.level == "supercluster"]
    assert len(sc_dec) == 1
    assert set(sc_dec[0].classes) == {"Astrocyte", "Oligodendrocyte", "Microglia"}

    # Cluster decision only for Astrocyte (the only multi-cluster supercluster).
    cl_dec = [d for d in manifest.decisions if d.level == "cluster"]
    assert {d.parent for d in cl_dec} == {"Astrocyte"}
    assert set(cl_dec[0].classes) == {"Astro-1", "Astro-2"}

    # Model files exist and hashes verify.
    for d in manifest.decisions:
        mp = tmp_path / d.model_file
        assert mp.exists() and fmt.sha256_file(mp) == d.model_sha256

    # Taxonomy tree written.
    tree = fmt.load_taxonomy(tmp_path)
    assert set(tree["Astrocyte"]) == {"Astro-1", "Astro-2"}
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_cns_reference.build`.

- [ ] **Step 3: Implement the orchestrator**

Create `scripts/build_cns_reference/build.py`:

```python
"""Orchestrate building a CNS reference bundle from a labeled, normalized atlas."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from rarecell.cns import format as fmt
from rarecell.logging import get_logger
from scripts.build_cns_reference import labels as labelmod
from scripts.build_cns_reference import sample as samplemod
from scripts.build_cns_reference import train as trainmod

log = get_logger("rarecell.build_cns")


def _build_one(
    adata: ad.AnnData,
    *,
    bundle_dir: Path,
    level: str,
    parent: str | None,
    label_key: str,
    donor_key: str,
    cells_per_class: int,
    min_donors: int,
    top_genes: int,
    seed: int,
    check_expression: bool,
) -> fmt.DecisionArtifact | None:
    sub, stats = samplemod.balanced_subsample(
        adata, label_key, donor_key=donor_key,
        cells_per_class=cells_per_class, min_donors=min_donors, seed=seed,
    )
    included = [c for c, s in stats.items() if s.included]
    if len(included) < 2:
        log.info("build.skip_single_class", level=level, parent=parent, included=included)
        return None
    model, metrics, panels = trainmod.train_decision(
        sub, label_key, donor_key=donor_key, top_genes=top_genes,
        seed=seed, check_expression=check_expression,
    )
    log.info("build.trained", level=level, parent=parent, **metrics)
    return fmt.write_decision(
        bundle_dir,
        level=level, parent=parent, model=model,
        marker_panels=panels, per_class=stats, metrics=metrics,
    )


def build_bundle(
    atlas: ad.AnnData,
    *,
    out_dir: Path,
    biccn_release: str,
    cells_per_class: int = 5000,
    min_donors: int = 10,
    top_genes: int = 300,
    seed: int = 0,
    check_expression: bool = True,
) -> fmt.BundleManifest:
    """Build supercluster (31-way) + per-supercluster cluster decisions into a bundle.

    `atlas` must be log1p-CP10K normalized with native obs columns present
    (resolved via labels.SUPERCLUSTER_CANDIDATES / CLUSTER_CANDIDATES /
    DONOR_CANDIDATES).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sc_key = labelmod.resolve_label_column(atlas.obs, labelmod.SUPERCLUSTER_CANDIDATES)
    cl_key = labelmod.resolve_label_column(atlas.obs, labelmod.CLUSTER_CANDIDATES)
    donor_key = labelmod.resolve_label_column(atlas.obs, labelmod.DONOR_CANDIDATES)

    decisions: list[fmt.DecisionArtifact] = []

    # Supercluster (root) decision.
    sc_art = _build_one(
        atlas, bundle_dir=out_dir, level="supercluster", parent=None,
        label_key=sc_key, donor_key=donor_key,
        cells_per_class=cells_per_class, min_donors=min_donors, top_genes=top_genes,
        seed=seed, check_expression=check_expression,
    )
    if sc_art is not None:
        decisions.append(sc_art)

    # Per-supercluster cluster decisions + taxonomy tree.
    tree: dict[str, list[str]] = {}
    for sc_name, grp in atlas.obs.groupby(sc_key, observed=True):
        children = sorted(map(str, grp[cl_key].unique()))
        tree[str(sc_name)] = children
        if len(children) < 2:
            continue
        sub = atlas[grp.index].copy()
        cl_art = _build_one(
            sub, bundle_dir=out_dir, level="cluster", parent=str(sc_name),
            label_key=cl_key, donor_key=donor_key,
            cells_per_class=cells_per_class, min_donors=min_donors, top_genes=top_genes,
            seed=seed, check_expression=check_expression,
        )
        if cl_art is not None:
            decisions.append(cl_art)

    fmt.write_taxonomy(out_dir, tree)
    manifest = fmt.BundleManifest(
        biccn_release=biccn_release, created_with="rarecell-build", decisions=decisions
    )
    fmt.write_manifest(out_dir, manifest)
    return manifest
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cns_reference/build.py tests/build_reference/test_build.py
git commit -m "feat(build): orchestrate supercluster + cluster bundle build"
```

---

## Task 7: Discover Curation API client + CLI entry

**Files:**
- Create: `scripts/build_cns_reference/discover.py`, `scripts/build_cns_reference/__main__.py`
- Test: `tests/build_reference/test_discover.py`
- Modify: `pyproject.toml` (root) — add `build-reference` dependency group.

- [ ] **Step 1: Add the build-reference dependency group**

In root `pyproject.toml`, under `[dependency-groups]`, add:

```toml
build-reference = [
    "httpx>=0.27",
]
```

- [ ] **Step 2: Sync the group**

Run: `uv sync --group build-reference`
Expected: `httpx` installed.

- [ ] **Step 3: Write the failing test (respx-mocked Curation API)**

Create `tests/build_reference/test_discover.py`:

```python
import httpx
import respx

from scripts.build_cns_reference import discover

COLLECTION_ID = "283d65eb-dd53-496d-adb7-7570c7caa443"


@respx.mock
def test_list_h5ad_datasets_parses_assets():
    payload = {
        "datasets": [
            {
                "title": "Supercluster: Astrocyte",
                "cell_count": 155025,
                "assets": [
                    {"filetype": "H5AD", "url": "https://x/astro.h5ad", "filesize": 100},
                    {"filetype": "RDS", "url": "https://x/astro.rds", "filesize": 100},
                ],
            },
            {
                "title": "No H5AD dataset",
                "cell_count": 10,
                "assets": [{"filetype": "RDS", "url": "https://x/only.rds", "filesize": 1}],
            },
        ]
    }
    respx.get(f"{discover.CURATION_BASE}/collections/{COLLECTION_ID}").mock(
        return_value=httpx.Response(200, json=payload)
    )
    datasets = discover.list_h5ad_datasets(COLLECTION_ID)
    assert len(datasets) == 1
    assert datasets[0].title == "Supercluster: Astrocyte"
    assert datasets[0].h5ad_url == "https://x/astro.h5ad"
    assert datasets[0].cell_count == 155025


@respx.mock
def test_download_streams_to_file(tmp_path):
    respx.get("https://x/astro.h5ad").mock(
        return_value=httpx.Response(200, content=b"H5ADBYTES")
    )
    dest = tmp_path / "astro.h5ad"
    discover.download("https://x/astro.h5ad", dest)
    assert dest.read_bytes() == b"H5ADBYTES"
```

- [ ] **Step 4: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_discover.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_cns_reference.discover`.

- [ ] **Step 5: Implement the Discover client (httpx lazy-imported)**

Create `scripts/build_cns_reference/discover.py`:

```python
"""CELLxGENE Discover Curation API client (build-only; httpx is lazy-imported)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rarecell.errors import ReferenceBuildError

CURATION_BASE = "https://api.cellxgene.cziscience.com/curation/v1"


def _httpx():
    try:
        import httpx
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ReferenceBuildError(
            "httpx is required for the build pipeline. Install with: "
            "uv sync --group build-reference"
        ) from e
    return httpx


@dataclass
class DiscoverDataset:
    title: str
    cell_count: int
    h5ad_url: str


def list_h5ad_datasets(collection_id: str) -> list[DiscoverDataset]:
    httpx = _httpx()
    resp = httpx.get(f"{CURATION_BASE}/collections/{collection_id}", timeout=60)
    resp.raise_for_status()
    out: list[DiscoverDataset] = []
    for ds in resp.json().get("datasets", []):
        h5ad = next(
            (a["url"] for a in ds.get("assets", []) if a.get("filetype") == "H5AD"), None
        )
        if h5ad is None:
            continue
        out.append(
            DiscoverDataset(
                title=ds.get("title", ""),
                cell_count=int(ds.get("cell_count", 0)),
                h5ad_url=h5ad,
            )
        )
    return out


def download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    httpx = _httpx()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=None, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for block in resp.iter_bytes(chunk):
                fh.write(block)
```

- [ ] **Step 6: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_discover.py -v`
Expected: PASS.

- [ ] **Step 7: Implement the CLI entry**

Create `scripts/build_cns_reference/__main__.py`:

```python
"""CLI: python -m scripts.build_cns_reference --out <dir> [--collection <id>] ...

Downloads BICCN WHB H5ADs, normalizes, and builds a CNS reference bundle.
Run from the repo root with the build-reference group synced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc

from rarecell.logging import get_logger
from scripts.build_cns_reference import build, discover

log = get_logger("rarecell.build_cns.cli")

WHB_COLLECTION = "283d65eb-dd53-496d-adb7-7570c7caa443"


def _load_and_normalize(paths: list[Path]) -> ad.AnnData:
    parts = []
    for p in paths:
        a = ad.read_h5ad(p)
        parts.append(a)
    atlas = ad.concat(parts, join="inner") if len(parts) > 1 else parts[0]
    sc.pp.normalize_total(atlas, target_sum=1e4)
    sc.pp.log1p(atlas)
    return atlas


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the CNS reference bundle from BICCN WHB.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collection", default=WHB_COLLECTION)
    ap.add_argument("--cache-dir", type=Path, default=Path("./biccn_cache"))
    ap.add_argument("--biccn-release", default="WHB-2023")
    ap.add_argument("--cells-per-class", type=int, default=5000)
    ap.add_argument("--min-donors", type=int, default=10)
    ap.add_argument("--top-genes", type=int, default=300)
    args = ap.parse_args(argv)

    datasets = discover.list_h5ad_datasets(args.collection)
    # The collection ships TWO orthogonal slicings of the SAME nuclei: by brain
    # region/dissection AND by supercluster. The region files form a
    # non-overlapping partition covering every cell exactly once (with native
    # supercluster + cluster labels), so we use those and EXCLUDE the
    # "Supercluster: ..." re-slices to avoid double-counting cells.
    datasets = [d for d in datasets if not d.title.startswith("Supercluster:")]
    log.info("cli.datasets", n=len(datasets))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ds in datasets:
        dest = args.cache_dir / (ds.title.replace("/", "_").replace(" ", "_") + ".h5ad")
        if not dest.exists():
            log.info("cli.download", title=ds.title, cells=ds.cell_count)
            discover.download(ds.h5ad_url, dest)
        paths.append(dest)

    atlas = _load_and_normalize(paths)
    build.build_bundle(
        atlas,
        out_dir=args.out,
        biccn_release=args.biccn_release,
        cells_per_class=args.cells_per_class,
        min_donors=args.min_donors,
        top_genes=args.top_genes,
    )
    log.info("cli.done", out=str(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Smoke-test the CLI parses (no network)**

Run: `uv run python -m scripts.build_cns_reference --help`
Expected: argparse help text prints; exit 0.

- [ ] **Step 9: Commit**

```bash
git add scripts/build_cns_reference/discover.py scripts/build_cns_reference/__main__.py tests/build_reference/test_discover.py pyproject.toml
git commit -m "feat(build): add Discover Curation API client + build CLI"
```

---

## Task 8: Network-gated integration test (real small bundle)

**Files:**
- Test: `tests/build_reference/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/build_reference/test_integration.py`. It is skipped by default (the suite runs without `-m integration`); it downloads the two smallest real supercluster files and asserts a bundle builds.

```python
import pytest

from scripts.build_cns_reference import discover

WHB_COLLECTION = "283d65eb-dd53-496d-adb7-7570c7caa443"


@pytest.mark.integration
def test_discover_lists_real_whb_h5ads():
    datasets = discover.list_h5ad_datasets(WHB_COLLECTION)
    assert len(datasets) >= 20
    titles = [d.title for d in datasets]
    assert any("Astrocyte" in t for t in titles)
    assert all(d.h5ad_url.endswith(".h5ad") for d in datasets)
```

- [ ] **Step 2: Verify it is skipped in the default run**

Run: `uv run pytest tests/build_reference/test_integration.py -v`
Expected: 1 deselected/skipped (no `-m integration`), 0 failures.

- [ ] **Step 3: (Manual, network) verify it passes when selected**

Run: `uv run pytest tests/build_reference/test_integration.py -v -m integration`
Expected: PASS (requires network). Document the result; do not block the plan on flaky network.

- [ ] **Step 4: Commit**

```bash
git add tests/build_reference/test_integration.py
git commit -m "test(build): add network-gated Discover integration test"
```

---

## Task 9: Document the build + retrain path

**Files:**
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: Add a "Building the CNS reference bundle" section**

Append to `CONTRIBUTING.md`:

```markdown
## Building the CNS reference bundle (maintainers / power users)

The CNS reference bundle (per-level CellTypist models + marker panels) is built
**offline, once per BICCN release**, and published as a GitHub release asset.
End users never run this — they fetch the small bundle at runtime.

```bash
uv sync --group build-reference          # installs httpx (build-only)
uv run python -m scripts.build_cns_reference \
    --out ./cns-reference-WHB-2023 \
    --cache-dir ./biccn_cache \
    --cells-per-class 5000 --min-donors 10
```

This downloads the BICCN Human Brain Cell Atlas v1.0 H5ADs from the CELLxGENE
Discover collection (`283d65eb-...`), balanced-subsamples to ~equal cells per
class, trains the 31-way supercluster model plus per-supercluster cluster
models, and writes a bundle directory. Upload the directory (tarred) as a
GitHub release asset; pin its tag as `reference_release` in profiles.

**Power-user retrain:** point `--out` at a local path and load it at runtime via
`reference_release="local:<path>"` (see Plan 2 runtime) to use a custom bundle.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: document CNS reference build + retrain path"
```

---

## Task 10: Full-suite + lint gate

- [ ] **Step 1: Run the whole build-reference test module**

Run: `uv run pytest tests/build_reference/ -v`
Expected: all PASS (integration deselected).

- [ ] **Step 2: Run the entire repo suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: previous count + new build-reference tests, all green.

- [ ] **Step 3: Lint + format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (fix and re-run if not).

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: ruff clean for cns build pipeline"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** Build pipeline (§4.1) → Tasks 4–7; bundle format (§4.3) → Tasks 1–2; GitHub-hosted artifact + retrain (§7, §7.1) → Tasks 8–9; balanced subsample to ~equal cells/class (§4.1 step 2) → Task 4; 31-way supercluster + per-supercluster cluster (decision #2, §4.1) → Task 6.
- **Out of scope here (Plan 2):** `rarecell.cns.taxonomy` tree application, `ReferenceBundle` lazy fetch/verify loader, `ProgressiveApplier`, `CNSTaxonomyConfig`, S2b pipeline stage, Colab demo. The `rarecell.cns.format` read helpers created here are what Plan 2 builds on.
- **Known follow-up:** exact native obs column names are resolved at runtime via candidate lists (Task 3); the integration test (Task 8) is the first place a real column-name mismatch would surface — if it does, extend the candidate lists in `labels.py`.
```
