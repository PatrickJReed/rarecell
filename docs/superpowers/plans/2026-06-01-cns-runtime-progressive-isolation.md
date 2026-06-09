# CNS Runtime Progressive Isolation — Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the runtime that loads a CNS reference bundle (built by Plan 1) and applies its CellTypist models as a progressive, hard-subsetting class gate (supercluster → cluster) inside `IsolateRunner`, with marker fallback, plus a Colab demo section.

**Architecture:** A new shipped subpackage area in `rarecell.cns`: `taxonomy.py` (resolve the decision path for a target node from the bundle), `bundle.py` (resolve a bundle from `local:<path>` or a GitHub release, stdlib download — no httpx at runtime), `progressive.py` (apply the model chain to a query AnnData, hard-subset on-path cells, marker fallback when a model is missing). A `CNSTaxonomyConfig` profile block drives it; a new `S2B_CLASS_GATE` pipeline stage runs between QC (S2) and clustering (S3), narrowing the query to the target node before the existing marker-driven isolation. The runtime only *loads* models and calls `celltypist.annotate` (works on scikit-learn 1.8 — the training-only sklearn shim from Plan 1 is NOT needed here).

**Tech Stack:** Python 3.11+, celltypist (core dep, annotate only), anndata/scanpy, pydantic v2, numpy, stdlib urllib/tarfile for bundle fetch, pytest. Repo is a uv workspace; ruff line-length 100; `mypy --strict`; tests via `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-06-01-progressive-taxonomy-classification-design.md` §4.2–§5, §9.

**Prereqs:** Plan 1 is merged — `rarecell.cns.format` provides `BundleManifest`, `DecisionArtifact`, `DecisionLevel`, `load_manifest`, `load_taxonomy`, `load_markers`, `sha256_file`, and `ReferenceBuildError`. The Plan 1 build pipeline (`scripts.build_cns_reference.build.build_bundle`) is used by tests to construct tiny real bundles.

---

## File Structure

Shipped (`packages/rarecell/src/rarecell/`):
- `cns/format.py` — MODIFIED: add `load_model(bundle_dir, artifact)` (sha-verify + lazy celltypist load).
- `cns/taxonomy.py` — NEW: `TaxonomyTree` — resolve the ordered decision path (with the on-path class to keep at each level) for a target node.
- `cns/bundle.py` — NEW: `ReferenceBundle` — resolve a bundle directory from `local:<path>` or a GitHub release tag (stdlib download + extract, cached).
- `cns/progressive.py` — NEW: `apply_progressive(...)` + `ProgressiveResult` — apply the model chain to a query, hard-subset, marker fallback, provenance.
- `profile/schema.py` — MODIFIED: add `CNSTaxonomyConfig` + `cns_taxonomy` field on `TargetCellProfile`.
- `state_machine/states.py` — MODIFIED: add `S2B_CLASS_GATE` + transitions.
- `state_machine/isolate.py` — MODIFIED: add `_s2b_class_gate` + wire into `run()`.

Tests (`packages/rarecell/tests/cns/`):
- `conftest.py` — a `tiny_bundle` fixture that builds a 1-supercluster-decision + 1-cluster-decision bundle from synthetic data.
- `test_format_load_model.py`, `test_taxonomy.py`, `test_bundle.py`, `test_progressive.py`, `test_cns_config.py`, `test_isolate_cns_gate.py`.

Demo:
- `examples/colab_demo.py` — MODIFIED: add a "Progressive BICCN class gate" section; regenerate `examples/colab_demo.ipynb` via jupytext.

---

## Task 1: `load_model` in the bundle format

**Files:**
- Modify: `packages/rarecell/src/rarecell/cns/format.py`
- Test: `packages/rarecell/tests/cns/__init__.py`, `packages/rarecell/tests/cns/conftest.py`, `packages/rarecell/tests/cns/test_format_load_model.py`

- [ ] **Step 1: Create the tiny-bundle fixture**

Create `packages/rarecell/tests/cns/__init__.py` (empty), then `packages/rarecell/tests/cns/conftest.py`. This builds a real (tiny) bundle once per module using the Plan 1 build pipeline (validating producer→consumer):

```python
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.build_cns_reference import build


def _atlas(seed: int = 0) -> ad.AnnData:
    """3 superclusters; Astrocyte has 2 clusters. log1p-CP10K-ish, capped <9.22."""
    rng = np.random.default_rng(seed)
    n_genes = 40
    rows, sc, cl, donors = [], [], [], []

    def block(shift: float, supercluster: str, cluster: str, n_donors: int = 4, per: int = 25) -> None:
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
```

The `Callable` import goes at the top of the conftest:

```python
from collections.abc import Callable
```

- [ ] **Step 2: Write the failing test**

Create `packages/rarecell/tests/cns/test_format_load_model.py`:

```python
from pathlib import Path

import pytest

from rarecell.cns import format as fmt
from rarecell.errors import ReferenceBuildError


def test_load_model_returns_usable_model(tiny_bundle: Path) -> None:
    manifest = fmt.load_manifest(tiny_bundle)
    sc_dec = next(d for d in manifest.decisions if d.level == "supercluster")
    model = fmt.load_model(tiny_bundle, sc_dec)
    # A celltypist Model exposes the trained classes.
    assert set(sc_dec.classes).issubset({str(c) for c in model.classifier.classes_})


def test_load_model_detects_sha_mismatch(tiny_bundle: Path) -> None:
    manifest = fmt.load_manifest(tiny_bundle)
    sc_dec = manifest.decisions[0]
    tampered = sc_dec.model_copy(update={"model_sha256": "0" * 64})
    with pytest.raises(ReferenceBuildError):
        fmt.load_model(tiny_bundle, tampered)
```

- [ ] **Step 3: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_format_load_model.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'load_model'`.

- [ ] **Step 4: Implement `load_model`**

Append to `packages/rarecell/src/rarecell/cns/format.py`:

```python
def load_model(bundle_dir: Path, artifact: DecisionArtifact) -> object:
    """Load and sha-verify the CellTypist model for a decision.

    celltypist is imported lazily so this module stays import-light.
    """
    path = Path(bundle_dir) / artifact.model_file
    if not path.exists():
        raise ReferenceBuildError(f"Missing model file {artifact.model_file} in bundle {bundle_dir}")
    actual = sha256_file(path)
    if actual != artifact.model_sha256:
        raise ReferenceBuildError(
            f"Model sha mismatch for {artifact.model_file}: {actual} != {artifact.model_sha256}"
        )
    from celltypist.models import Model

    return Model.load(str(path))
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_format_load_model.py -v`
Expected: PASS (building the fixture trains a couple of tiny models; ~20-40s).

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/format.py packages/rarecell/tests/cns/
git commit -m "feat(cns): add sha-verifying load_model to bundle format"
```

---

## Task 2: `TaxonomyTree` — resolve the decision path

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/taxonomy.py`
- Test: `packages/rarecell/tests/cns/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/cns/test_taxonomy.py`:

```python
from pathlib import Path

import pytest

from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.errors import ReferenceBuildError


def test_path_to_supercluster_target(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astrocyte", "supercluster")
    assert len(path) == 1
    artifact, keep_class = path[0]
    assert artifact.level == "supercluster"
    assert keep_class == "Astrocyte"


def test_path_to_cluster_target(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astro-1", "cluster")
    assert [a.level for a, _ in path] == ["supercluster", "cluster"]
    assert [keep for _, keep in path] == ["Astrocyte", "Astro-1"]
    assert path[1][0].parent == "Astrocyte"


def test_path_to_unknown_target_raises(tiny_bundle: Path) -> None:
    tax = TaxonomyTree.load(tiny_bundle)
    with pytest.raises(ReferenceBuildError):
        tax.path_to("NotACluster", "cluster")
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.taxonomy`.

- [ ] **Step 3: Implement `TaxonomyTree`**

Create `packages/rarecell/src/rarecell/cns/taxonomy.py`:

```python
"""Resolve the ordered decision path toward a target node from a bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rarecell.cns.format import (
    BundleManifest,
    DecisionArtifact,
    DecisionLevel,
    load_manifest,
    load_taxonomy,
)
from rarecell.errors import ReferenceBuildError


@dataclass
class TaxonomyTree:
    manifest: BundleManifest
    tree: dict[str, list[str]]  # supercluster -> [clusters]

    @classmethod
    def load(cls, bundle_dir: Path) -> TaxonomyTree:
        return cls(load_manifest(bundle_dir), load_taxonomy(bundle_dir))

    def _decision(self, level: DecisionLevel, parent: str | None) -> DecisionArtifact:
        for d in self.manifest.decisions:
            if d.level == level and d.parent == parent:
                return d
        raise ReferenceBuildError(
            f"No {level} decision (parent={parent!r}) in bundle manifest"
        )

    def supercluster_of(self, cluster: str) -> str:
        for sc, clusters in self.tree.items():
            if cluster in clusters:
                return sc
        raise ReferenceBuildError(f"Cluster {cluster!r} not found in taxonomy tree")

    def path_to(
        self, target: str, target_level: DecisionLevel
    ) -> list[tuple[DecisionArtifact, str]]:
        """Ordered [(decision, on-path class to keep), ...] from root to target."""
        if target_level == "supercluster":
            if target not in self.tree:
                raise ReferenceBuildError(f"Supercluster {target!r} not in taxonomy tree")
            return [(self._decision("supercluster", None), target)]
        # cluster target
        parent = self.supercluster_of(target)  # raises if unknown
        return [
            (self._decision("supercluster", None), parent),
            (self._decision("cluster", parent), target),
        ]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_taxonomy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/taxonomy.py packages/rarecell/tests/cns/test_taxonomy.py
git commit -m "feat(cns): add TaxonomyTree decision-path resolver"
```

---

## Task 3: `ReferenceBundle` — resolve local / GitHub-release bundle

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/bundle.py`
- Test: `packages/rarecell/tests/cns/test_bundle.py`

- [ ] **Step 1: Write the failing test** (local-path resolution only; network path is exercised by Plan 2's integration use, not unit tests)

Create `packages/rarecell/tests/cns/test_bundle.py`:

```python
from pathlib import Path

import pytest

from rarecell.cns.bundle import ReferenceBundle
from rarecell.errors import ReferenceBuildError


def test_resolve_local_path(tiny_bundle: Path, tmp_path: Path) -> None:
    rb = ReferenceBundle.resolve(f"local:{tiny_bundle}", cache_dir=tmp_path)
    assert rb.path == tiny_bundle
    # manifest is loadable through the bundle
    assert rb.manifest.biccn_release == "WHB-test"


def test_resolve_missing_local_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ReferenceBuildError):
        ReferenceBundle.resolve(f"local:{tmp_path / 'nope'}", cache_dir=tmp_path)
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.bundle`.

- [ ] **Step 3: Implement `ReferenceBundle`**

Create `packages/rarecell/src/rarecell/cns/bundle.py`:

```python
"""Resolve a CNS reference bundle from a local path or a GitHub release tag.

Runtime download uses the standard library (urllib/tarfile) so the shipped
package needs no HTTP dependency.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rarecell.cns.format import BundleManifest, load_manifest
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.bundle")

# GitHub release asset URL template. A release tagged `cns-<release>` carries
# an asset `cns-reference-<release>.tar.gz` extracting to `cns-reference-<release>/`.
RELEASE_URL = (
    "https://github.com/PatrickJReed/rarecell/releases/download/"
    "cns-{release}/cns-reference-{release}.tar.gz"
)


@dataclass
class ReferenceBundle:
    path: Path

    @property
    def manifest(self) -> BundleManifest:
        return load_manifest(self.path)

    @classmethod
    def resolve(cls, reference_release: str, *, cache_dir: Path) -> ReferenceBundle:
        """Resolve to a bundle directory.

        ``local:<path>`` uses ``<path>`` directly. Otherwise treat
        ``reference_release`` as a release tag and download+cache the asset.
        """
        if reference_release.startswith("local:"):
            path = Path(reference_release[len("local:") :])
            if not (path / "manifest.json").exists():
                raise ReferenceBuildError(f"No bundle (manifest.json) at {path}")
            return cls(path)

        cache_dir = Path(cache_dir)
        dest = cache_dir / f"cns-reference-{reference_release}"
        if not (dest / "manifest.json").exists():
            cls._download_and_extract(reference_release, cache_dir)
        if not (dest / "manifest.json").exists():
            raise ReferenceBuildError(
                f"Bundle for release {reference_release!r} not found after download"
            )
        return cls(dest)

    @staticmethod
    def _download_and_extract(release: str, cache_dir: Path) -> None:  # pragma: no cover - network
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = RELEASE_URL.format(release=release)
        tar_path = cache_dir / f"cns-reference-{release}.tar.gz"
        log.info("bundle.download", release=release, url=url)
        urllib.request.urlretrieve(url, tar_path)  # noqa: S310 - https github asset
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(cache_dir, filter="data")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_bundle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/bundle.py packages/rarecell/tests/cns/test_bundle.py
git commit -m "feat(cns): add ReferenceBundle local/release resolver"
```

---

## Task 4: `apply_progressive` — the progressive applier

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/progressive.py`
- Test: `packages/rarecell/tests/cns/test_progressive.py`

The applier tracks a boolean keep-mask over the input cells. At each decision it runs the level's model on the surviving subset, keeps cells predicted as the on-path class with confidence ≥ `min_confidence`, and writes a per-level label column. If a model is missing and `marker_fallback` is on, it scores the node's marker panel for the on-path class instead.

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/cns/test_progressive.py`. It rebuilds the same synthetic atlas the bundle was trained on (so predictions are meaningful) and asserts the gate keeps mostly Astrocyte cells.

```python
from pathlib import Path

import numpy as np

from rarecell.cns.bundle import ReferenceBundle
from rarecell.cns.progressive import apply_progressive
from rarecell.cns.taxonomy import TaxonomyTree


def test_apply_progressive_supercluster_gate(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=1)  # fresh draw from the same distributions
    bundle = ReferenceBundle.resolve(f"local:{tiny_bundle}", cache_dir=tiny_bundle.parent)
    tax = TaxonomyTree.load(bundle.path)
    path = tax.path_to("Astrocyte", "supercluster")

    result = apply_progressive(query, bundle.path, path, min_confidence=0.0)

    kept = query[result.mask]
    # Most kept cells should truly be Astrocyte; most Astrocytes should be kept.
    true_sc = query.obs["supercluster_term"].to_numpy()
    precision = (kept.obs["supercluster_term"] == "Astrocyte").mean()
    recall = result.mask[true_sc == "Astrocyte"].mean()
    assert precision >= 0.8
    assert recall >= 0.7
    assert "taxonomy_supercluster" in query.obs
    assert result.provenance["levels"][0]["level"] == "supercluster"


def test_apply_progressive_marker_fallback_when_model_absent(
    tiny_bundle: Path, atlas_factory
) -> None:
    query = atlas_factory(seed=2)
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astrocyte", "supercluster")
    # Point the decision at a nonexistent model file to force fallback.
    artifact, keep = path[0]
    broken = artifact.model_copy(update={"model_file": "nodes/supercluster/_missing.pkl"})
    result = apply_progressive(
        query, tiny_bundle, [(broken, keep)], min_confidence=0.0, marker_fallback=True
    )
    # Fallback still yields a non-empty, mostly-correct astrocyte subset.
    assert result.mask.sum() > 0
    assert result.provenance["levels"][0]["method"] == "marker_fallback"
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_progressive.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.progressive`.

- [ ] **Step 3: Implement `apply_progressive`**

Create `packages/rarecell/src/rarecell/cns/progressive.py`:

```python
"""Apply a CNS reference model chain to a query AnnData as a progressive gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from rarecell.cns.format import DecisionArtifact, load_markers, load_model
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.progressive")


@dataclass
class ProgressiveResult:
    mask: np.ndarray  # bool over the input cells: True == on-path to target
    provenance: dict[str, Any]


def _predict_with_model(
    sub: ad.AnnData, bundle_dir: Path, artifact: DecisionArtifact
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_label, confidence) arrays over ``sub`` using the model."""
    import celltypist

    model = load_model(bundle_dir, artifact)
    pred = celltypist.annotate(sub, model=model)
    labels = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    conf = pred.probability_matrix.max(axis=1).to_numpy()
    return labels, conf


def _predict_with_markers(
    sub: ad.AnnData, bundle_dir: Path, artifact: DecisionArtifact, keep_class: str
) -> tuple[np.ndarray, np.ndarray]:
    """Marker fallback: score the on-path class's panel; label keep_class where the
    score exceeds mean + 1 std, else "__other__". Confidence is 1.0/0.0."""
    panels = load_markers(bundle_dir, artifact.markers_file)
    genes = [g for g in panels.get(keep_class, []) if g in sub.var_names]
    if not genes:
        raise ReferenceBuildError(
            f"Marker fallback has no usable genes for class {keep_class!r}"
        )
    sc.tl.score_genes(sub, gene_list=genes, score_name="_cns_fallback")
    s = sub.obs["_cns_fallback"].to_numpy()
    passed = s > (s.mean() + s.std())
    labels = np.where(passed, keep_class, "__other__")
    conf = passed.astype(float)
    return labels, conf


def apply_progressive(
    adata: ad.AnnData,
    bundle_dir: Path,
    path: list[tuple[DecisionArtifact, str]],
    *,
    min_confidence: float = 0.5,
    marker_fallback: bool = True,
) -> ProgressiveResult:
    """Apply the decision chain to ``adata`` (post-QC, log1p-CP10K).

    Writes ``obs["taxonomy_<level>"]`` for surviving cells and returns the
    keep-mask (cells on the path toward the target) plus provenance.
    """
    n = adata.n_obs
    mask = np.ones(n, dtype=bool)
    obs_names = adata.obs_names.to_numpy()
    levels: list[dict[str, Any]] = []

    for artifact, keep_class in path:
        sub = adata[mask].copy()
        method = "model"
        try:
            labels, conf = _predict_with_model(sub, bundle_dir, artifact)
        except ReferenceBuildError:
            if not marker_fallback:
                raise
            method = "marker_fallback"
            labels, conf = _predict_with_markers(sub, bundle_dir, artifact, keep_class)

        sub_keep = (labels == keep_class) & (conf >= min_confidence)

        col = f"taxonomy_{artifact.level}"
        if col not in adata.obs:
            adata.obs[col] = ""
        adata.obs.loc[sub.obs_names, col] = labels

        # Drop, from the global mask, the surviving cells that failed this level.
        sub_names = sub.obs_names.to_numpy()
        drop_names = set(sub_names[~sub_keep].tolist())
        if drop_names:
            mask &= ~np.array([nm in drop_names for nm in obs_names], dtype=bool)

        levels.append(
            {
                "level": artifact.level,
                "keep_class": keep_class,
                "method": method,
                "n_in": int(sub.n_obs),
                "n_kept": int(sub_keep.sum()),
            }
        )
        log.info("cns_gate.level", **levels[-1])

    return ProgressiveResult(mask=mask, provenance={"levels": levels})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_progressive.py -v`
Expected: PASS. (If `celltypist.annotate`'s result attributes differ in this celltypist version — e.g. `probability_matrix` — adjust `_predict_with_model` to the actual attribute names; the test will tell you. Do NOT weaken the precision/recall thresholds.)

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/progressive.py packages/rarecell/tests/cns/test_progressive.py
git commit -m "feat(cns): add progressive applier with marker fallback"
```

---

## Task 5: `CNSTaxonomyConfig` profile block

**Files:**
- Modify: `packages/rarecell/src/rarecell/profile/schema.py`
- Test: `packages/rarecell/tests/cns/test_cns_config.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/cns/test_cns_config.py`:

```python
from rarecell.profile.schema import CNSTaxonomyConfig, TargetCellProfile


def test_default_is_disabled() -> None:
    cfg = CNSTaxonomyConfig()
    assert cfg.enabled is False
    assert cfg.target_level == "supercluster"


def test_profile_has_cns_taxonomy_default(minimal_profile_kwargs: dict) -> None:
    p = TargetCellProfile(**minimal_profile_kwargs)
    assert p.cns_taxonomy.enabled is False
```

Add a `minimal_profile_kwargs` fixture to `packages/rarecell/tests/cns/conftest.py` (append):

```python
@pytest.fixture
def minimal_profile_kwargs() -> dict:
    return {
        "profile_id": "p1",
        "name": "n",
        "description": "d",
        "target_lineage": "astrocyte",
        "tissue": ["brain"],
        "expected_abundance": {"min_fraction": 0.01, "max_fraction": 0.2},
        "positive_markers": {"astro": {"genes": ["AQP4", "GFAP"]}},
        "negative_markers": {},
        "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10.0},
    }
```

(If the existing repo already has a shared minimal-profile fixture in `packages/rarecell/tests/conftest.py`, import/reuse it instead of duplicating — check first and prefer reuse.)

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_cns_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'CNSTaxonomyConfig'`.

- [ ] **Step 3: Implement the config + field**

In `packages/rarecell/src/rarecell/profile/schema.py`, add the model near the other config blocks (e.g. after `BICCNRules`):

```python
class CNSTaxonomyConfig(BaseModel):
    enabled: bool = False
    target_node: str | None = None  # e.g. "Astrocyte"
    target_level: Literal["supercluster", "cluster"] = "supercluster"
    reference_release: str | None = None  # bundle tag or "local:<path>"
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    on_missing: Literal["marker_fallback", "skip"] = "marker_fallback"
```

Then add the field to `TargetCellProfile` (next to `biccn_rules`):

```python
    cns_taxonomy: CNSTaxonomyConfig = Field(default_factory=CNSTaxonomyConfig)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_cns_config.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing profile/freeze tests to confirm no hash regressions**

Run: `uv run pytest packages/rarecell/tests/ -k "profile or freeze or validate or report" -q`
Expected: PASS (the new field has a default; content-hash tests recompute, not hardcode).

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell/src/rarecell/profile/schema.py packages/rarecell/tests/cns/test_cns_config.py packages/rarecell/tests/cns/conftest.py
git commit -m "feat(profile): add CNSTaxonomyConfig block"
```

---

## Task 6: `S2B_CLASS_GATE` pipeline stage

**Files:**
- Modify: `packages/rarecell/src/rarecell/state_machine/states.py`
- Modify: `packages/rarecell/src/rarecell/state_machine/isolate.py`
- Test: `packages/rarecell/tests/cns/test_isolate_cns_gate.py`

- [ ] **Step 1: Add the state + transitions**

In `packages/rarecell/src/rarecell/state_machine/states.py`, add `S2B_CLASS_GATE = auto()` to `IsolateState` (after `S2_QC`), and update `_TRANSITIONS` so QC flows through the gate:

```python
    IsolateState.S2_QC: {IsolateState.S2B_CLASS_GATE},
    IsolateState.S2B_CLASS_GATE: {IsolateState.S3_CLUSTER},
```

(Leave the rest unchanged; `S3_CLUSTER`'s entry already points to `S4_GATE1`.)

- [ ] **Step 2: Write the failing test**

Create `packages/rarecell/tests/cns/test_isolate_cns_gate.py`:

```python
from pathlib import Path

import numpy as np

from rarecell.cns.gate import apply_cns_class_gate
from rarecell.profile.schema import CNSTaxonomyConfig


def test_cns_gate_narrows_to_target(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=3)
    cfg = CNSTaxonomyConfig(
        enabled=True,
        target_node="Astrocyte",
        target_level="supercluster",
        reference_release=f"local:{tiny_bundle}",
        min_confidence=0.0,
    )
    narrowed, provenance = apply_cns_class_gate(query, cfg, cache_dir=tiny_bundle.parent)
    assert narrowed.n_obs < query.n_obs
    assert (narrowed.obs["supercluster_term"] == "Astrocyte").mean() >= 0.8
    assert provenance["target_node"] == "Astrocyte"


def test_cns_gate_disabled_is_noop(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=4)
    cfg = CNSTaxonomyConfig(enabled=False)
    narrowed, provenance = apply_cns_class_gate(query, cfg, cache_dir=tiny_bundle.parent)
    assert narrowed.n_obs == query.n_obs
    assert provenance == {"enabled": False}
```

- [ ] **Step 3: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_isolate_cns_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.gate`.

- [ ] **Step 4: Implement the gate helper**

Create `packages/rarecell/src/rarecell/cns/gate.py` (a thin orchestration over bundle+taxonomy+applier, kept separate from `IsolateRunner` so it is unit-testable):

```python
"""Profile-driven CNS class gate: resolve bundle -> path -> apply -> subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad

from rarecell.cns.bundle import ReferenceBundle
from rarecell.cns.progressive import apply_progressive
from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.errors import RareCellError, ReferenceBuildError
from rarecell.logging import get_logger
from rarecell.profile.schema import CNSTaxonomyConfig

log = get_logger("rarecell.cns.gate")


def apply_cns_class_gate(
    adata: ad.AnnData, cfg: CNSTaxonomyConfig, *, cache_dir: Path
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Return (narrowed_adata, provenance). No-op when disabled."""
    if not cfg.enabled:
        return adata, {"enabled": False}
    if not cfg.target_node or not cfg.reference_release:
        raise ReferenceBuildError(
            "cns_taxonomy.enabled requires target_node and reference_release"
        )

    try:
        bundle = ReferenceBundle.resolve(cfg.reference_release, cache_dir=Path(cache_dir))
        tax = TaxonomyTree.load(bundle.path)
        path = tax.path_to(cfg.target_node, cfg.target_level)
        result = apply_progressive(
            adata,
            bundle.path,
            path,
            min_confidence=cfg.min_confidence,
            marker_fallback=(cfg.on_missing == "marker_fallback"),
        )
    except RareCellError as e:  # ReferenceBuildError is a RareCellError subclass
        if cfg.on_missing == "skip":
            log.warning("cns_gate.skipped", error=str(e))
            return adata, {"enabled": True, "skipped": True, "error": str(e)}
        raise

    narrowed = adata[result.mask].copy()
    prov: dict[str, Any] = {
        "enabled": True,
        "target_node": cfg.target_node,
        "target_level": cfg.target_level,
        "n_in": int(adata.n_obs),
        "n_out": int(narrowed.n_obs),
        **result.provenance,
    }
    log.info("cns_gate.done", n_in=prov["n_in"], n_out=prov["n_out"])
    return narrowed, prov
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_isolate_cns_gate.py -v`
Expected: PASS.

- [ ] **Step 6: Wire the gate into `IsolateRunner`**

In `packages/rarecell/src/rarecell/state_machine/isolate.py`:

Add the import near the top:
```python
from rarecell.cns.gate import apply_cns_class_gate
```

Add the stage method (after `_s2_qc`):
```python
    def _s2b_class_gate(self) -> None:
        narrowed, prov = apply_cns_class_gate(
            self.adata, self.profile.cns_taxonomy, cache_dir=self.out_dir
        )
        self.adata = narrowed
        self.adata.uns["cns_gate"] = prov
```

In `run()`, insert the stage between the S2_QC block and the S3_CLUSTER block:
```python
            self.state = IsolateState.S2B_CLASS_GATE
            self.logger.info("runner.state", state=self.state.name)
            self._s2b_class_gate()
```

- [ ] **Step 7: Run the runner + gate tests + a smoke of the existing runner tests**

Run: `uv run pytest packages/rarecell/tests/cns/ packages/rarecell/tests/state_machine/ -q`
Expected: PASS (existing runner tests use profiles with `cns_taxonomy` disabled by default, so the new stage is a no-op for them).

- [ ] **Step 8: Commit**

```bash
git add packages/rarecell/src/rarecell/state_machine/states.py packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/src/rarecell/cns/gate.py packages/rarecell/tests/cns/test_isolate_cns_gate.py
git commit -m "feat(cns): wire S2B progressive class gate into IsolateRunner"
```

---

## Task 7: Colab demo section

**Files:**
- Modify: `examples/colab_demo.py`
- Regenerate: `examples/colab_demo.ipynb` (jupytext)

- [ ] **Step 1: Add the demo section to the percent script**

In `examples/colab_demo.py`, add a new section AFTER the profile-freeze section and BEFORE "## 8. Run isolation". Use jupytext percent cells:

```python
# %% [markdown]
# ## 7b. Progressive BICCN class gate (optional)
#
# Before the marker-driven isolation, narrow the merged dataset to the
# **Astrocyte supercluster** using the pre-trained BICCN reference bundle
# (a small download — no atlas). This is the progressive
# supercluster -> cluster framework from BICCN's Human Brain Cell Atlas.

# %%
from rarecell.cns.gate import apply_cns_class_gate
from rarecell.profile.schema import CNSTaxonomyConfig

cns_cfg = CNSTaxonomyConfig(
    enabled=True,
    target_node="Astrocyte",
    target_level="supercluster",
    reference_release="WHB-2023",  # GitHub release tag; downloads the small bundle
    min_confidence=0.5,
)

gated, gate_prov = apply_cns_class_gate(adata, cns_cfg, cache_dir=local_root)
print(
    f"CNS class gate: {gate_prov['n_in']:,} -> {gate_prov['n_out']:,} cells "
    f"kept as {cns_cfg.target_node}"
)
print("Astrocyte fraction after gate:",
      f"{(gated.obs['cell_type_original'] == 'Astro').mean():.1%}")
adata = gated  # downstream isolation now runs on the narrowed population
```

(Note: this requires a published `cns-WHB-2023` GitHub release bundle. If none is published yet, the cell will fail at download — that's expected until Plan 1's build output is released; the section is otherwise inert to the rest of the notebook because it just reassigns `adata`.)

- [ ] **Step 2: Regenerate the notebook**

Run: `uv run jupytext --sync examples/colab_demo.py`
Expected: `examples/colab_demo.ipynb` updated.

- [ ] **Step 3: Verify the script parses and lint is clean**

Run: `uv run python -c "import ast; ast.parse(open('examples/colab_demo.py').read()); print('OK')"`
Run: `uv run ruff check examples/colab_demo.py`
Expected: OK / clean.

- [ ] **Step 4: Commit**

```bash
git add examples/colab_demo.py examples/colab_demo.ipynb
git commit -m "docs(demo): add progressive BICCN class-gate section"
```

---

## Task 8: Full-suite + lint gate

- [ ] **Step 1: Run the cns suite**

Run: `uv run pytest packages/rarecell/tests/cns/ -v`
Expected: all PASS.

- [ ] **Step 2: Run the entire repo suite**

Run: `uv run pytest -q`
Expected: previous count + new cns tests, all green (integration deselected).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run mypy packages/rarecell/src/rarecell/cns/`
Expected: clean. (Ignore pre-existing format drift in `examples/colab_demo.*` and `test_draft_anchor.py` — not part of this work.)

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: cns runtime suite + lint green"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** runtime subpackage §4.2 → Tasks 1–4 (`load_model`, `TaxonomyTree`, `ReferenceBundle`, `apply_progressive`); profile block §5.1 → Task 5; S2b hard-subset stage §5.2 → Task 6; demo §9 → Task 7. Marker fallback + graceful `on_missing` degradation → Tasks 4 & 6.
- **Runtime stays light:** `cns.format`/`taxonomy`/`bundle`/`progressive`/`gate` use only stdlib + numpy/scanpy/anndata + lazy celltypist; no httpx at runtime (bundle download is stdlib urllib). The training-only sklearn shim is NOT imported here.
- **Producer↔consumer validation:** the `tiny_bundle` fixture builds a real bundle via the Plan 1 pipeline and the runtime consumes it — a genuine end-to-end check.
- **Known risks to watch during execution:**
  1. `celltypist.annotate` result attribute names (`predicted_labels`, `probability_matrix`) — verify against the installed celltypist; the Task 4 test surfaces a mismatch.
  2. Adding `cns_taxonomy` to `TargetCellProfile` shifts the canonical model dump; confirm content-hash/freeze tests recompute (Task 5 Step 5) rather than hardcode.
  3. `apply_progressive` writes `obs["taxonomy_*"]` as object columns; ensure no dtype warnings break tests.
```
