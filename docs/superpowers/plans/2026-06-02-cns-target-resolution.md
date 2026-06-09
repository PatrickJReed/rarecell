# CNS Target Resolution & Characterization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve a query to a CNS taxonomy target (`mode`/`gate_node`/`gate_level`/`characterize_level`) via deterministic node retrieval + an agent decision, and characterize the isolated population against reference clusters.

**Architecture:** Three new units — `cns.retrieve` (deterministic node scorer over the bundle), `agent.resolve` (Claude picks node-vs-program from a retrieval shortlist, cited), `cns.characterize` (label isolated cells against reference clusters + annotations) — plus light wiring into `CNSTaxonomyConfig`, the build's S3 annotation, and `IsolateRunner`.

**Tech Stack:** Python 3.11+, pydantic v2, celltypist (apply), anndata/pandas, the existing `rarecell.cns` bundle + `rarecell.agent` Claude client. ruff line-length 100; `mypy --strict`; `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-06-02-cns-target-resolution-design.md`

**Prereqs:** The CNS bundle (Plan 1 build) + runtime (Plan 2) are merged. The vendored Table S3 CSV already includes `top_regions`/`top_dissections` columns. Test fixtures: `packages/rarecell/tests/cns/conftest.py` provides `tiny_bundle` (a real built bundle) and `atlas_factory`.

---

## File Structure

Shipped (`packages/rarecell/src/rarecell/`):
- `cns/retrieve.py` — NEW: `NodeDescriptor`, `NodeMatch`, `build_catalog`, `score_nodes`.
- `cns/characterize.py` — NEW: `CharacterizationResult`, `characterize`.
- `agent/resolve.py` — NEW: `TargetResolution`, `resolve_target`, `resolve_cns_target`.
- `profile/schema.py` — MODIFIED: `CNSTaxonomyConfig` gains `mode`, `characterize_level`, `rationale`, `citations`.
- `state_machine/isolate.py` — MODIFIED: post-isolation characterize stage.

Dev tooling:
- `scripts/build_cns_reference/annotate_s3.py` — MODIFIED: carry `top_regions` into the annotation map.

Tests:
- `packages/rarecell/tests/cns/test_retrieve.py`, `test_characterize.py`
- `packages/rarecell/tests/agent/test_resolve.py`
- `packages/rarecell/tests/cns/test_cns_config.py` — extended
- `tests/build_reference/test_annotate_s3.py` — extended

---

## Task 1: Build addition — carry regions into the S3 annotation map

**Files:**
- Modify: `scripts/build_cns_reference/annotate_s3.py`
- Test: `tests/build_reference/test_annotate_s3.py`

- [ ] **Step 1: Write the failing test** — append to `tests/build_reference/test_annotate_s3.py`:

```python
def test_build_s3_map_includes_regions() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [52],
            "class_auto": ["ASTRO"],
            "subtype_auto": [np.nan],
            "neuropeptide_auto": [np.nan],
            "top_enriched_genes": ["AQP4, GFAP"],
            "top_regions": ["Cerebral cortex: 40%, Thalamus: 20%"],
        }
    )
    m = annotate_s3.build_s3_map(df)
    assert m["52"]["regions"] == ["Cerebral cortex", "Thalamus"]
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest tests/build_reference/test_annotate_s3.py::test_build_s3_map_includes_regions -v`
Expected: FAIL with `KeyError: 'regions'`.

- [ ] **Step 3: Implement** — in `scripts/build_cns_reference/annotate_s3.py`, add a region parser and include it in `build_s3_map`. Add this helper above `build_s3_map`:

```python
def _parse_regions(raw: str) -> list[str]:
    """Parse 'Cerebral cortex: 40%, Thalamus: 20%' -> ['Cerebral cortex', 'Thalamus']."""
    out: list[str] = []
    for part in _clean(raw).split(","):
        name = part.split(":")[0].strip()
        if name:
            out.append(name)
    return out
```

Then in `build_s3_map`, add a `regions` key to each entry (alongside `class`/`subtype`/`neuropeptide`/`markers`):

```python
        out[str(int(r["cluster_id"]))] = {
            "class": _clean(r.get("class_auto")),
            "subtype": _clean(r.get("subtype_auto")),
            "neuropeptide": _clean(r.get("neuropeptide_auto")),
            "markers": markers,
            "regions": _parse_regions(r.get("top_regions", "")),
        }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/build_reference/test_annotate_s3.py -v`
Expected: PASS (all, including the new test).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cns_reference/annotate_s3.py tests/build_reference/test_annotate_s3.py
git commit -m "feat(build): carry Table S3 top_regions into cluster annotations"
```

---

## Task 2: Extend `CNSTaxonomyConfig`

**Files:**
- Modify: `packages/rarecell/src/rarecell/profile/schema.py`
- Test: `packages/rarecell/tests/cns/test_cns_config.py`

- [ ] **Step 1: Write the failing test** — append to `packages/rarecell/tests/cns/test_cns_config.py`:

```python
def test_cns_config_resolution_fields_default() -> None:
    cfg = CNSTaxonomyConfig()
    assert cfg.mode == "node"
    assert cfg.characterize_level == "cluster"
    assert cfg.rationale is None
    assert cfg.citations == []
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_cns_config.py::test_cns_config_resolution_fields_default -v`
Expected: FAIL with `AttributeError: ... 'mode'`.

- [ ] **Step 3: Implement** — in `packages/rarecell/src/rarecell/profile/schema.py`, add fields to `CNSTaxonomyConfig` (after `on_missing`):

```python
    mode: Literal["node", "program"] = "node"
    characterize_level: Literal["cluster", "subcluster"] = "cluster"
    rationale: str | None = None
    citations: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_cns_config.py -v`
Expected: PASS.

- [ ] **Step 5: Run the profile/freeze tests** — Run: `uv run pytest packages/rarecell/tests/ -k "profile or freeze or validate" -q` → PASS (new fields have defaults).

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell/src/rarecell/profile/schema.py packages/rarecell/tests/cns/test_cns_config.py
git commit -m "feat(profile): add mode/characterize_level/rationale/citations to CNSTaxonomyConfig"
```

---

## Task 3: `cns.retrieve` — node catalog

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/retrieve.py`
- Test: `packages/rarecell/tests/cns/test_retrieve.py`

- [ ] **Step 1: Write the failing test** — create `packages/rarecell/tests/cns/test_retrieve.py`:

```python
from pathlib import Path

from rarecell.cns.retrieve import NodeDescriptor, build_catalog


def test_build_catalog_has_superclusters_and_clusters(tiny_bundle: Path) -> None:
    catalog = build_catalog(tiny_bundle)
    levels = {n.level for n in catalog}
    assert levels == {"supercluster", "cluster"}
    sc = [n for n in catalog if n.level == "supercluster"]
    assert any(n.name == "Astrocyte" for n in sc)
    # superclusters carry a marker panel from the bundle
    astro = next(n for n in sc if n.name == "Astrocyte")
    assert isinstance(astro, NodeDescriptor) and isinstance(astro.markers, list)
    # clusters carry a parent supercluster
    clusters = [n for n in catalog if n.level == "cluster"]
    assert all(n.parent for n in clusters)
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.retrieve`.

- [ ] **Step 3: Implement `build_catalog`** — create `packages/rarecell/src/rarecell/cns/retrieve.py`:

```python
"""Deterministic retrieval over the CNS reference taxonomy.

Builds a searchable catalog of reference nodes (superclusters + clusters) from
the bundle, and scores them against a query's markers / lineage / tissue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rarecell.cns.format import load_annotations, load_markers, load_taxonomy

# Map a profile's free-text lineage to a Siletti class code.
_LINEAGE_TO_CLASS = {
    "astrocyte": "ASTRO",
    "microglia": "MGL",
    "oligodendrocyte": "OLIGO",
    "oligodendrocyte precursor": "OPC",
    "opc": "OPC",
    "neuron": "NEUR",
    "endothelial": "ENDO",
    "fibroblast": "FIB",
    "pericyte": "PER",
    "ependymal": "EPEN",
}


@dataclass
class NodeDescriptor:
    name: str
    level: str  # "supercluster" | "cluster"
    parent: str | None
    cell_class: str
    markers: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    neurotransmitter: str = ""


def build_catalog(bundle_dir: Path) -> list[NodeDescriptor]:
    tree = load_taxonomy(bundle_dir)  # supercluster -> [cluster names]
    ann = load_annotations(bundle_dir)  # cluster name -> {class, markers, regions, ...}
    sc_panels = load_markers(bundle_dir, "nodes/supercluster/_markers.json")

    catalog: list[NodeDescriptor] = []
    for sc, clusters in tree.items():
        catalog.append(
            NodeDescriptor(
                name=sc, level="supercluster", parent=None, cell_class=sc,
                markers=list(sc_panels.get(sc, [])),
            )
        )
        for cl in clusters:
            a = ann.get(cl, {})
            raw_markers = a.get("markers", [])
            raw_regions = a.get("regions", [])
            catalog.append(
                NodeDescriptor(
                    name=cl, level="cluster", parent=sc,
                    cell_class=str(a.get("class", "")),
                    markers=[str(g) for g in raw_markers] if isinstance(raw_markers, list) else [],
                    regions=[str(x) for x in raw_regions] if isinstance(raw_regions, list) else [],
                    neurotransmitter=str(a.get("neurotransmitter", "")),
                )
            )
    return catalog
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_retrieve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/retrieve.py packages/rarecell/tests/cns/test_retrieve.py
git commit -m "feat(cns): add reference node catalog (build_catalog)"
```

---

## Task 4: `cns.retrieve` — `score_nodes`

**Files:**
- Modify: `packages/rarecell/src/rarecell/cns/retrieve.py`
- Test: `packages/rarecell/tests/cns/test_retrieve.py`

- [ ] **Step 1: Write the failing test** — append to `packages/rarecell/tests/cns/test_retrieve.py`:

```python
from rarecell.cns.retrieve import NodeMatch, score_nodes


def _catalog() -> list[NodeDescriptor]:
    return [
        NodeDescriptor("Astrocyte", "supercluster", None, "Astrocyte", ["AQP4", "GFAP", "SLC1A2"]),
        NodeDescriptor("MGE interneuron", "supercluster", None, "MGE interneuron", ["LHX6", "GAD1"]),
        NodeDescriptor("Astro_52", "cluster", "Astrocyte", "ASTRO", ["AQP4", "GFAP"],
                       regions=["Cerebral cortex"]),
        NodeDescriptor("MGE_259", "cluster", "MGE interneuron", "NEUR", ["LHX6", "PVALB"]),
    ]


def test_score_ranks_marker_match_first() -> None:
    matches = score_nodes(_catalog(), markers=["AQP4", "GFAP"], lineage="astrocyte")
    top_sc = next(m for m in matches if m.node.level == "supercluster")
    assert top_sc.node.name == "Astrocyte"
    assert top_sc.signals["marker_overlap"] > 0
    assert top_sc.signals["class_match"] == 1.0
    assert isinstance(top_sc, NodeMatch)


def test_score_region_signal_and_top_k() -> None:
    matches = score_nodes(_catalog(), markers=["AQP4"], lineage="astrocyte",
                          tissue="cerebral cortex", top_k_per_level=1)
    # only the best per level returned
    assert sum(1 for m in matches if m.node.level == "cluster") == 1
    astro_cl = next(m for m in matches if m.node.level == "cluster")
    assert astro_cl.node.name == "Astro_52"
    assert astro_cl.signals["region_match"] == 1.0
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_retrieve.py -v`
Expected: FAIL with `ImportError: cannot import name 'NodeMatch'` / `score_nodes`.

- [ ] **Step 3: Implement `score_nodes`** — append to `packages/rarecell/src/rarecell/cns/retrieve.py`:

```python
@dataclass
class NodeMatch:
    node: NodeDescriptor
    score: float
    signals: dict[str, float]


def _overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = {x.upper() for x in a}, {x.upper() for x in b}
    return len(sa & sb) / min(len(sa), len(sb))


def _class_match(lineage: str | None, cell_class: str) -> float:
    if not lineage or not cell_class:
        return 0.0
    le, ce = lineage.lower().strip(), cell_class.lower().strip()
    if le == ce or le in ce or ce in le:
        return 1.0
    code = _LINEAGE_TO_CLASS.get(le)
    return 1.0 if code and code.lower() == ce else 0.0


def _region_match(tissue: str | None, regions: list[str]) -> float:
    if not tissue or not regions:
        return 0.0
    toks = set(tissue.lower().split())
    rtoks = set(" ".join(regions).lower().split())
    return 1.0 if toks & rtoks else 0.0


def score_nodes(
    catalog: list[NodeDescriptor], *, markers: list[str],
    lineage: str | None = None, tissue: str | None = None, top_k_per_level: int = 8,
) -> list[NodeMatch]:
    matches: list[NodeMatch] = []
    for n in catalog:
        s_marker = _overlap(markers, n.markers)
        s_class = _class_match(lineage, n.cell_class)
        s_region = _region_match(tissue, n.regions)
        score = 0.5 * s_marker + 0.3 * s_class + 0.2 * s_region
        matches.append(
            NodeMatch(
                node=n, score=score,
                signals={"marker_overlap": s_marker, "class_match": s_class, "region_match": s_region},
            )
        )
    out: list[NodeMatch] = []
    for level in ("supercluster", "cluster"):
        ranked = sorted((m for m in matches if m.node.level == level), key=lambda m: -m.score)
        out.extend(ranked[:top_k_per_level])
    return out
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_retrieve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/retrieve.py packages/rarecell/tests/cns/test_retrieve.py
git commit -m "feat(cns): add score_nodes (marker/class/region retrieval)"
```

---

## Task 5: `cns.characterize`

**Files:**
- Create: `packages/rarecell/src/rarecell/cns/characterize.py`
- Test: `packages/rarecell/tests/cns/test_characterize.py`

- [ ] **Step 1: Write the failing test** — create `packages/rarecell/tests/cns/test_characterize.py`:

```python
from pathlib import Path

from rarecell.cns.characterize import characterize


def test_characterize_labels_and_summarizes(tiny_bundle: Path, atlas_factory) -> None:
    # Use only Astrocyte cells as the "isolated" population.
    query = atlas_factory(seed=7)
    isolated = query[query.obs["supercluster_term"] == "Astrocyte"].copy()
    result = characterize(isolated, tiny_bundle, level="cluster", parent_node="Astrocyte")

    assert len(result.per_cell_labels) == isolated.n_obs
    assert result.summary  # non-empty
    fractions = sum(row["fraction"] for row in result.summary)
    assert 0.99 <= fractions <= 1.01
    # rows carry annotation fields
    assert "class" in result.summary[0] and "fraction" in result.summary[0]
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_characterize.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.cns.characterize`.

- [ ] **Step 3: Implement** — create `packages/rarecell/src/rarecell/cns/characterize.py`:

```python
"""Characterize an isolated population against reference clusters + annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import pandas as pd

from rarecell.cns.format import load_annotations, load_model
from rarecell.cns.taxonomy import TaxonomyTree
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.characterize")


@dataclass
class CharacterizationResult:
    per_cell_labels: pd.Series
    summary: list[dict[str, Any]]


def characterize(
    isolated: ad.AnnData, bundle_dir: Path, *,
    level: Literal["cluster", "subcluster"], parent_node: str,
) -> CharacterizationResult:
    """Classify isolated cells into the parent_node's child clusters and summarize.

    ``level="subcluster"`` falls back to ``"cluster"`` (no subcluster models in v1).
    """
    import celltypist

    tax = TaxonomyTree.load(bundle_dir)
    # v1 has supercluster + cluster models only.
    artifact = tax._decision("cluster", parent_node)
    model = load_model(bundle_dir, artifact)
    pred = celltypist.annotate(isolated, model=model)
    labels = pred.predicted_labels["predicted_labels"].astype(str).to_numpy()
    per_cell = pd.Series(labels, index=isolated.obs_names, name="reference_cluster")

    ann = load_annotations(bundle_dir)
    n = len(per_cell)
    summary: list[dict[str, Any]] = []
    for cl, count in per_cell.value_counts().items():
        a = ann.get(str(cl), {})
        summary.append(
            {
                "cluster": str(cl),
                "n": int(count),
                "fraction": float(count) / n,
                "class": a.get("class", ""),
                "neurotransmitter": a.get("neurotransmitter", ""),
                "regions": a.get("regions", []),
                "markers": a.get("markers", []),
            }
        )
    summary.sort(key=lambda d: -d["fraction"])
    return CharacterizationResult(per_cell_labels=per_cell, summary=summary)
```

Note: `level` is accepted for interface stability; v1 always uses the cluster model (subcluster falls back). Reference `ReferenceBuildError` is imported because `tax._decision` raises it when the parent has no cluster model — the caller (Task 8) handles that.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_characterize.py -v`
Expected: PASS. (If `celltypist.annotate`'s attributes differ, mirror the fix already used in `cns/progressive.py`.)

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/cns/characterize.py packages/rarecell/tests/cns/test_characterize.py
git commit -m "feat(cns): add characterize (label isolated cells vs reference clusters)"
```

---

## Task 6: `agent.resolve` — `TargetResolution` + `resolve_target`

**Files:**
- Create: `packages/rarecell/src/rarecell/agent/resolve.py`
- Test: `packages/rarecell/tests/agent/test_resolve.py`

- [ ] **Step 1: Write the failing test** — create `packages/rarecell/tests/agent/test_resolve.py`. A fake client returns a canned JSON block (mirrors how the drafting flow consumes `client.messages_create`).

```python
import json

from rarecell.cns.retrieve import NodeDescriptor, NodeMatch
from rarecell.agent.resolve import TargetResolution, resolve_target


class _FakeClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def messages_create(self, *, messages, tools=None):
        return {"content": [{"type": "text", "text": "```json\n" + json.dumps(self._payload) + "\n```"}]}


def _candidates() -> list[NodeMatch]:
    return [
        NodeMatch(NodeDescriptor("Astrocyte", "supercluster", None, "Astrocyte", ["AQP4", "GFAP"]),
                  0.6, {"marker_overlap": 0.2, "class_match": 1.0, "region_match": 0.0}),
        NodeMatch(NodeDescriptor("Astro_52", "cluster", "Astrocyte", "ASTRO", ["AQP4"]),
                  0.4, {"marker_overlap": 0.1, "class_match": 1.0, "region_match": 0.0}),
    ]


def test_resolve_returns_validated_program_decision() -> None:
    client = _FakeClient(
        {
            "mode": "program",
            "gate_node": "Astrocyte",
            "gate_level": "supercluster",
            "characterize_level": "cluster",
            "rationale": "SNAP is an astrocyte gene program, not a discrete cluster.",
            "citations": ["pmid:38448582"],
            "confidence": 0.8,
        }
    )

    class _P:  # minimal profile stand-in (resolve only reads a few attrs)
        name = "SNAP astrocytes"
        target_lineage = "astrocyte"
        description = "SNAP-expressing astrocytes from DLPFC"
        positive_markers: dict = {}

    res = resolve_target(_P(), candidates=_candidates(), client=client)
    assert isinstance(res, TargetResolution)
    assert res.mode == "program"
    assert res.gate_node == "Astrocyte"
    assert res.characterize_level == "cluster"
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/agent/test_resolve.py -v`
Expected: FAIL with `ModuleNotFoundError: rarecell.agent.resolve`.

- [ ] **Step 3: Implement** — create `packages/rarecell/src/rarecell/agent/resolve.py`:

```python
"""Agent target resolution: choose node-vs-program from a retrieval shortlist."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from rarecell.cns.retrieve import NodeMatch
from rarecell.logging import get_logger

_log = get_logger("rarecell.agent.resolve")


class TargetResolution(BaseModel):
    mode: Literal["node", "program"]
    gate_node: str
    gate_level: Literal["supercluster", "cluster"]
    characterize_level: Literal["cluster", "subcluster"]
    rationale: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def _extract_json_block(text: str) -> dict | None:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _build_resolution_prompt(profile: Any, candidates: list[NodeMatch]) -> str:
    lines = [
        "You map a target cell population to the BICCN whole-human-brain taxonomy.",
        "Decide whether the target is a DISCRETE reference node (gate on it: mode='node')",
        "or a gene PROGRAM / state within a broader container (gate on the container and",
        "isolate by markers: mode='program').",
        "",
        f"Target name: {getattr(profile, 'name', '')}",
        f"Lineage: {getattr(profile, 'target_lineage', '')}",
        f"Description: {getattr(profile, 'description', '')}",
        f"Positive marker panels: {list(getattr(profile, 'positive_markers', {}) or {})}",
        "",
        "Candidate reference nodes (name | level | class | top markers | signals):",
    ]
    for m in candidates:
        lines.append(
            f"- {m.node.name} | {m.node.level} | {m.node.cell_class} | "
            f"{m.node.markers[:8]} | {m.signals}"
        )
    lines += [
        "",
        "Reply with ONLY a json fenced block:",
        '```json',
        '{"mode":"node|program","gate_node":"<one candidate name or its parent>",',
        '"gate_level":"supercluster|cluster","characterize_level":"cluster|subcluster",',
        '"rationale":"...","citations":["pmid:..."],"confidence":0.0}',
        '```',
    ]
    return "\n".join(lines)


def resolve_target(profile: Any, *, candidates: list[NodeMatch], client: Any) -> TargetResolution:
    """Ask the agent to resolve the target from the candidate shortlist."""
    msg = _build_resolution_prompt(profile, candidates)
    resp = client.messages_create(messages=[{"role": "user", "content": msg}])
    text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise ValueError("Resolution response had no text blocks.")
    parsed = _extract_json_block(text_blocks[0]["text"])
    if parsed is None:
        raise ValueError("Resolution response did not contain a parseable JSON block.")
    return TargetResolution.model_validate(parsed)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/agent/test_resolve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/agent/resolve.py packages/rarecell/tests/agent/test_resolve.py
git commit -m "feat(agent): add resolve_target (node-vs-program decision)"
```

---

## Task 7: `agent.resolve` — `resolve_cns_target` orchestrator

**Files:**
- Modify: `packages/rarecell/src/rarecell/agent/resolve.py`
- Test: `packages/rarecell/tests/agent/test_resolve.py`

- [ ] **Step 1: Write the failing test** — append to `packages/rarecell/tests/agent/test_resolve.py`:

```python
from pathlib import Path

from rarecell.agent.resolve import resolve_cns_target
from rarecell.profile.schema import CNSTaxonomyConfig


def test_resolve_cns_target_populates_config(tiny_bundle: Path) -> None:
    client = _FakeClient(
        {
            "mode": "program", "gate_node": "Astrocyte", "gate_level": "supercluster",
            "characterize_level": "cluster", "rationale": "program", "citations": [], "confidence": 0.7,
        }
    )

    class _P:
        name = "x"
        target_lineage = "astrocyte"
        description = "d"
        positive_markers = {"astro": type("M", (), {"genes": ["AQP4", "GFAP"]})()}

    cfg = resolve_cns_target(
        _P(), bundle_dir=tiny_bundle, reference_release=f"local:{tiny_bundle}", client=client
    )
    assert isinstance(cfg, CNSTaxonomyConfig)
    assert cfg.enabled and cfg.mode == "program"
    assert cfg.target_node == "Astrocyte" and cfg.target_level == "supercluster"
    assert cfg.reference_release == f"local:{tiny_bundle}"
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/agent/test_resolve.py::test_resolve_cns_target_populates_config -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_cns_target'`.

- [ ] **Step 3: Implement** — append to `packages/rarecell/src/rarecell/agent/resolve.py` (add imports for `Path`, `build_catalog`, `score_nodes`, `CNSTaxonomyConfig` at the top of the file):

```python
def _profile_markers(profile: Any) -> list[str]:
    genes: list[str] = []
    for panel in (getattr(profile, "positive_markers", {}) or {}).values():
        genes.extend(getattr(panel, "genes", []) or [])
    return genes


def resolve_cns_target(
    profile: Any, *, bundle_dir: Path, reference_release: str, client: Any,
    top_k_per_level: int = 8,
) -> CNSTaxonomyConfig:
    """Run retrieval + agent resolution, returning a populated CNSTaxonomyConfig."""
    catalog = build_catalog(bundle_dir)
    tissue = " ".join(getattr(profile, "tissue", []) or []) or None
    candidates = score_nodes(
        catalog, markers=_profile_markers(profile),
        lineage=getattr(profile, "target_lineage", None), tissue=tissue,
        top_k_per_level=top_k_per_level,
    )
    res = resolve_target(profile, candidates=candidates, client=client)
    return CNSTaxonomyConfig(
        enabled=True,
        target_node=res.gate_node,
        target_level=res.gate_level,
        mode=res.mode,
        characterize_level=res.characterize_level,
        reference_release=reference_release,
        rationale=res.rationale,
        citations=res.citations,
    )
```

Add these imports at the top of `resolve.py`:

```python
from pathlib import Path

from rarecell.cns.retrieve import NodeMatch, build_catalog, score_nodes
from rarecell.profile.schema import CNSTaxonomyConfig
```

(Replace the existing `from rarecell.cns.retrieve import NodeMatch` line with the combined import above.)

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/agent/test_resolve.py -v`
Expected: PASS (both resolve tests).

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/agent/resolve.py packages/rarecell/tests/agent/test_resolve.py
git commit -m "feat(agent): add resolve_cns_target orchestrator -> CNSTaxonomyConfig"
```

---

## Task 8: Wire characterization into `IsolateRunner`

**Files:**
- Modify: `packages/rarecell/src/rarecell/state_machine/isolate.py`
- Test: `packages/rarecell/tests/cns/test_isolate_characterize.py`

- [ ] **Step 1: Write the failing test** — create `packages/rarecell/tests/cns/test_isolate_characterize.py`. It calls the new helper directly (the full runner is covered elsewhere); the helper does the characterize-and-store.

```python
from pathlib import Path

from rarecell.state_machine.isolate import characterize_isolated
from rarecell.profile.schema import CNSTaxonomyConfig


def test_characterize_isolated_stores_summary(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=8)
    isolated = query[query.obs["supercluster_term"] == "Astrocyte"].copy()
    cfg = CNSTaxonomyConfig(
        enabled=True, target_node="Astrocyte", target_level="supercluster",
        mode="program", characterize_level="cluster",
        reference_release=f"local:{tiny_bundle}",
    )
    characterize_isolated(isolated, cfg, cache_dir=tiny_bundle.parent)
    assert "cns_characterization" in isolated.uns
    assert isolated.uns["cns_characterization"]["summary"]


def test_characterize_isolated_noop_when_disabled(tiny_bundle: Path, atlas_factory) -> None:
    isolated = atlas_factory(seed=9)
    characterize_isolated(isolated, CNSTaxonomyConfig(enabled=False), cache_dir=tiny_bundle.parent)
    assert "cns_characterization" not in isolated.uns
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/rarecell/tests/cns/test_isolate_characterize.py -v`
Expected: FAIL with `ImportError: cannot import name 'characterize_isolated'`.

- [ ] **Step 3: Implement the helper** — in `packages/rarecell/src/rarecell/state_machine/isolate.py`, add at module level (after the imports):

```python
def characterize_isolated(isolated, cfg, *, cache_dir):
    """If the CNS taxonomy gate is enabled, label the isolated cells against
    reference clusters and store the summary in ``isolated.uns``."""
    from rarecell.cns.bundle import ReferenceBundle
    from rarecell.cns.characterize import characterize
    from rarecell.errors import RareCellError

    if not cfg.enabled or not cfg.reference_release or not cfg.target_node:
        return
    try:
        bundle = ReferenceBundle.resolve(cfg.reference_release, cache_dir=Path(cache_dir))
        result = characterize(
            isolated, bundle.path, level=cfg.characterize_level, parent_node=cfg.target_node
        )
    except RareCellError as e:  # missing model / unresolvable bundle -> skip
        isolated.uns["cns_characterization"] = {"skipped": True, "error": str(e)}
        return
    isolated.obs["reference_cluster"] = result.per_cell_labels.reindex(isolated.obs_names).to_numpy()
    isolated.uns["cns_characterization"] = {"summary": result.summary}
```

Then call it in `run()` right before `write_isolation_report(...)` (after `isolated` is finalized at S6):

```python
            characterize_isolated(isolated, self.profile.cns_taxonomy, cache_dir=self.out_dir)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/rarecell/tests/cns/test_isolate_characterize.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing runner tests** — Run: `uv run pytest packages/rarecell/tests/state_machine/ -q` → PASS (default profiles have `cns_taxonomy.enabled=False`, so the call is a no-op).

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/tests/cns/test_isolate_characterize.py
git commit -m "feat(cns): characterize the isolated population in IsolateRunner"
```

---

## Task 9: Full-suite + lint gate

- [ ] **Step 1: Run the cns + agent + build suites**

Run: `uv run pytest packages/rarecell/tests/cns/ packages/rarecell/tests/agent/ tests/build_reference/ -q`
Expected: all PASS.

- [ ] **Step 2: Run the entire repo suite**

Run: `uv run pytest -q`
Expected: previous count + new tests, all green (integration deselected).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run mypy packages/rarecell/src/rarecell/cns/ packages/rarecell/src/rarecell/agent/resolve.py`
Expected: clean. (Ignore pre-existing drift in `examples/colab_demo.*` and `test_draft_anchor.py`.)

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: cns target-resolution suite + lint green"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** retrieve §4.1 → Tasks 3–4; resolve §4.2 → Tasks 6–7; characterize §4.3 → Task 5 + Task 8 wiring; regions build addition §5 → Task 1; `CNSTaxonomyConfig` fields §6 → Task 2; report §6 → characterization lands in `uns` (Task 8); a `report.py` surface is a thin follow-up reading `uns["cns_characterization"]`.
- **Deferred (consistent with spec §9):** the demo notebook section, the `report.py` rendering of the rationale/summary, and the `mode="node"` S3–S6 short-circuit. These are follow-ups that need a published bundle to be meaningful.
- **Reuse:** `characterize` and `resolve` both lean on existing bundle read helpers (`load_taxonomy`/`load_annotations`/`load_markers`/`load_model`) and `TaxonomyTree`; the agent client contract (`messages_create` → json-fenced block) mirrors `agent/draft.py`.
- **Risk to watch:** `celltypist.annotate` result attribute names in `characterize` (mirror `cns/progressive.py` if they differ); `tax._decision` raising `ReferenceBuildError` when a supercluster has no cluster model (handled by the Task 8 try/except).
```
