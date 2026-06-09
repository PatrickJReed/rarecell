# Progressive BICCN-Taxonomy Isolation for CNS Cell Types — Design

**Date:** 2026-06-01 (rev. 2 — refocused on CNS / BICCN sole reference)
**Status:** Draft for review
**Topic owner:** Patrick J. Reed

## 1. Motivation

`rarecell` isolates a target population with a **flat** procedure: cluster all
cells, score the profile's marker panels per cluster, keep the clusters the
recommender judges to be the target (`state_machine/isolate.py` S3→S6). The
existing reference hooks don't learn anything — CellTypist only *loads* models,
and "BICCN" is nine hardcoded marker rules (`core/evidence.py:31`).

We are **refocusing `rarecell` on CNS cell types** and making the **BICCN Human
Brain Cell Atlas v1.0** the *sole* gold-standard reference for cell identity.
Isolation becomes **progressive along the BICCN taxonomy** —
*supercluster → cluster → subcluster* — narrowing the population one level at a
time toward the target (the analogue of the ALS T-cell project's chained
all-cells → immune → adaptive-immune classifiers).

**The governing constraint:** `rarecell` must stay **Colab-compatible** and must
**not download large reference datasets at runtime to build classifiers.** This
single requirement drives the architecture (§3).

### Prior art and the reference taxonomy

The **BICCN Human Brain Cell Atlas v1.0** (Siletti et al., *Science* 2023,
[doi:10.1126/science.add7046](https://doi.org/10.1126/science.add7046)) defines a
three-level whole-human-brain taxonomy: **31 superclusters → 461 clusters → 3,313
subclusters** (plus a neurotransmitter axis). The data is hosted on
[CZ CELLxGENE Discover](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443)
as 26 H5AD datasets split by brain region **and** by supercluster (e.g.
"Supercluster: Astrocyte", 155,025 cells). Crucially, the Discover H5ADs
**preserve the native taxonomy labels** in `obs`; the harmonized **Census** SOMA
does **not** (it flattens to Cell Ontology). So the build pipeline reads the
**Discover collection files**, not the Census query API.

`Astrocyte` is itself a **supercluster** (top level), so "isolate astrocytes from
all cells" is a **single supercluster-level decision** — deeper levels
(cluster/subcluster) are only needed when the target is a specific astrocyte
subtype. (The same DLPFC tissue underpins the PsychAD Class/Subclass/Subtype
atlas, an independent confirmation of the progressive paradigm.)

## 2. Goals and non-goals

**Goals**

- BICCN WHB v1 as the **single** reference taxonomy and label source for CNS.
- **Progressive** narrowing along supercluster → cluster → subcluster, to whatever
  depth the target sits at.
- **No large reference download at runtime.** All heavy BICCN work happens in a
  one-time **offline build step**; runtime loads only a small distilled bundle.
- Pre-trained **CellTypist models** as the primary per-level classifier, with
  **marker-gene panels** as a fallback / interpretability layer.
- Hard-subset narrowing (a level keeps only on-path cells), with full audit trail.

**Non-goals (v1)**

- Mouse / non-human, non-CNS tissues, spatial/multiome.
- Runtime training from raw reference data (an optional power-user "retrain" path
  may exist but is off the default runtime path).
- Census query API on the runtime path (used, if at all, only inside the offline
  build step). The earlier general Cell-Ontology/OLS4 backbone is dropped — the
  BICCN taxonomy tree is shipped in the bundle, so no ontology service is needed.
- **Subcluster level** (3,313 nodes) — deferred to a later release. v1 builds the
  **supercluster + cluster** layers only.

## 3. Core principle: separate "build the reference" from "use the reference"

```
┌─ OFFLINE, ONE-TIME (maintainer / CI) ───────────────┐     ┌─ RUNTIME (user, Colab) ───────────────┐
│ download BICCN WHB H5ADs from CELLxGENE Discover     │     │ load small bundle for target's path   │
│ → train CellTypist model per taxonomy decision       │ ==> │ → progressive hard-subset narrowing   │
│ → extract marker panel per node                      │     │   on the query (offline after fetch)  │
│ → emit small, versioned "CNS reference bundle"       │     │ → existing marker-driven fine isolation│
└──────────────────────────────────────────────────────┘     └────────────────────────────────────────┘
   heavy (GBs, network), happens once per BICCN release          light (MBs), happens every run
```

The BICCN reference is **fixed** (a published taxonomy), so classifiers are built
**once** and distilled into MB-scale artifacts. The bundle is **lazy per branch**:
runtime fetches only the artifacts along the path toward the requested target. For
the astrocyte demo that is a single supercluster-level model — a few MB.

## 4. Architecture

Two halves: an **offline build pipeline** (§4.1) and a **runtime subpackage**
(§4.2). They communicate only through the **CNS reference bundle** (§4.3).

### 4.1 Offline reference-build pipeline (`scripts/build_cns_reference.py`)

Maintainer/CI only — never invoked at user runtime. Its heavy dependencies
(Discover download, training) are isolated from the library's runtime deps.

The build trains **one multi-class model per parent node**: a single **31-way
supercluster** model at the root, and — for the cluster layer — **one multi-class
model per supercluster** over that supercluster's child clusters. (A full
multi-class model is reusable across any target sharing that parent, unlike
one-vs-rest.) Steps:

1. **Fetch** the BICCN WHB H5ADs from the CELLxGENE Discover collection via the
   Curation API (`api.cellxgene.cziscience.com/curation/v1/...`). Native taxonomy
   labels (supercluster/cluster/subcluster) are read from `obs`.
2. **Balanced subsample.** Down-sample the **entire** atlas to an approximately
   **equal number of cells per class** for the decision being trained — equal per
   supercluster for the root model; equal per child cluster for each
   supercluster's cluster model. Donor-stratified; cap per class at a fixed budget
   (e.g. ~5–10k cells/class); require ≥ `min_donors` per class. This is the central
   driver: it keeps training tractable and prevents abundance bias toward common
   classes.
3. **Normalize** to log1p-CP10K (matches the query pipeline's S2 output).
4. **Train** the multi-class CellTypist model (`celltypist.train`,
   feature_selection=True); validate on held-out donors; record metrics.
5. **Extract** each node's top marker genes (from the trained model / differential
   expression) into a per-node marker panel for the fallback layer.
6. **Emit** the model, per-node marker panels, and metadata into the bundle
   (§4.3), with content hashes and provenance (BICCN release, dataset_ids,
   per-class n_cells / n_donors).

v1 produces the **31-way supercluster** model plus a **cluster model for each
supercluster** (subcluster deferred). Reproducible and re-runnable when a new
BICCN release ships.

### 4.2 Runtime subpackage `rarecell.cns`

Gated by a light `[cns]` extra (`celltypist` for apply only; the marker-only path
needs no extra). Components, each a single-purpose, independently testable unit:

- **`cns/taxonomy.py` — TaxonomyTree.** Loads the shipped BICCN taxonomy tree
  (supercluster→cluster→subcluster edges + node metadata). `path_to(target) ->
  list[DecisionNode]` returns the ordered root→target decision path. No network,
  no ontology service.
- **`cns/bundle.py` — ReferenceBundle.** Resolves and loads the small per-branch
  artifacts (CellTypist models + marker panels + manifest) for a decision path;
  fetches lazily (GitHub release asset / hosted URL) and caches locally; verifies
  hashes against the manifest. Pins `reference_release`.
- **`cns/progressive.py` — ProgressiveApplier.** Applies the path to the query
  (post-QC, log1p-CP10K). At the root, run the **31-way supercluster** model and
  **hard-subset** to cells predicted as the target's supercluster; if the target
  is at cluster level, run that supercluster's **cluster model** on the survivors
  and subset to the target cluster. Cells predicted as the on-path class with
  confidence ≥ `min_confidence` are kept; other survivors are **dropped** (a
  conservative hard gate — per-cell marker rescue for low-confidence calls is
  deferred past v1). If the level's **model is unavailable** (missing file / sha
  mismatch) and `marker_fallback` is on, **fall back to the node's marker panel**
  scored via `score_genes`. Writes `obs["taxonomy_supercluster"]`,
  `obs["taxonomy_cluster"]` and records per-level retained counts + provenance in
  `adata.uns["cns_gate"]` (no silent loss).

### 4.3 CNS reference bundle format

A versioned, hash-manifested directory/tarball, organized by taxonomy node so a
branch can be fetched without the whole bundle:

```
cns-reference-<biccn_release>/
├── manifest.json                 # version, hashes, taxonomy edges, provenance
├── taxonomy.json                 # supercluster→cluster→subcluster tree + node metadata
└── nodes/
    ├── supercluster/
    │   ├── _decision.celltypist.pkl       # single 31-way supercluster model
    │   └── <node>.markers.json            # per-supercluster marker panel (fallback)
    ├── cluster/
    │   └── <supercluster>/
    │       ├── _decision.celltypist.pkl   # multi-class model over this supercluster's clusters
    │       └── <node>.markers.json
    └── subcluster/ …                      # deferred (later release)
```

v1 ships the **supercluster + cluster** layers; the supercluster model is global
(one file), and cluster models are per-supercluster (fetched lazily per branch, or
shipped in full since they are individually small). The subcluster layer is
produced later by re-running the build pipeline under the same format. Models are
MB-scale; marker panels are KB-scale; hosted as **GitHub release assets**.

## 5. Integration with the isolation pipeline

### 5.1 Profile schema (`profile/schema.py`)

Replace the earlier Cell-Ontology config with a BICCN-native block (default
disabled, so existing profiles are unaffected):

```python
class CNSTaxonomyConfig(BaseModel):
    enabled: bool = False
    target_node: str | None = None              # e.g. "Astrocyte"
    target_level: Literal["supercluster", "cluster"] = "supercluster"  # subcluster: later release
    reference_release: str | None = None         # pinned BICCN bundle version (GitHub release tag)
    min_confidence: float = 0.5                   # below → cell dropped (conservative gate)
    on_missing: Literal["marker_fallback", "skip"] = "marker_fallback"
```

### 5.2 New pipeline stage S2b — progressive class gate

Insert between S2 (QC + normalization) and S3 (cluster) in
`state_machine/isolate.py`:

- If `profile.cns_taxonomy.enabled`: resolve the path → load the bundle branch →
  `ProgressiveApplier.apply` → **hard-subset** the post-QC `adata` to target-node
  cells. Dropped cells + per-level provenance recorded in `uns["cns_gate"]`. The
  existing S3→S6 then run on the narrowed population — i.e. the fine isolation,
  reusing the `stage` scaffold in `core/clustering.py`
  (`stage="class"|"subclass"|"subtype"` mapped to supercluster/cluster/subcluster).
- If disabled, missing-and-`on_missing="skip"`, or fully degraded: pipeline runs
  exactly as today.

## 6. Error handling and degradation

| Failure | Behavior |
|---|---|
| `[cns]` extra / `celltypist` absent | Marker-panel fallback (no extra needed) or skip; logged. |
| Bundle unreachable / offline | If marker panels are bundled in-package: marker fallback. Else skip per `on_missing`. |
| Per-cell confidence < `min_confidence` | Cell dropped (conservative hard gate; per-cell marker rescue deferred past v1). |
| Model unavailable at a level (missing/sha mismatch) | Fall back to that node's marker panel for that level (when `marker_fallback`). |
| Train/query gene mismatch | Intersect features; CellTypist tolerates missing genes; warn on low overlap. |
| Reproducibility | Pin `reference_release`; manifest hashes + model hashes recorded in `manifest.json`. |

## 7. Dependencies and packaging

- **Runtime `[cns]` extra:** `celltypist` (apply only). Marker-only fallback uses
  the existing `scanpy` path — no new dependency. Core library unchanged.
- **Build pipeline (dev/CI only):** Discover download client (httpx), `celltypist`
  (train), `anndata`/`scanpy`. Declared as a separate `[build-reference]`/dev
  group, **never** on the user install path.
- **Bundle hosting:** **GitHub release assets**, fetched on demand and cached. The
  supercluster-layer marker panels are also **vendored in-package** so the
  marker-fallback path works with zero network.

### 7.1 Optional power-user retrain path

A documented, supported-but-off-the-default-path script lets advanced users
rebuild a bundle themselves — e.g. to train against a newer BICCN release, a
different tissue subset, or custom negative budgets. It reuses
`scripts/build_cns_reference.py` (the same offline pipeline, §4.1) and writes a
local bundle that the runtime can load via `reference_release="local:<path>"`.
This keeps full reproducibility/extensibility available without putting heavy
dependencies or downloads on the default user runtime. Documented in
`CONTRIBUTING.md` / a `docs/` how-to; not part of the Colab demo flow.

## 8. Testing strategy

- **TaxonomyTree (unit):** shipped tree fixture → assert `path_to("Astrocyte")`
  and deeper paths, and parent/child integrity.
- **ProgressiveApplier (unit):** synthetic query + mock per-level models → assert
  correct hard-subset narrowing; force a `None`/low-confidence level → assert
  marker-panel fallback path and provenance recording.
- **ReferenceBundle (unit):** mock hosted artifacts → assert lazy per-branch fetch,
  hash verification, and caching.
- **Build pipeline (unit):** mock Curation API + tiny synthetic H5ADs with native
  labels → run train → assert bundle layout, manifest hashes, held-out metric
  recording. (Network-gated live variant downloads one small real dataset.)
- **Integration (network-gated):** build a tiny real supercluster model → apply to
  one brainSCOPE sample → assert reasonable astrocyte precision/recall vs original
  annotations.
- **Determinism:** fixed seeds; identical inputs + pinned `reference_release` →
  identical masks and artifact hashes.

## 9. Demo integration (Colab)

Add a "Progressive BICCN class gate" section to `examples/colab_demo.ipynb`:
fetch the small supercluster bundle, apply the **astrocyte supercluster
classifier** to the merged SZ dataset, show how many cells pass the gate before
fine isolation, and plot the per-level composition. Colab pulls only the MB-scale
bundle — no atlas download. Frame against the BICCN supercluster→cluster→subcluster
paradigm.

## 10. Resolved design decisions

1. **Bundle hosting** — GitHub release assets; supercluster marker panels also
   vendored in-package for a zero-network marker fallback. (§7)
2. **Model shape** — a **full multi-class** model per parent node (one 31-way
   supercluster model; one per-supercluster cluster model), not one-vs-rest, so
   models are reusable across targets. The build step **down-samples the entire
   BICCN to ~equal cells per class** for each decision. (§4.1)
3. **v1 depth** — **supercluster + cluster**. Subcluster deferred to a later
   release. (§2, §4.1, §4.3)
4. **Power-user retrain path** — documented now as an optional off-default script
   reusing the build pipeline. (§7.1)
