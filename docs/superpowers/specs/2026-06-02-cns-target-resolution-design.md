# CNS Target Resolution & Characterization — Design

**Date:** 2026-06-02
**Status:** Draft for review
**Topic owner:** Patrick J. Reed

## 1. Motivation

The CNS reference bundle now describes 31 superclusters and 461 clusters richly
— each cluster carries a Siletti **class** (ASTRO/MGL/NEUR…), **curated marker
panel**, **neurotransmitter**, and (after a small addition) **top regions** —
plus per-level CellTypist classifiers. What's missing is the bridge between a
**query** (a paper → a drafted `TargetCellProfile`) and the **right place(s) in
the taxonomy**: *how is a cluster identified as relevant to the question?*

Queries come in three flavors that resolve differently:

1. **Named coarse type** ("astrocytes") → a supercluster / class.
2. **Discrete fine subtype** ("chandelier PVALB interneurons") → a specific
   cluster, by name + markers + class + region.
3. **A state or program, not a discrete cluster** ("**SNAP-expressing
   astrocytes**", "disease-associated microglia") → does *not* map to one node;
   the taxonomy gives the coarse container and the program is found by marker
   scoring *within* it.

The demo (Ling et al. SNAP astrocytes) is case 3. So the design must (a) decide,
per query, whether the target is a **node** to gate on or a **program** within a
container, and (b) in *both* cases **characterize** the isolated cells against
reference clusters. The resolution decision is made by a **hybrid**: a
deterministic retrieval scorer surfaces candidate nodes; the agent (Claude)
makes the node-vs-program judgment with a cited rationale.

## 2. Goals and non-goals

**Goals**
- Resolve a query to a target: `{mode, gate_node, gate_level, characterize_level}`
  with a cited, human-reviewable rationale.
- Deterministic node retrieval over the bundle (marker / class / region signals)
  to scale past 461 nodes and ground the agent.
- Characterize every isolated population against reference clusters + annotations.
- Handle the **node** and **program** paths with the same back-end.

**Non-goals (v1)**
- Subcluster-level classifiers (the bundle is supercluster + cluster; subcluster
  characterization falls back to cluster).
- Re-targeting / multi-target isolation (one resolved target per run).
- Replacing the existing marker-driven isolation — resolution *feeds* it.

## 3. Architecture & data flow

```
PAPER / PROMPT
  → draft_profile_from_prompt → TargetCellProfile (markers, lineage, tissue)   [existing]
  → TARGET RESOLUTION (new):
        1. retrieve: score_nodes(catalog, markers, lineage, tissue) → top-K candidates / level
        2. resolve:  Claude(candidates + paper) → TargetResolution (cited)
        → fills profile.cns_taxonomy { mode, gate_node, gate_level, characterize_level }
  → IsolateRunner:
        S2b CNS gate → narrow query to gate_node
        mode="node"    → gate is the isolation
        mode="program" → gate is the container; S3–S6 marker pipeline isolates within
  → CHARACTERIZE (new): classify isolated cells at characterize_level → reference
        clusters + ABC/Siletti annotations → summary
  → IsolationReport (+ target rationale + characterization table)
```

Three new units — **retrieval scorer**, **agent resolver**, **characterization**
— plus light wiring into the profile schema, drafting flow, and report.

## 4. Components

### 4.1 `rarecell.cns.retrieve` — deterministic node retrieval

Builds a searchable catalog from the bundle and ranks nodes against the query.
Pure-Python, no LLM, no network; operates on a local bundle directory.

```python
@dataclass
class NodeDescriptor:
    name: str            # "Astrocyte" | "Astro_52"
    level: str           # "supercluster" | "cluster"
    parent: str | None   # supercluster name for clusters
    cell_class: str      # Siletti class for clusters; the name itself for superclusters
    markers: list[str]   # superclusters: model panel; clusters: curated S3 panel
    regions: list[str]   # S3 top dissections
    neurotransmitter: str

@dataclass
class NodeMatch:
    node: NodeDescriptor
    score: float
    signals: dict[str, float]   # {"marker_overlap", "class_match", "region_match"}

def build_catalog(bundle_dir: Path) -> list[NodeDescriptor]
def score_nodes(
    catalog: list[NodeDescriptor], *, markers: list[str],
    lineage: str | None = None, tissue: str | None = None, top_k_per_level: int = 8,
) -> list[NodeMatch]
```

- **build_catalog** reads `taxonomy.json` (supercluster → clusters) and
  `annotations.json` (per cluster: class, markers, regions, neurotransmitter).
  Supercluster `markers` come from `nodes/supercluster/_markers.json`; cluster
  `markers` prefer the curated S3 panel from annotations, falling back to the
  cluster decision's `_markers.json`.
- **score_nodes** computes three normalized (0–1) signals per node:
  - `marker_overlap` — `|query_markers ∩ node.markers| / min(len(query), len(node))`
  - `class_match` — query `lineage` mapped to a class code vs `node.cell_class`
    (exact 1.0, related/substring 0.5, else 0.0; superclusters match on name).
  - `region_match` — query `tissue` token overlap with `node.regions` (0 when
    tissue unspecified, so it never penalizes).
  - `score = 0.5·marker_overlap + 0.3·class_match + 0.2·region_match` (weights are
    module constants). Returns the top `top_k_per_level` per level, each carrying
    its per-signal breakdown so the agent and a human see *why* it ranked.

### 4.2 `rarecell.agent.resolve_target` — agent decision (`[agent]` extra)

Given the drafted profile + the retrieval shortlist, Claude returns a structured,
cited decision via the existing Anthropic structured-output pattern (the same one
`draft_profile_from_prompt` uses; `respx`-mocked in tests). Only the top-K
candidates are placed in the prompt, so it scales independent of taxonomy size.

```python
class TargetResolution(BaseModel):
    mode: Literal["node", "program"]
    gate_node: str
    gate_level: Literal["supercluster", "cluster"]
    characterize_level: Literal["cluster", "subcluster"]
    rationale: str
    citations: list[str]
    confidence: float = Field(ge=0, le=1)

def resolve_target(
    profile: TargetCellProfile, *, candidates: list[NodeMatch],
    client: AnthropicClient, session: KnowledgeSession,
) -> TargetResolution
```

The system prompt frames the core judgment: *is the target a discrete reference
node (gate on it) or a gene program / state within a container (gate on the
container, isolate by markers)?* It must choose `gate_node` from the candidates
(or a candidate's parent supercluster), justify with citations to the paper's
language, and set `characterize_level` one level below the gate. The result
populates `profile.cns_taxonomy` and is reviewable/frozen with the profile.

### 4.3 `rarecell.cns.characterize` — deterministic post-isolation labeling

```python
@dataclass
class CharacterizationResult:
    per_cell_labels: pd.Series          # isolated cell → reference cluster label
    summary: list[dict]                 # per cluster: {fraction, n, class, neurotransmitter, regions, markers}

def characterize(
    isolated: ad.AnnData, bundle_dir: Path, *,
    level: Literal["cluster", "subcluster"], parent_node: str,
) -> CharacterizationResult
```

Loads the bundle classifier for `parent_node` at `level` (reusing the progressive
applier's model-load + `celltypist.annotate`), assigns each isolated cell to a
reference cluster, joins the ABC/Siletti annotations, and rolls up a per-cluster
summary sorted by fraction. `subcluster` falls back to `cluster` in v1 (no
subcluster models). Writes `isolated.uns["cns_characterization"]` and feeds the
report. Fully deterministic; unit-testable on the synthetic bundle.

## 5. Small build addition (regions)

The `region_match` signal needs **top regions/dissections per cluster**, trimmed
out of the vendored Table S3. Add `top_regions` (and `top_dissections`) back to
`scripts/build_cns_reference/data/siletti_table_s3.csv` and carry them into
`annotations.json` via `annotate_s3.build_s3_map` / `cluster_annotations`. One
new column group; no change to the build's control flow.

## 6. Wiring into existing code

- **`profile/schema.py`** — `CNSTaxonomyConfig` gains `mode:
  Literal["node","program"] = "node"` and `characterize_level:
  Literal["cluster","subcluster"] = "cluster"`. Optional `rationale: str | None`
  and `citations: list[str]` for auditability.
- **Drafting flow (`rarecell.agent.draft`)** — after drafting the profile, call
  `score_nodes` then `resolve_target`, and populate `profile.cns_taxonomy`. Gated
  by the presence of a bundle / `[agent]` extra; if resolution is skipped, the
  profile keeps `cns_taxonomy.enabled = False` (today's behavior).
- **`state_machine/isolate.py`** — after the final isolation, a `characterize`
  call (when `cns_taxonomy.enabled`), storing the result in `uns` and the report.
  The S2b gate is unchanged; `mode` drives `characterize_level` and intent. (A
  future optimization: `mode="node"` can short-circuit S3–S6, since the gate
  already selected the target. Out of scope for v1.)
- **`report.py`** — include the resolved target (mode, gate_node, rationale,
  citations, confidence) and the characterization summary table.
- **Demo** — show the cited target rationale and "your SNAP astrocytes map to
  Astro_52 (60%, AQP4+, frontal cortex), Astro_53 (30%)…".

## 7. Error handling & degradation

| Condition | Behavior |
|---|---|
| No bundle / resolution disabled | `cns_taxonomy.enabled=False`; pipeline runs as today (no gate, no characterization). |
| Retrieval finds no candidate above a floor | Resolver still runs with the best-available; if confidence is low, log and leave the gate disabled (marker-only isolation). |
| Agent (`[agent]`) absent | Resolution skipped; user may set `cns_taxonomy` manually. |
| `characterize` model missing for `parent_node` | Skip characterization with a logged warning; isolation still completes. |
| Determinism | Retrieval + characterization fixed by the pinned `reference_release`; resolver is recorded (rationale/citations) in the frozen profile. |

## 8. Testing

- **retrieve (unit):** synthetic catalog → assert ranking + per-signal scores for
  an identity query (marker overlap high) vs a program query (marker overlap low,
  class match high); `top_k_per_level` honored.
- **resolve_target (unit):** `respx`-mock the Anthropic call → assert a
  `program` decision (gate=container supercluster, characterize=cluster) for a
  SNAP-style profile and a `node` decision for a named-subtype profile.
- **characterize (unit):** synthetic bundle + isolated AnnData → assert per-cell
  labels, the summary fractions, and annotation join; subcluster→cluster fallback.
- **build addition (unit):** `annotate_s3` carries `top_regions` into the map.
- **integration (network-gated):** end-to-end on a brainSCOPE sample with the
  tiny bundle — resolve → gate → isolate → characterize produces a non-empty,
  annotated summary.

## 9. Scope / open questions

1. **Class-code mapping** — `lineage → class` (e.g. "astrocyte" → "ASTRO") needs a
   small mapping table or fuzzy match; confirm the lineage vocabulary the drafting
   flow emits.
2. **Where resolution lives** — inside `draft_profile_from_prompt` vs a separate
   `resolve_target` step the CLI/demo calls. (Recommend separate function, called
   by the drafting flow — single responsibility, independently testable.)
3. **`mode="node"` short-circuit** of S3–S6 — deferred to a follow-up.
