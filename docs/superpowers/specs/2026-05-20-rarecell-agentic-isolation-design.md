# rarecell — Agentic Rare-Cell Isolation Workflow (v1 Design)

- **Status:** Draft
- **Date:** 2026-05-20
- **Author:** Patrick Reed
- **Genesis:** Distilled from the ALS T-Cell Atlas project (`ALS_Project_Reedp1`) — specifically Phase 1 (per-dataset rare-cell isolation) of that pipeline's three-phase architecture.

## 1. Goal

Ship a pip-installable Python package, `rarecell`, that turns the rare-cell-isolation toolkit hidden inside `als_utils.py` into a generic, sharable, agentic workflow. A user hands the package a single AnnData with raw counts and a description of the rare population they're targeting; the package returns an isolated subset, a frozen profile artifact, and a fully auditable report. The agent is an *advisor*: it makes per-cluster recommendations, retrieves supporting literature, and explains decisions. Control flow is deterministic; reasoning is LLM-driven.

## 2. Non-goals (v1)

- **Cross-dataset integration.** Harmony across multiple datasets, scVI/scANVI, Scanorama, scIB benchmarking — all deferred to a future companion package.
- **Deep phenotyping.** STCAT, starCAT, cNMF, Hotspot, Milo, scCODA, CNA, scDRS, PyDESeq2, LIANA+, pseudobulk DE, TF activity inference — all deferred.
- **Dataset-specific metadata harmonization.** No `_resolve_dsXX` resolvers, no donor-metadata builders.
- **Bundled trained reference models.** The PsychAD MSSM CellTypist brain model is *not* shipped. Brain/CNS preset profiles document the gap and point users to model URLs. A companion workflow for building CellTypist classifiers from BICCN data on demand is captured as a follow-up.
- **Web UI.** Jupyter widgets cover interactive use; no Streamlit dashboard.
- **Conda packaging, Docker images.** PyPI + devcontainer only in v1.

## 3. Architecture

### 3.1 Layered package, three pip-installables in one monorepo

Three packages live under `packages/` in the repo, managed with `uv` workspaces. They share a release cycle (v0.x) and CI but install independently.

```
rarecell/                             # monorepo root
├── packages/
│   ├── rarecell/                     # library + bundled advisor agent
│   ├── rarecell-mcp/                 # exposed MCP server (workflow surface)
│   └── rarecell-mcp-knowledge/       # consumed MCP server (literature + markers)
├── docs/                             # mkdocs-material
├── examples/                         # example notebooks
├── tests/                            # cross-package integration tests
├── .github/workflows/
├── pyproject.toml                    # workspace root
├── uv.lock
├── LICENSE                           # Apache 2.0
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

### 3.2 `rarecell` (the library) — five layers

```
src/rarecell/
├── core/             # Pure science. No LLM imports.
│   ├── ingest.py     # AnnData validators + count restoration + symbol conversion
│   ├── qc.py         # Configurable QC (immune-sensitive defaults) + Scrublet
│   ├── markers.py    # score_panel + profile-driven panel/negative scoring
│   ├── clustering.py # taxonomy_cluster (HVGs → PCA → in-dataset Harmony → silhouette Leiden)
│   ├── annotate.py   # CellTypist (profile-driven) + Enrichr + consensus labels
│   ├── evidence.py   # BICCN trinarization + multi-evidence consensus tables
│   ├── purify.py     # subcluster_and_purify
│   └── io.py         # h5ad sanitization, checkpoint save/load, figure dir setup
├── profile/          # The target-cell profile system
│   ├── schema.py     # pydantic TargetCellProfile
│   ├── presets/      # t_cell_cns.yaml, t_cell_pbmc.yaml, b_cell.yaml,
│   │                 # nk_cell.yaml, microglia.yaml, dendritic_cell.yaml,
│   │                 # monocyte_macrophage.yaml
│   └── draft.py      # NL prompt → draft profile (uses RAG)
├── rag/              # Retrieval abstractions
│   ├── base.py       # Retriever protocol + RetrievalHit / Citation models
│   ├── literature.py # MCP client → rarecell-mcp-knowledge
│   ├── markers_db.py # MCP client → rarecell-mcp-knowledge
│   └── cache.py      # Local SQLite cache at ~/.cache/rarecell/rag.sqlite
├── agent/            # The advisor agent
│   ├── system_prompt.md
│   ├── tools.py      # Read-only + recommendation tools (no side-effecting core fns)
│   ├── modes.py      # State machines: profile_draft | isolate | review
│   ├── basic.py      # BasicRecommender fallback (no LLM)
│   └── loop.py       # Claude Agent SDK driver
├── notebook/         # Jupyter widgets + magics
│   ├── magics.py     # %%rarecell draft | isolate | review
│   └── widgets.py    # IsolateWidget (gate dialogs, evidence table renderer)
├── cli.py            # Typer CLI
└── report.py         # IsolationReport (manifest + decisions + bibliography)
```

**Layering rule:** `core` knows nothing about LLMs or MCP. `profile` depends only on `core`. `rag` is standalone. `agent` depends on `core` + `profile` + `rag`. `notebook` and `cli` are thin front-ends over `agent`. Anyone can `pip install rarecell` and use `rarecell.core` from vanilla scanpy with no LLM at all.

### 3.3 `rarecell-mcp` — exposed workflow surface

FastMCP server that wraps the library's high-level workflows, not its primitives. Four tools and two resource types:

```
Tools:
  draft_profile(prompt, output_path)
  validate_input(adata_path)
  run_isolation(input_path, profile_path, output_dir, auto_policy?)
  inspect_report(report_path, question?)

Resources:
  rarecell://presets/{name}.yaml
  rarecell://reports/{run_id}/...
```

Critically, raw clustering / QC / scoring tools are **not** exposed. Letting external LLM clients call those directly would bypass the state machine and produce unauditable outputs. Reproducibility is non-negotiable.

### 3.4 `rarecell-mcp-knowledge` — consumed by the agent

FastMCP server aggregating two knowledge sources behind one MCP surface:

```
Tools:
  search_literature(query, year_range?, tissue?)  → list[Citation+Snippet]
  fetch_abstract(pmid_or_doi)                     → full abstract
  search_markers(cell_type, tissue?)              → marker hits with citations
  get_canonical_panel(name)                       → curated marker panel
  enrichr_enrich(genes, library)                  → enrichment results
```

Backends:
- **Literature:** evaluates community PubMed/Europe PMC MCP servers; falls back to a thin in-repo Europe PMC REST wrapper if no community option meets a small bar (license, schema stability, no auth, full abstracts, maintained in the last 12 months). The evaluation criteria are documented in the docs.
- **Markers:** local CellMarker 2.0 + PanglaoDB SQLite (~50 MB, downloaded on first run); live MSigDB REST; live Enrichr REST.

### 3.5 Install footprint

- `pip install rarecell` — library + core only. No LLM dependencies.
- `pip install rarecell[agent]` — library + advisor agent + `rarecell-mcp-knowledge` for RAG.
- `pip install rarecell-mcp` — standalone exposed server for Claude Desktop / Claude Code / Cursor users.

## 4. The target-cell profile

### 4.1 Why it exists

The profile is the **frozen reproducibility artifact**. It encodes everything the workflow needs to know about what "rare population" means in this run: positive markers, negative/exclusion markers, reference-label patterns, BICCN trinarization rules, QC parameters, expected abundance. Once frozen (with `frozen: true` and a content hash), it is immutable; downstream artifacts pin to its hash.

### 4.2 Schema (pydantic, serialized to YAML)

Key fields:

- **Identity:** `name`, `description`, `target_lineage`, `tissue`, `expected_abundance` (min/max fraction with rationale).
- **Positive markers:** named panels with gene lists, z-score thresholds, citations.
- **Negative markers:** non-target panels (neuron, astrocyte, etc.) with exclusion thresholds.
- **Reference labels:** which CellTypist models to score; match patterns per model; ability to disable models per profile.
- **BICCN rules:** which BICCN class/subclass cell types to trinarize against.
- **QC overrides:** `min_genes_per_cell`, `max_pct_mt`, etc. — defaulted from `target_lineage` (lymphoid/myeloid → immune-sensitive permissive floor; neuronal → standard).
- **Purify controls:** enable/disable, high-resolution value, min-cluster-purity gate.
- **Batch correction:** `in_dataset: "harmony" | "none"`; batch key (default `sample_id`). Never cross-dataset.
- **Run controls:** auto-policy per gate, budget ceiling, model selection.
- **Provenance:** `drafted_from` (user prompt, drafting model, RAG sources consulted), `drafted_at`, `human_reviewed`, `reviewer` email.
- **Integrity:** `schema_version`, `profile_id`, `content_hash` (SHA-256 of canonical YAML), `frozen` boolean.

### 4.3 The freeze gate

**Hard rule:** `frozen: true` requires `human_reviewed: true`. The pydantic validator raises `UnreviewedProfileError` if `frozen=true` and `human_reviewed=false`. The agent cannot ship a profile to disk in frozen state without a human signoff. This is the most important interlock in the system: it prevents the agent from running away with marker panels that were never inspected by the scientist.

### 4.4 Drafting flow

1. User provides a natural-language prompt: *"rare exhausted CD8+ T-cells in postmortem ALS spinal cord."*
2. Agent decomposes into a structured query (target, tissue, lineage).
3. Parallel RAG retrievals: `search_literature` + `search_markers` via `rarecell-mcp-knowledge`.
4. Agent drafts a YAML profile with: positive panels from marker DB hits, sanity-checked against literature; negative panels seeded from tissue presets; reference labels suggested from the CellTypist registry; QC defaults from `target_lineage`; `expected_abundance` from literature retrievals; citations attached to every panel.
5. Notebook widget (or CLI prompt) renders the draft for review. User accepts, edits, or rejects.
6. On accept: validator computes `content_hash`, sets `frozen: true`, writes `profile.yaml` to disk.

### 4.5 Preset library shipped with v1

`t_cell_cns.yaml`, `t_cell_pbmc.yaml`, `b_cell.yaml`, `nk_cell.yaml`, `microglia.yaml`, `dendritic_cell.yaml`, `monocyte_macrophage.yaml`. CNS-tissue presets document the missing-brain-model gap and offer two options: download the PsychAD model from a documented URL, or run with only the PBMC-trained `Immune_All_Low.pkl` (which loses brain context but still works).

## 5. The advisor workflow

### 5.1 Control flow is deterministic; LLM provides reasoning

The state machine is plain Python. The LLM's role at each state is to recommend per-cluster decisions, retrieve justifying citations, and answer user questions. The LLM never decides which state runs next, never skips QC, never changes clustering resolution mid-run. This is the central reproducibility commitment.

### 5.2 The `isolate` state machine

```
S0  Load profile + AnnData
S1  Ingest & validate           (validate_counts, symbol conversion)
S2  QC + Scrublet               (params from profile)
S3  Cluster + annotate          (taxonomy_cluster, all profile CellTypist models,
                                 panel + negative + BICCN scoring)
S4  Consensus table              ── GATE 1: per-cluster keep/drop/purify decisions
S5a Purify flagged clusters     (subcluster_and_purify, re-score evidence)
                                 ── GATE 2: sub-cluster keep/drop decisions
S5b Skip if none flagged
S6  Final isolation              (optional re-cluster at finer resolution)
                                 ── GATE 3: accept output, iterate, or abort
S7  Write IsolationReport
```

Three hard gates. Each gate has:
- Interactive mode: prompts the user.
- Auto mode: applies the profile's `auto_policy` (defaults: accept agent recommendation; abort if recommendation confidence < 0.6 or final abundance > 5× the profile's expected bounds).
- Logging: both the agent's recommendation and the user's (or auto-policy's) decision are written to `decisions.jsonl`.

### 5.3 What the agent does at each gate

For each ambiguous cluster the agent emits a structured recommendation:

```json
{
  "cluster_id": "7",
  "recommendation": "purify",
  "confidence": 0.62,
  "evidence_summary": {
    "ptprc_score": 0.04, "pan_t_cell_score": 1.8,
    "biccn_tcell_prob": 0.71, "biccn_neuron_prob": 0.18,
    "celltypist_immune_pct_tcell": 0.58,
    "negative_contamination": "neuron"
  },
  "reasoning": "T-cell evidence is strong but neuron contamination present...",
  "citations": ["pmid:32747940", "cellmarker:T_cell:brain"]
}
```

The widget (or CLI table) renders this alongside the consensus-table row. The user accepts, overrides, or asks the agent a follow-up question.

### 5.4 The agent's tool surface — deliberately narrow

```
Read-only tools:
  get_consensus_table()              → DataFrame
  get_cluster_evidence(cluster)      → dict
  search_literature(query)           → list[Citation]
  search_markers_db(query)           → list[MarkerHit]

Side-effecting tools:
  present_recommendation(rec, options) → user_decision
  log_decision(cluster, rec, user_decision, reasoning)
```

The agent does **not** have direct access to `taxonomy_cluster`, `run_qc`, `score_panel`, or any other primitive that would let it bypass the state machine. Same reasoning as the exposed MCP server.

### 5.5 The `BasicRecommender` fallback

If the user installs `rarecell` without the `[agent]` extra (or runs offline without API access), the state machine still works. A pure-heuristic `BasicRecommender` produces recommendations from the consensus-table evidence using fixed thresholds (e.g., "keep if PTPRC > 1.0 AND pan_t_cell > 1.0 AND no negative panel > 1.5"). Manifest records `degraded_mode: true`.

### 5.6 Agent modes

- **`profile_draft`** — NL → RAG → draft YAML → user freezes. (Section 4.4.)
- **`isolate`** — the state machine above.
- **`review`** — takes an `IsolationReport` directory, replays each decision, surfaces anomalies. Useful for sharing runs with collaborators and self-audit before publication.

## 6. RAG and citation propagation

### 6.1 Retriever abstraction

```python
class Retriever(Protocol):
    def search(self, query: str, **kwargs) -> list[RetrievalHit]: ...

class RetrievalHit(BaseModel):
    citation: Citation       # PMID, DOI, CellMarker ID, etc.
    title: str
    snippet: str
    payload: dict            # structured payload (marker list, etc.)
    retrieved_at: datetime
    source: Literal["europepmc", "cellmarker", "panglaodb",
                    "msigdb", "enrichr"]
```

Two concrete retrievers in `rarecell.rag`: `LiteratureRetriever` and `MarkersDBRetriever`. Both speak MCP to `rarecell-mcp-knowledge`. Both are wrapped by `LocalCacheRetriever` (SQLite cache, 30-day default TTL).

### 6.2 Citation propagation

```
RAG retrieval (Citation)
   │
   ▼
Profile.positive_markers[panel].citations   ← attached during drafting
   │
   ▼
IsolationReport.manifest.profile_provenance.rag_sources
IsolationReport.manifest.decisions[i].supporting_citations
   │
   ▼
report.bibliography.bib   ← BibTeX file generated at end of run
```

Every panel and every gate decision points back to specific citations. The report ships with a BibTeX file ready for inclusion in a paper.

### 6.3 Offline and degraded modes

- **Offline:** if `rarecell-mcp-knowledge` is unreachable, the agent runs against cached hits only. Profile drafting *from scratch* fails cleanly; isolation against a frozen profile still works (the profile already has its citations baked in).
- **No LLM:** `BasicRecommender` runs heuristic-only; manifest marks `degraded_mode: true`.

## 7. Front-ends

### 7.1 Three front-ends, one backend (`rarecell.agent`)

**Jupyter** — `%load_ext rarecell` registers cell magics:

```python
%%rarecell draft
"""Rare exhausted CD8+ T-cells in postmortem ALS spinal cord."""

%%rarecell isolate
input: adata
profile: profile.yaml
```

`IsolateWidget` renders the consensus table with conditional formatting (the same color-coded `plot_consensus_evidence_table` semantics from `als_utils`), one ipywidgets button per gate decision, agent reasoning streaming in a side panel.

**CLI** — `rarecell` via Typer:

```
rarecell draft   --prompt "rare T cells in postmortem ALS brain" \
                 --out profile.yaml --interactive
rarecell isolate --input adata.h5ad --profile profile.yaml \
                 --out-dir runs/run_001 --auto-policy recommendation \
                 --budget-usd 5
rarecell review  --report runs/run_001
```

**MCP server** — `rarecell-mcp`, four tools (Section 3.3). Same backend.

### 7.2 The `IsolationReport` artifact

A directory, portable and replay-able:

```
runs/run_001/
├── manifest.json         # Schema-validated authoritative record
├── profile.yaml          # Frozen profile (with content_hash)
├── input.h5ad.sha256
├── isolated.h5ad         # The output
├── decisions.jsonl       # One JSON object per gate decision (append-only)
├── figures/              # QC, UMAPs, consensus tables, BICCN dotplots, etc.
├── bibliography.bib      # Every cited paper/marker DB entry
├── run.log               # structlog JSON
└── replay.sh             # Bash script to reproduce
```

`manifest.json` captures: schema version, run ID, timestamps, `rarecell` version + commit, profile content hash, input hash, dependency versions, agent model + budget used, input/QC/isolated summaries, RAG sources used, decision counts per gate.

`decisions.jsonl` is append-only JSONL — streams cleanly during long runs, grep-able after.

### 7.3 Replay mode

```bash
cd runs/run_001
bash replay.sh
# → rarecell isolate --input <resolved> --profile profile.yaml \
#                    --out-dir replay/ --auto-policy from_decisions \
#                    --decisions decisions.jsonl
```

`--auto-policy from_decisions` replays the previous run's gate decisions verbatim — no user prompts, no agent calls. Same input + same profile + same decisions = byte-identical `isolated.h5ad` (modulo timestamps). Doubles as the regression test for any refactor (Section 8).

## 8. Testing

### 8.1 Five test categories

1. **Unit tests** — per `core/` module, pydantic validators, RAG retrievers (MCP mocked), citation propagation, `IsolationReport` round-trip. Coverage target: 80%+ on `core/`, `profile/`, `report.py`.
2. **Integration tests (synthetic)** — `tests/fixtures/make_synthetic.py` generates a 5,000-cell AnnData with 4 clusters and one planted ~5%-rare population. Every gate decision has a known correct answer; tests assert the agent's recommendation matches.
3. **Integration tests (public)** — small public dataset (10x PBMC 3k or CELLxGENE subset, ~10 MB). Cached on first test run.
4. **Replay determinism** — runs isolation against the synthetic fixture twice with same input + profile + recorded decisions; asserts byte-identical `isolated.h5ad`. This is the regression test of record.
5. **MCP smoke tests** — start `rarecell-mcp` and `rarecell-mcp-knowledge` as subprocesses; call every advertised tool; validate response schemas.
6. **Property tests (hypothesis)** — profile YAMLs round-trip through pydantic without loss; manifest hashes stable.

### 8.2 Explicitly out of scope

Full ALS atlas reproduction is **not** a CI artifact. Tests verify the package, not the science. The ALS pipeline becomes one of the example notebooks.

## 9. Errors

Single base class `RareCellError`. Three categories:

- **User-input** — `MissingRawCountsError`, `InvalidProfileError`, `UnreviewedProfileError`, `IncompatibleSchemaError`. Raised with actionable messages. Recoverable.
- **Runtime** — `MCPUnreachableError`, `LLMBudgetExceededError`, `CacheCorruptedError`. Have explicit fallbacks (cache, BasicRecommender, retry-with-backoff).
- **Catastrophic** — `IsolationAbortedError`. Catches OOM, disk full, corrupted h5ad. Partial run is saved with `manifest.status = "failed"` for inspection.

Resumption: `rarecell isolate --resume runs/run_001` picks up from the last completed state, useful after `LLMBudgetExceededError`.

## 10. Governance

- **License:** Apache 2.0 (explicit patent grant; matters for academic-industry collaborations on therapeutic-target discovery).
- **Repo:** monorepo, three packages under `packages/`, managed with `uv` workspaces.
- **CI (GitHub Actions):** ruff lint + format check, mypy on `core/`/`profile/`/`agent/`, pytest unit tests on Python 3.11 + 3.12, integration tests gated on `pull_request` only (CI minutes), MCP smoke tests, mkdocs build, wheel build + install smoke.
- **Pre-commit:** ruff, ruff format, mypy (loose), `pytest --testmon`.
- **Docs (mkdocs-material):** quickstart, profile authoring guide, "adapting to your tissue" tutorial, agent advisor walkthrough, reproducibility & sharing reports, per-module API reference (mkdocstrings), "evaluating a community PubMed MCP server" appendix.
- **Versioning:** SemVer. v0.1.0 = first preview; v1.0.0 when the public API stabilizes. Profile `schema_version` is versioned independently with an auto-migration path.
- **Community files:** Contributor Covenant CoC, conventional-commits CHANGELOG, issue/PR templates, SECURITY.md.

## 11. Open follow-ups (deliberately deferred)

1. **CellTypist-from-BICCN companion workflow.** Build CellTypist classifiers from BICCN reference data on demand for any user-defined target cell type / tissue / hierarchy level. Removes the biggest "out-of-the-box doesn't work" footgun for non-PBMC tissues. (Tracked in `~/.claude/.../memory/rarecell_celltypist_from_biccn.md`.)
2. **Cross-dataset integration package.** Lifts Phase 2 of the ALS pipeline (sweep-optimized Harmony, scVI/scANVI, scIB benchmarking) into a sibling package consuming `IsolationReport` outputs.
3. **Deep phenotyping package.** Lifts Phase 3 (Milo, scCODA, scDRS, PyDESeq2, LIANA+, cNMF, Hotspot, etc.).
4. **Conda packaging + Docker images.** Post-v1.
5. **`prior-runs RAG`** — let the agent retrieve over decisions made in prior `IsolationReport` runs as institutional memory.

## 12. Success criteria for v1

- A scientist who has never seen `rarecell` can:
  - `pip install rarecell[agent]`,
  - run `rarecell draft --prompt "..."` and produce a frozen, human-reviewed profile,
  - run `rarecell isolate --input my.h5ad --profile profile.yaml`,
  - receive an `IsolationReport` with an `isolated.h5ad`, a manifest, figures, and a BibTeX file,
  - share the report directory with a collaborator who can `bash replay.sh` to reproduce.
- The synthetic-fixture replay test is byte-deterministic in CI.
- The CNS T-cell example notebook reproduces a sensible T-cell subset on a small public ALS or PBMC dataset within a $5 LLM budget.
- A user with Claude Desktop can install `rarecell-mcp`, configure it, and drive an isolation end-to-end from the desktop client.
