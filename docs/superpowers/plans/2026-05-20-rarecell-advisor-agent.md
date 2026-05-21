# rarecell Advisor Agent — Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a Claude-backed advisor agent into `rarecell` — `ClaudeRecommender` (LLM-backed implementation of the Recommender protocol), `rarecell.rag` retrievers that consume `rarecell-mcp-knowledge` for citations, profile drafting from natural-language prompts, and `review` mode for post-hoc audits. Also fix the Plan 1 deferral: wire `S5_GATE2` and `S6_GATE3` into `IsolateRunner`.

**Architecture:** `rarecell.rag` retrievers wrap `rarecell-mcp-knowledge` in-process (call `KnowledgeApp` tool methods directly, no MCP transport). `ClaudeRecommender` uses the Anthropic SDK with structured output (JSON schema) — narrow, deterministic-shape calls, no agentic tool-use loop. Profile drafting uses an Anthropic SDK call with literature + marker RAG hits injected into the prompt context. Tests use `respx` to mock Anthropic HTTP calls. The `[agent]` optional extra gates installation of `anthropic` so the core library stays LLM-free.

**Tech Stack:** Python 3.11+, anthropic SDK, existing `rarecell` and `rarecell-mcp-knowledge` packages, pytest, respx, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-rarecell-agentic-isolation-design.md` §5–§6.

**Plan 1 + Plan 2 status:** Both merged to main. 81 tests passing on main.

---

## File Structure

```
packages/rarecell/src/rarecell/
├── rag/                              # NEW (this plan)
│   ├── __init__.py
│   ├── base.py                       # Retriever protocol + RetrievalHit re-export
│   ├── knowledge.py                  # In-process wrapper around KnowledgeApp
│   ├── literature.py                 # LiteratureRetriever (search_literature, fetch_abstract)
│   └── markers_db.py                 # MarkersDBRetriever (search_markers, get_canonical_panel)
├── agent/                            # NEW (this plan)
│   ├── __init__.py
│   ├── system_prompt.md              # Domain-tuned system prompt
│   ├── client.py                     # Anthropic client wrapper + AnthropicProtocol
│   ├── recommender.py                # ClaudeRecommender — structured-output recommendations
│   └── draft.py                      # NL → draft TargetCellProfile via RAG + Claude
├── profile/
│   └── draft.py                      # → MODIFIED: thin re-export of agent/draft.py
├── recommender/
│   └── __init__.py                   # → MODIFIED: re-export ClaudeRecommender
└── state_machine/
    └── isolate.py                    # → MODIFIED: wire S5_GATE2, S6_GATE3, abundance abort
```

Tests:

```
packages/rarecell/tests/
├── rag/
│   ├── test_knowledge.py
│   ├── test_literature.py
│   └── test_markers_db.py
├── agent/
│   ├── test_client.py
│   ├── test_recommender.py
│   ├── test_draft.py
│   └── fixtures/
│       └── anthropic_responses.py    # canned structured JSON responses
└── state_machine/
    └── test_isolate_gates.py         # gate-2/gate-3 wiring tests
```

**Boundaries:**
- `rag/` is independent of `agent/` — it just returns `RetrievalHit`s. Could be used without an LLM.
- `agent/` is independent of `state_machine/isolate.py` — `ClaudeRecommender` implements the existing `Recommender` Protocol, so wiring into `IsolateRunner` is a constructor-arg change at the user site (or in a default-factory helper).
- All Anthropic HTTP is mocked via `respx` in tests — never live.
- `[agent]` extra in `packages/rarecell/pyproject.toml` gates installation of `anthropic`. Without it, importing `rarecell.agent.*` raises ImportError; `rarecell.core` and `rarecell.recommender.BasicRecommender` keep working.

---

## Task 1: Add `[agent]` extra with anthropic dep

**Files:**
- Modify: `packages/rarecell/pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml` for the rarecell package**

Use the Read tool on `packages/rarecell/pyproject.toml`. Locate the `[project.optional-dependencies]` block — it currently has `agent = []`.

- [ ] **Step 2: Add `anthropic` to the `agent` extra**

Edit the `agent` line:

```toml
agent = [
  "anthropic>=0.39",
  "rarecell-mcp-knowledge",
]
```

The `rarecell-mcp-knowledge` dep is a workspace member, so uv will resolve it from the local source. This makes `pip install rarecell[agent]` pull in both anthropic AND the knowledge MCP server package.

- [ ] **Step 3: Sync**

```bash
uv sync --all-packages --all-extras --dev
```

Expected: anthropic resolves; no errors.

- [ ] **Step 4: Smoke-import**

```bash
uv run python -c "import anthropic; print(anthropic.__version__)"
```
Expected: prints an anthropic version >= 0.39.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/pyproject.toml uv.lock
git commit -m "Add [agent] extra with anthropic + rarecell-mcp-knowledge"
```

---

## Task 2: `rag/base.py` — Retriever protocol + re-exports

**Files:**
- Create: `packages/rarecell/src/rarecell/rag/__init__.py`
- Create: `packages/rarecell/src/rarecell/rag/base.py`
- Create: `packages/rarecell/tests/rag/__init__.py` (empty)
- Create: `packages/rarecell/tests/rag/test_base.py`

The `rag` package's public surface is small: a `Retriever` Protocol and a re-export of `RetrievalHit` and `Citation` from `rarecell_mcp_knowledge.citation`. (Yes, the rag package depends on the knowledge package; that's intentional — both ship under the `[agent]` extra.)

- [ ] **Step 1: Write the failing test**

`packages/rarecell/tests/rag/test_base.py`:

```python
from rarecell.rag.base import Retriever
from rarecell.rag import Citation, RetrievalHit


def test_protocol_importable():
    assert Retriever is not None


def test_citation_reexported_from_knowledge_pkg():
    c = Citation(source_id="pmid:1", source="europepmc")
    assert c.source == "europepmc"


def test_retrieval_hit_reexported():
    c = Citation(source_id="pmid:1", source="europepmc")
    h = RetrievalHit(citation=c, title="t", snippet="s", payload={}, source="europepmc")
    assert h.title == "t"
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell/tests/rag/test_base.py -v`

- [ ] **Step 3: Write `base.py`**

`packages/rarecell/src/rarecell/rag/base.py`:

```python
"""Retriever protocol for the rag layer."""
from __future__ import annotations
from typing import Protocol


class Retriever(Protocol):
    """Anything that returns RetrievalHits given a query.

    Concrete retrievers in rarecell.rag adapt rarecell-mcp-knowledge's
    KnowledgeApp surface to this interface.
    """

    def search(self, query: str, **kwargs) -> list: ...
```

- [ ] **Step 4: Write `rag/__init__.py`**

```python
"""rarecell.rag — retrieval-augmented context for the advisor agent."""
from rarecell.rag.base import Retriever
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit

__all__ = ["Retriever", "Citation", "RetrievalHit"]
```

- [ ] **Step 5: Run, expect 3 pass**

Run: `uv run pytest packages/rarecell/tests/rag/test_base.py -v`

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/rag/ packages/rarecell/tests/rag/
git add packages/rarecell/src/rarecell/rag/ packages/rarecell/tests/rag/
git commit -m "Add rarecell.rag.base Retriever protocol + Citation re-exports"
```

---

## Task 3: `rag/knowledge.py` — in-process wrapper around KnowledgeApp

**Files:**
- Create: `packages/rarecell/src/rarecell/rag/knowledge.py`
- Create: `packages/rarecell/tests/rag/test_knowledge.py`

This wraps `rarecell_mcp_knowledge.server.build_app(...)` so other retrievers can share a single in-process `KnowledgeApp`. No MCP transport.

- [ ] **Step 1: Write the failing test**

`packages/rarecell/tests/rag/test_knowledge.py`:

```python
from pathlib import Path
from rarecell.rag.knowledge import build_knowledge_session
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv


PLAN2_FIXTURES = (
    Path(__file__).resolve().parents[5]
    / "packages/rarecell-mcp-knowledge/tests/data"
)


def test_session_exposes_tool_map(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    names = sorted(session.tool_names)
    assert "search_literature" in names
    assert "search_markers" in names
    assert "get_canonical_panel" in names


def test_session_search_markers_returns_hits(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    hits = session.call("search_markers", {"cell_type": "T cell", "tissue": "blood"})
    all_genes = {g for h in hits for g in h["payload"]["genes"]}
    assert "CD3D" in all_genes
```

The `parents[5]` lookup: test file is at `packages/rarecell/tests/rag/test_knowledge.py`. parents[0]=rag, parents[1]=tests, parents[2]=rarecell, parents[3]=packages, parents[4]=worktree-root. So parents[5] would be too high — adjust to `parents[4]`. Verify by printing PLAN2_FIXTURES in the test once before running; it must resolve to `<repo-root>/packages/rarecell-mcp-knowledge/tests/data/`.

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Write `knowledge.py`**

```python
"""In-process wrapper around rarecell-mcp-knowledge's KnowledgeApp."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.server import build_app


@dataclass
class KnowledgeSession:
    """Holds a single KnowledgeApp + tool_map. Reuse across retrievers."""
    _app: Any
    catalog_path: Path
    cache_path: Path

    @property
    def tool_names(self) -> list[str]:
        return self._app.list_tool_names()

    def call(self, name: str, kwargs: dict) -> Any:
        return self._app.call_tool(name, kwargs)


def build_knowledge_session(
    *, catalog_path: Path, cache_path: Path,
) -> KnowledgeSession:
    """Build a KnowledgeSession wrapping a KnowledgeApp."""
    catalog = MarkersCatalog(catalog_path)
    app = build_app(catalog=catalog, cache_path=cache_path)
    return KnowledgeSession(_app=app, catalog_path=catalog_path,
                             cache_path=cache_path)
```

- [ ] **Step 4: Run, expect 2 pass**

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/rag/ packages/rarecell/tests/rag/
git add packages/rarecell/src/rarecell/rag/knowledge.py packages/rarecell/tests/rag/test_knowledge.py
git commit -m "Add KnowledgeSession in-process wrapper around KnowledgeApp"
```

---

## Task 4: `rag/literature.py` — LiteratureRetriever

**Files:**
- Create: `packages/rarecell/src/rarecell/rag/literature.py`
- Create: `packages/rarecell/tests/rag/test_literature.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import respx
from httpx import Response
from rarecell.rag.knowledge import build_knowledge_session
from rarecell.rag.literature import LiteratureRetriever


@respx.mock
def test_literature_retriever_returns_hits(tmp_path):
    respx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ).mock(return_value=Response(200, json={
        "hitCount": 1,
        "resultList": {"result": [{
            "id": "12345", "pmid": "12345",
            "title": "T cell markers in brain",
            "abstractText": "Pan-T markers...",
            "doi": "10.1234/abc", "pubYear": "2024",
            "authorString": "X",
        }]},
    }))

    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite",
        cache_path=tmp_path / "c.sqlite",
    )
    retriever = LiteratureRetriever(session=session)
    hits = retriever.search("T cell brain")
    assert len(hits) == 1
    assert hits[0].citation.source_id == "pmid:12345"
    assert hits[0].source == "europepmc"


def test_literature_retriever_fetch_abstract(tmp_path):
    import respx
    from httpx import Response
    with respx.mock:
        respx.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        ).mock(return_value=Response(200, json={
            "resultList": {"result": [{
                "id": "111", "pmid": "111", "title": "T cell",
                "abstractText": "abstract", "doi": "10.1/x", "pubYear": "2024",
                "authorString": "X",
            }]},
        }))
        session = build_knowledge_session(
            catalog_path=tmp_path / "m.sqlite",
            cache_path=tmp_path / "c.sqlite",
        )
        hit = LiteratureRetriever(session=session).fetch_abstract("111")
        assert hit.citation.source_id == "pmid:111"
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Write `literature.py`**

```python
"""LiteratureRetriever — adapts KnowledgeSession to a Retriever-shaped API."""
from __future__ import annotations
from rarecell.rag.knowledge import KnowledgeSession
from rarecell_mcp_knowledge.citation import RetrievalHit


class LiteratureRetriever:
    """Wraps KnowledgeSession's `search_literature` and `fetch_abstract` tools."""

    def __init__(self, session: KnowledgeSession):
        self.session = session

    def search(
        self, query: str, *,
        year_range: tuple[int, int] | None = None,
        tissue: str | None = None, page_size: int = 10,
    ) -> list[RetrievalHit]:
        kwargs: dict = {"query": query, "page_size": page_size}
        if year_range is not None:
            kwargs["year_range"] = list(year_range)
        if tissue is not None:
            kwargs["tissue"] = tissue
        raw_hits = self.session.call("search_literature", kwargs)
        return [RetrievalHit.model_validate(h) for h in raw_hits]

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit:
        raw = self.session.call("fetch_abstract", {"pmid_or_doi": pmid_or_doi})
        return RetrievalHit.model_validate(raw)
```

- [ ] **Step 4: Run, expect 2 pass**

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/rag/literature.py packages/rarecell/tests/rag/test_literature.py
git commit -m "Add LiteratureRetriever over KnowledgeSession"
```

---

## Task 5: `rag/markers_db.py` — MarkersDBRetriever

**Files:**
- Create: `packages/rarecell/src/rarecell/rag/markers_db.py`
- Create: `packages/rarecell/tests/rag/test_markers_db.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from rarecell.rag.knowledge import build_knowledge_session
from rarecell.rag.markers_db import MarkersDBRetriever
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

PLAN2_FIXTURES = (
    Path(__file__).resolve().parents[4]
    / "packages/rarecell-mcp-knowledge/tests/data"
)


def test_markers_retriever_search(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    retriever = MarkersDBRetriever(session=session)
    hits = retriever.search("T cell", tissue="blood")
    all_genes = {g for h in hits for g in h.payload["genes"]}
    assert "CD3D" in all_genes


def test_markers_retriever_canonical_panel(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "markers.sqlite",
        cache_path=tmp_path / "cache.sqlite",
    )
    retriever = MarkersDBRetriever(session=session)
    hit = retriever.canonical_panel("T cell")
    assert "CD3D" in hit.payload["genes"]
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Write `markers_db.py`**

```python
"""MarkersDBRetriever — adapts KnowledgeSession to a Retriever-shaped API."""
from __future__ import annotations
from rarecell.rag.knowledge import KnowledgeSession
from rarecell_mcp_knowledge.citation import RetrievalHit


class MarkersDBRetriever:
    """Wraps KnowledgeSession's `search_markers` and `get_canonical_panel` tools."""

    def __init__(self, session: KnowledgeSession):
        self.session = session

    def search(
        self, cell_type: str, *, tissue: str | None = None,
    ) -> list[RetrievalHit]:
        kwargs: dict = {"cell_type": cell_type}
        if tissue is not None:
            kwargs["tissue"] = tissue
        raw_hits = self.session.call("search_markers", kwargs)
        return [RetrievalHit.model_validate(h) for h in raw_hits]

    def canonical_panel(self, name: str) -> RetrievalHit:
        raw = self.session.call("get_canonical_panel", {"name": name})
        return RetrievalHit.model_validate(raw)
```

- [ ] **Step 4: Run, expect 2 pass**

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/rag/markers_db.py packages/rarecell/tests/rag/test_markers_db.py
git commit -m "Add MarkersDBRetriever over KnowledgeSession"
```

---

## Task 6: System prompt + `agent/client.py` — Anthropic SDK wrapper

**Files:**
- Create: `packages/rarecell/src/rarecell/agent/__init__.py`
- Create: `packages/rarecell/src/rarecell/agent/system_prompt.md`
- Create: `packages/rarecell/src/rarecell/agent/client.py`
- Create: `packages/rarecell/tests/agent/__init__.py` (empty)
- Create: `packages/rarecell/tests/agent/test_client.py`

- [ ] **Step 1: Write the system prompt**

`packages/rarecell/src/rarecell/agent/system_prompt.md`:

```markdown
You are a single-cell genomics advisor specialized in rare-cell isolation.
You operate against a frozen TargetCellProfile.

Your decisions must be evidence-based — every recommendation cites either the
consensus-table evidence row for that cluster or a literature/marker-DB hit
from the supplied RAG context.

For each ambiguous cluster you emit a recommendation: one of `keep`, `drop`,
or `purify`, with a confidence score in [0, 1], a short reasoning string, and
the list of citation IDs that supported the decision.

You never modify the AnnData, never re-cluster, never change profile parameters.
Your only output is structured per-cluster recommendations. The state machine
runs the workflow; you advise on per-cluster decisions only.

When the evidence is ambiguous (e.g., strong positive panel but moderate
contamination), prefer `purify` over `drop`. Surgical subclustering can
recover real cells that a wholesale drop would lose.

When negative contamination is high (`is_contaminant_frac` > 0.4) AND positive
panel pass fraction is low (< 0.2), prefer `drop`.

When positive panel pass fraction is high (>= 0.5) AND contamination is low
(< 0.1), prefer `keep`.
```

- [ ] **Step 2: Write the failing test**

`packages/rarecell/tests/agent/test_client.py`:

```python
import respx
from httpx import Response
from rarecell.agent.client import AnthropicClient


@respx.mock
def test_client_loads_system_prompt():
    client = AnthropicClient(api_key="fake-key")
    assert "single-cell genomics advisor" in client.system_prompt


@respx.mock
def test_client_call_with_messages_returns_text():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-haiku-4-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }))

    client = AnthropicClient(api_key="fake-key", model="claude-haiku-4-5")
    resp = client.messages_create(messages=[{"role": "user", "content": "Hi"}])
    assert resp["content"][0]["text"] == "Hello"
```

- [ ] **Step 3: Run, expect ImportError**

- [ ] **Step 4: Write `agent/client.py`**

```python
"""Anthropic SDK wrapper for rarecell.

Thin layer over the anthropic Python client that:
  - Loads the rarecell system prompt from system_prompt.md
  - Exposes a `messages_create()` helper returning the raw response dict
  - Defers the actual SDK import to runtime so the package imports
    without the [agent] extra
"""
from __future__ import annotations
from importlib import resources
from typing import Any


def _load_system_prompt() -> str:
    return (resources.files("rarecell.agent") / "system_prompt.md").read_text()


class AnthropicClient:
    def __init__(
        self, *, api_key: str, model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
    ):
        # Lazy import — anthropic only required when the [agent] extra is installed
        import anthropic  # noqa: F401
        self._sdk = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = _load_system_prompt()

    def messages_create(self, *, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Thin pass-through to anthropic.messages.create. Returns the raw JSON dict."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
        response = self._sdk.messages.create(**kwargs)
        return response.model_dump()
```

- [ ] **Step 5: Write `agent/__init__.py`**

```python
"""rarecell.agent — advisor agent built on the Anthropic SDK."""
```

- [ ] **Step 6: Run, expect 2 pass**

Note: `respx.mock` is required for these tests. The `respx.post` decorator does NOT mock the anthropic SDK's internal HTTP — anthropic's client uses httpx under the hood, which respx can mock. Verify that the test actually exercises the mocked endpoint. If it doesn't (because anthropic uses a different transport), adjust to mock the appropriate endpoint or use `unittest.mock.patch` on `anthropic.Anthropic.messages.create`.

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/agent/ packages/rarecell/tests/agent/
git add packages/rarecell/src/rarecell/agent/ packages/rarecell/tests/agent/
git commit -m "Add agent.AnthropicClient + system prompt"
```

---

## Task 7: `agent/recommender.py` — ClaudeRecommender via structured output

**Files:**
- Create: `packages/rarecell/src/rarecell/agent/recommender.py`
- Create: `packages/rarecell/tests/agent/fixtures/__init__.py` (empty)
- Create: `packages/rarecell/tests/agent/fixtures/anthropic_responses.py`
- Create: `packages/rarecell/tests/agent/test_recommender.py`

ClaudeRecommender implements the existing `Recommender` Protocol from Plan 1's `rarecell.recommender.base`. Plan 1's BasicRecommender keeps working; ClaudeRecommender is the LLM-backed swap-in.

Approach: ask Claude to return structured JSON (a list of `{cluster_id, recommendation, confidence, evidence_summary, reasoning, citations}` objects). Use tool-use forcing or schema-constrained JSON output. Simplest: instruct in the prompt to output a single JSON object, then parse it.

- [ ] **Step 1: Write the canned Anthropic response fixture**

`packages/rarecell/tests/agent/fixtures/anthropic_responses.py`:

```python
"""Canned Anthropic API responses for ClaudeRecommender tests."""

RECOMMENDATIONS_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "content": [
        {
            "type": "text",
            "text": """```json
{
  "recommendations": [
    {
      "cluster_id": "0",
      "recommendation": "keep",
      "confidence": 0.9,
      "evidence_summary": {"score_pan_t_mean": 2.0, "is_contaminant_frac": 0.02},
      "reasoning": "Strong positive signal and low contamination.",
      "citations": []
    },
    {
      "cluster_id": "1",
      "recommendation": "drop",
      "confidence": 0.85,
      "evidence_summary": {"score_pan_t_mean": 0.1, "is_contaminant_frac": 0.5},
      "reasoning": "Weak positive and heavy contamination.",
      "citations": []
    },
    {
      "cluster_id": "2",
      "recommendation": "purify",
      "confidence": 0.55,
      "evidence_summary": {"score_pan_t_mean": 1.5, "is_contaminant_frac": 0.15},
      "reasoning": "Mixed signal — recommend surgical subclustering.",
      "citations": []
    }
  ]
}
```"""
        }
    ],
}
```

- [ ] **Step 2: Write the failing test**

`packages/rarecell/tests/agent/test_recommender.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
from rarecell.agent.recommender import ClaudeRecommender
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance, QCParams,
)
from packages.rarecell.tests.agent.fixtures.anthropic_responses import (
    RECOMMENDATIONS_RESPONSE,
)


def _profile():
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def _table():
    return pd.DataFrame({
        "cluster": ["0", "1", "2"],
        "n_cells": [100, 100, 100],
        "score_pan_t_mean": [2.0, 0.1, 1.5],
        "pass_pan_t_frac": [0.9, 0.05, 0.7],
        "is_contaminant_frac": [0.02, 0.5, 0.15],
    })


def test_claude_recommender_parses_structured_response():
    mock_client = MagicMock()
    mock_client.messages_create.return_value = RECOMMENDATIONS_RESPONSE

    recommender = ClaudeRecommender(profile=_profile(), client=mock_client)
    recs = recommender.recommend(_table())

    by_id = {r.cluster_id: r for r in recs}
    assert by_id["0"].recommendation == "keep"
    assert by_id["1"].recommendation == "drop"
    assert by_id["2"].recommendation == "purify"
    assert all(0 <= r.confidence <= 1 for r in recs)
    assert mock_client.messages_create.called


def test_claude_recommender_handles_malformed_json():
    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": "Not JSON at all."}],
    }

    recommender = ClaudeRecommender(profile=_profile(), client=mock_client)
    # On parse failure, fall back to empty recommendation list with a
    # warning rather than crashing the runner.
    recs = recommender.recommend(_table())
    assert recs == []
```

- [ ] **Step 3: Run, expect ImportError**

- [ ] **Step 4: Write `agent/recommender.py`**

```python
"""ClaudeRecommender — LLM-backed implementation of the Recommender protocol."""
from __future__ import annotations
import json
import re
from typing import Any
import pandas as pd

from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommendation, Recommender


_log = get_logger("rarecell.agent.recommender")


def _extract_json_block(text: str) -> dict | None:
    """Find the first ```json ... ``` block in text and parse it.
    Falls back to parsing the entire text as JSON.
    """
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class ClaudeRecommender(Recommender):
    """Asks Claude to recommend keep/drop/purify per cluster.

    Output format: a single JSON object with key "recommendations" mapping
    to a list of {cluster_id, recommendation, confidence, evidence_summary,
    reasoning, citations} objects.
    """

    def __init__(self, *, profile: TargetCellProfile, client: Any):
        # `client` is duck-typed AnthropicClient (or a mock); not statically typed
        # so the rarecell package can be imported without the [agent] extra.
        self.profile = profile
        self.client = client

    def _build_user_message(self, table: pd.DataFrame) -> str:
        positive_panels = list(self.profile.positive_markers.keys())
        negative_panels = list(self.profile.negative_markers.keys())
        return (
            "Profile target: " + self.profile.name + " in "
            + ", ".join(self.profile.tissue) + ".\n"
            "Positive panels: " + ", ".join(positive_panels) + "\n"
            "Negative panels: " + ", ".join(negative_panels) + "\n\n"
            "Consensus table (one row per cluster):\n"
            + table.to_string(index=False) + "\n\n"
            "Return a single ```json``` block with this exact shape:\n"
            '```json\n{\n  "recommendations": [\n    {\n'
            '      "cluster_id": "<id>",\n'
            '      "recommendation": "keep" | "drop" | "purify",\n'
            '      "confidence": <float 0-1>,\n'
            '      "evidence_summary": {...},\n'
            '      "reasoning": "<short string>",\n'
            '      "citations": ["<id>", ...]\n'
            '    }\n  ]\n}\n```'
        )

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]:
        user = self._build_user_message(table)
        resp = self.client.messages_create(
            messages=[{"role": "user", "content": user}]
        )
        # resp["content"] is a list of content blocks; take the first text block
        text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
        if not text_blocks:
            _log.warning("claude_recommender.no_text_blocks")
            return []
        parsed = _extract_json_block(text_blocks[0]["text"])
        if parsed is None or "recommendations" not in parsed:
            _log.warning("claude_recommender.parse_failure")
            return []
        out: list[Recommendation] = []
        for entry in parsed["recommendations"]:
            try:
                out.append(Recommendation(**entry))
            except Exception as e:  # pydantic validation
                _log.warning("claude_recommender.validation_failure",
                              error=str(e), entry=entry)
        return out
```

- [ ] **Step 5: Run, expect 2 pass**

Run: `uv run pytest packages/rarecell/tests/agent/test_recommender.py -v`

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/agent/ packages/rarecell/tests/agent/
git add packages/rarecell/src/rarecell/agent/recommender.py packages/rarecell/tests/agent/
git commit -m "Add ClaudeRecommender with JSON-block structured output"
```

---

## Task 8: Wire S5_GATE2 and S6_GATE3 into IsolateRunner (Plan 1 deferral)

**Files:**
- Modify: `packages/rarecell/src/rarecell/state_machine/isolate.py`
- Create: `packages/rarecell/tests/state_machine/test_isolate_gates.py`

Plan 1's `IsolateRunner` never transitions through `S5_GATE2` (sub-cluster decisions inside purify) or `S6_GATE3` (final abundance abort). Plan 3 adds them because `ClaudeRecommender` needs gate-2 to log sub-cluster decisions and gate-3 to honor the `abort_on_anomaly` policy.

Approach:
- Gate 2: after `subcluster_and_purify`, ask the recommender for keep/drop/purify on the sub-clusters present in the returned AnnData. Log those decisions to `decisions.jsonl` with `gate=2`.
- Gate 3: after `_select_isolated`, compute the isolated abundance fraction. If `profile.auto_policy.gates.gate3_final == "abort_on_anomaly"` and the fraction is outside `expected_abundance * max_abundance_deviation`, raise `IsolationAbortedError`.

- [ ] **Step 1: Write the failing tests**

`packages/rarecell/tests/state_machine/test_isolate_gates.py`:

```python
from pathlib import Path
import json
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, BICCNRules, ReferenceLabels, BatchCorrection, PurifyParams,
    AutoPolicy, GateAutoPolicy,
)
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner


def _profile_with_purify():
    return TargetCellProfile(
        profile_id="syn-t", name="syn T", description="d",
        target_lineage="lymphoid", tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.02, max_fraction=0.10),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D", "CD3E", "CD3G", "TRAC"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(min_genes_per_cell=10, max_pct_mt=100,
                    max_genes_per_cell=10000, min_cells_per_gene=1),
        purify=PurifyParams(enabled=True, min_cluster_purity=0.5),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True, reviewer="test@x",
    ).freeze()


def test_gate2_decisions_logged_when_purify_runs(tmp_path: Path):
    """Whenever the runner enters S5_PURIFY, it must enter S5_GATE2 and write
    at least one gate=2 decision to decisions.jsonl."""
    profile = _profile_with_purify()
    runner = IsolateRunner(
        adata=make_synthetic(seed=0), profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path, auto_policy="recommendation",
    )
    runner.run()

    decisions = (tmp_path / "decisions.jsonl").read_text().strip().splitlines()
    gates = [json.loads(line)["gate"] for line in decisions]
    # If gate 1 produced any "purify" decisions, gate 2 must have run too.
    if 1 in gates and any(json.loads(line).get("user_decision") == "purify"
                          for line in decisions
                          if json.loads(line)["gate"] == 1):
        assert 2 in gates, "Gate 2 must run when any gate-1 decision is 'purify'"


def test_gate3_aborts_when_abundance_out_of_bounds(tmp_path: Path):
    """A profile with gate3_final='abort_on_anomaly' and very narrow expected
    abundance bounds should abort if the isolated fraction is wildly outside."""
    from rarecell.errors import IsolationAbortedError
    import pytest

    profile = TargetCellProfile(
        profile_id="abort-t", name="t", description="d",
        target_lineage="lymphoid", tissue=["pbmc"],
        # Expect <2% — synthetic fixture's planted T cluster is ~5%
        expected_abundance=ExpectedAbundance(min_fraction=0.0001,
                                              max_fraction=0.001),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D", "CD3E", "CD3G", "TRAC"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(min_genes_per_cell=10, max_pct_mt=100,
                    max_genes_per_cell=10000, min_cells_per_gene=1),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        auto_policy=AutoPolicy(gates=GateAutoPolicy(
            gate3_final="abort_on_anomaly",
            max_abundance_deviation=2.0,
        )),
        human_reviewed=True, reviewer="test@x",
    ).freeze()

    runner = IsolateRunner(
        adata=make_synthetic(seed=0), profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path, auto_policy="recommendation",
    )
    with pytest.raises(IsolationAbortedError, match="abundance"):
        runner.run()
```

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_isolate_gates.py -v`

- [ ] **Step 3: Modify `state_machine/isolate.py`**

Open `packages/rarecell/src/rarecell/state_machine/isolate.py`. Add three changes:

**Change A — `IsolateAbortedError` import:** at the top of the file, replace the existing `from rarecell.errors import UnreviewedProfileError` with:

```python
from rarecell.errors import IsolationAbortedError, UnreviewedProfileError
```

**Change B — new `_s5_gate2` helper after `_s5_purify`:**

```python
    def _s5_gate2(self, purified_adata) -> list[str]:
        """Gate 2: per-sub-cluster decisions after surgical purify.

        Returns the list of sub-cluster IDs to keep (in the purified adata's
        leiden labelling).
        """
        from rarecell.core import evidence as evidence_mod
        table = evidence_mod.score_evidence(
            purified_adata, self.profile, cluster_key="leiden")
        recs = self.recommender.recommend(table)
        user_decisions = self._decide_for_gate(2, recs)
        self._log_decisions(2, recs, user_decisions)
        return [cid for cid, d in user_decisions.items() if d == "keep"]
```

**Change C — new `_s6_gate3` helper:**

```python
    def _s6_gate3(self, isolated, input_n_obs: int) -> None:
        """Gate 3: final abundance abort policy.

        If profile.auto_policy.gates.gate3_final == "abort_on_anomaly" and
        the isolated fraction is outside expected_abundance * max_deviation,
        raise IsolationAbortedError.
        """
        policy = self.profile.auto_policy.gates
        if policy.gate3_final != "abort_on_anomaly":
            return
        frac = isolated.n_obs / max(input_n_obs, 1)
        lo = self.profile.expected_abundance.min_fraction / policy.max_abundance_deviation
        hi = self.profile.expected_abundance.max_fraction * policy.max_abundance_deviation
        if not (lo <= frac <= hi):
            raise IsolationAbortedError(
                f"Gate 3 abort: isolated abundance {frac:.4f} is outside "
                f"expected bounds [{lo:.4f}, {hi:.4f}] "
                f"(max_abundance_deviation={policy.max_abundance_deviation})."
            )
```

**Change D — `run()` method:** modify the section that handles purify and final selection. Find the existing code:

```python
            if suspect:
                self.state = IsolateState.S5_PURIFY
                purified = self._s5_purify(suspect)
                if purified is not None:
                    self.adata = purified
                    extra = sorted(set(purified.obs["leiden"].astype(str)) & set(suspect))
                    kept.extend(extra)
            self.state = IsolateState.S6_FINAL
            isolated = self._select_isolated(kept)
            self.state = IsolateState.S7_REPORT
```

Replace with:

```python
            if suspect:
                self.state = IsolateState.S5_PURIFY
                purified = self._s5_purify(suspect)
                if purified is not None:
                    self.adata = purified
                    # Gate 2: per-sub-cluster decisions
                    self.state = IsolateState.S5_GATE2
                    sub_kept = self._s5_gate2(purified)
                    kept = sorted(set(kept) | set(sub_kept))
            self.state = IsolateState.S6_FINAL
            isolated = self._select_isolated(kept)
            # Gate 3: final abundance abort policy
            self.state = IsolateState.S6_GATE3
            self._s6_gate3(isolated, self._input_n_obs)
            self.state = IsolateState.S7_REPORT
```

- [ ] **Step 4: Run, expect 2 pass**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_isolate_gates.py -v`

Also re-run the original isolate runner test to confirm no regression:

```bash
uv run pytest packages/rarecell/tests/state_machine/ -v
```

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/tests/state_machine/test_isolate_gates.py
git add packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/tests/state_machine/test_isolate_gates.py
git commit -m "Wire S5_GATE2 + S6_GATE3 into IsolateRunner"
```

---

## Task 9: ClaudeRecommender end-to-end against synthetic fixture

**Files:**
- Create: `tests/integration/test_claude_recommender_e2e.py`

This is an integration test that runs `IsolateRunner` with `ClaudeRecommender` against the synthetic fixture, using a mocked Anthropic client that returns canned recommendations. Verifies the full advisor pipeline glues together cleanly.

- [ ] **Step 1: Write the test**

```python
"""End-to-end: ClaudeRecommender drives IsolateRunner on synthetic fixture."""
from pathlib import Path
from unittest.mock import MagicMock
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.agent.recommender import ClaudeRecommender
from rarecell.state_machine.isolate import IsolateRunner
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, BICCNRules, ReferenceLabels, BatchCorrection, PurifyParams,
)


def _profile():
    return TargetCellProfile(
        profile_id="claude-t", name="t", description="d",
        target_lineage="lymphoid", tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.02, max_fraction=0.10),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D", "CD3E", "CD3G", "TRAC"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(min_genes_per_cell=10, max_pct_mt=100,
                    max_genes_per_cell=10000, min_cells_per_gene=1),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True, reviewer="test@x",
    ).freeze()


def _make_response(table_df):
    """Build a canned Anthropic response that 'keep's every cluster.

    For the synthetic test we don't need clever Claude reasoning; we just
    need the structured-output path to round-trip.
    """
    cluster_ids = list(table_df["cluster"].astype(str))
    recs_json = (
        '{"recommendations": ['
        + ", ".join(
            f'{{"cluster_id": "{cid}", "recommendation": "keep", '
            f'"confidence": 0.9, "evidence_summary": {{}}, '
            f'"reasoning": "auto-keep", "citations": []}}'
            for cid in cluster_ids
        )
        + ']}'
    )
    return {
        "content": [{"type": "text", "text": f"```json\n{recs_json}\n```"}],
    }


def test_claude_recommender_drives_runner(tmp_path):
    profile = _profile()
    adata = make_synthetic(seed=0)

    captured_tables = []

    def fake_messages_create(*, messages, tools=None):
        # Pull the table out of the prompt by extracting cluster IDs the model
        # would see. For the test we just return keep-all using whatever
        # clusters exist in the most recent score_evidence call.
        # We can rely on the table having been computed before this is called.
        from pandas import DataFrame
        # In a real test we'd parse the prompt; instead we generate a stub
        # that the runner-side parser will accept for any cluster set.
        # The simplest approach is to rebuild the table here from the runner's
        # adata, but that's coupling. Use a static 3-cluster keep-all response;
        # the runner will validate the cluster IDs match the actual clusters
        # via the consensus-table column. If they don't match, _select_isolated
        # returns the empty mask, which fails the test — surface that.
        return _make_response(DataFrame({"cluster": ["0", "1", "2", "3"]}))

    mock_client = MagicMock()
    mock_client.messages_create.side_effect = fake_messages_create

    recommender = ClaudeRecommender(profile=profile, client=mock_client)
    runner = IsolateRunner(
        adata=adata, profile=profile, recommender=recommender,
        out_dir=tmp_path, auto_policy="recommendation",
    )
    result = runner.run()
    assert result.isolated.n_obs > 0
    assert mock_client.messages_create.called
```

This test is intentionally permissive — it verifies the wiring works, not that ClaudeRecommender makes good calls. Real Claude calls are tested only with mocked responses; live Anthropic API is never hit in CI.

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/integration/test_claude_recommender_e2e.py -v`

If `n_obs == 0`, the mock's cluster IDs don't match the runner's actual leiden labels. Fix by widening the mock to cover more cluster IDs (e.g., "0".."9"), or by inspecting the prompt the runner sends (via `mock_client.messages_create.call_args`) to extract the actual cluster IDs.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check tests/integration/test_claude_recommender_e2e.py
git add tests/integration/test_claude_recommender_e2e.py
git commit -m "Add ClaudeRecommender end-to-end test driving IsolateRunner"
```

---

## Task 10: `agent/draft.py` — NL → draft TargetCellProfile

**Files:**
- Create: `packages/rarecell/src/rarecell/agent/draft.py`
- Modify: `packages/rarecell/src/rarecell/profile/draft.py` (re-export)
- Create: `packages/rarecell/tests/agent/test_draft.py`

Drafting flow: given a natural-language description (e.g., "rare T cells in postmortem ALS brain"), call literature + marker RAG, then ask Claude to compose a `TargetCellProfile`-shaped JSON. The user then reviews + freezes.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest.mock import MagicMock
from rarecell.agent.draft import draft_profile_from_prompt
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv
from rarecell.rag.knowledge import build_knowledge_session


PLAN2_FIXTURES = (
    Path(__file__).resolve().parents[4]
    / "packages/rarecell-mcp-knowledge/tests/data"
)


CANNED_PROFILE_JSON = '''```json
{
  "profile_id": "draft-tcell-pbmc",
  "name": "T cells (drafted)",
  "description": "Drafted from NL prompt",
  "target_lineage": "lymphoid",
  "tissue": ["pbmc"],
  "expected_abundance": {"min_fraction": 0.1, "max_fraction": 0.6},
  "positive_markers": {
    "pan_t": {
      "genes": ["CD3D", "CD3E"],
      "threshold_z": 1.0,
      "citations": [{"source_id": "panglaodb:T_cell", "source": "panglaodb"}]
    }
  },
  "negative_markers": {},
  "qc": {"min_genes_per_cell": 200, "max_pct_mt": 10}
}
```'''


def test_draft_profile_from_prompt_returns_target_cell_profile(tmp_path):
    catalog = MarkersCatalog(tmp_path / "m.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=PLAN2_FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=PLAN2_FIXTURES / "panglaodb_tiny.tsv",
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite", cache_path=tmp_path / "c.sqlite",
    )

    mock_client = MagicMock()
    mock_client.messages_create.return_value = {
        "content": [{"type": "text", "text": CANNED_PROFILE_JSON}],
    }

    profile = draft_profile_from_prompt(
        prompt="rare T cells in PBMC",
        client=mock_client,
        session=session,
    )
    assert profile.profile_id == "draft-tcell-pbmc"
    assert "CD3D" in profile.positive_markers["pan_t"].genes
    # The draft is NOT frozen — the user must review + freeze
    assert profile.frozen is False
    assert profile.human_reviewed is False
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Write `agent/draft.py`**

```python
"""NL → draft TargetCellProfile, grounded by literature + markers RAG."""
from __future__ import annotations
import json
import re
from typing import Any

from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile
from rarecell.rag.knowledge import KnowledgeSession
from rarecell.rag.literature import LiteratureRetriever
from rarecell.rag.markers_db import MarkersDBRetriever


_log = get_logger("rarecell.agent.draft")


def _extract_json_block(text: str) -> dict | None:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _build_drafting_prompt(
    user_prompt: str,
    literature_hits: list,
    marker_hits: list,
) -> str:
    lit_lines = [
        f"- {h.citation.source_id}: {h.title} — {h.snippet[:150]}"
        for h in literature_hits[:5]
    ]
    marker_lines = [
        f"- {h.title}: {', '.join(h.payload.get('genes', [])[:10])}"
        for h in marker_hits[:5]
    ]
    return (
        f"User prompt:\n{user_prompt}\n\n"
        "Literature hits:\n" + "\n".join(lit_lines) + "\n\n"
        "Marker DB hits:\n" + "\n".join(marker_lines) + "\n\n"
        "Draft a TargetCellProfile as a single ```json``` block matching this shape:\n"
        '```json\n{\n'
        '  "profile_id": "<kebab-case-slug>",\n'
        '  "name": "<short name>",\n'
        '  "description": "<one paragraph>",\n'
        '  "target_lineage": "lymphoid" | "myeloid" | "neural" | "epithelial",\n'
        '  "tissue": ["<tissue1>", ...],\n'
        '  "expected_abundance": {"min_fraction": <float>, "max_fraction": <float>},\n'
        '  "positive_markers": {"<panel_name>": {"genes": [...], "threshold_z": 1.0, "citations": [...]}},\n'
        '  "negative_markers": {},\n'
        '  "qc": {"min_genes_per_cell": <int>, "max_pct_mt": <float>}\n'
        '}\n```\n'
        'The draft must NOT set frozen or human_reviewed; the user reviews + freezes separately.'
    )


def draft_profile_from_prompt(
    *, prompt: str, client: Any, session: KnowledgeSession,
) -> TargetCellProfile:
    """Draft a TargetCellProfile from a natural-language prompt.

    Retrieves literature + marker hits, asks Claude to compose a profile,
    and returns the parsed (un-frozen) TargetCellProfile.

    Raises ValueError if the model's response doesn't parse.
    """
    lit_retriever = LiteratureRetriever(session=session)
    marker_retriever = MarkersDBRetriever(session=session)

    # Naive RAG: search both layers with the user's prompt as the query.
    try:
        literature_hits = lit_retriever.search(prompt, page_size=5)
    except Exception as e:
        _log.warning("draft.literature_search_failed", error=str(e))
        literature_hits = []
    try:
        marker_hits = marker_retriever.search(prompt)
    except Exception as e:
        _log.warning("draft.marker_search_failed", error=str(e))
        marker_hits = []

    user_msg = _build_drafting_prompt(prompt, literature_hits, marker_hits)
    resp = client.messages_create(messages=[{"role": "user", "content": user_msg}])
    text_blocks = [b for b in resp.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise ValueError("Drafting response had no text blocks.")
    parsed = _extract_json_block(text_blocks[0]["text"])
    if parsed is None:
        raise ValueError("Drafting response did not contain a parseable JSON block.")

    return TargetCellProfile.model_validate(parsed)
```

- [ ] **Step 4: Modify `profile/draft.py` to re-export**

Replace the stub at `packages/rarecell/src/rarecell/profile/draft.py` (or create if absent) with:

```python
"""Profile drafting — thin re-export of rarecell.agent.draft.

Drafting requires the [agent] extra. Import from rarecell.profile.draft for
convenience; the implementation lives in rarecell.agent.draft so the core
profile module stays LLM-free.
"""
try:
    from rarecell.agent.draft import draft_profile_from_prompt
except ImportError as e:
    raise ImportError(
        "Profile drafting requires the [agent] extra. "
        "Install with: pip install rarecell[agent]"
    ) from e

__all__ = ["draft_profile_from_prompt"]
```

- [ ] **Step 5: Run, expect 1 pass**

Run: `uv run pytest packages/rarecell/tests/agent/test_draft.py -v`

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/agent/draft.py packages/rarecell/src/rarecell/profile/draft.py packages/rarecell/tests/agent/test_draft.py
git add packages/rarecell/src/rarecell/agent/draft.py packages/rarecell/src/rarecell/profile/draft.py packages/rarecell/tests/agent/test_draft.py
git commit -m "Add draft_profile_from_prompt + profile.draft re-export"
```

---

## Task 11: Re-export ClaudeRecommender from `rarecell.recommender`

**Files:**
- Modify: `packages/rarecell/src/rarecell/recommender/__init__.py`

- [ ] **Step 1: Read the existing `__init__.py`**

`packages/rarecell/src/rarecell/recommender/__init__.py` currently exports `Recommender`, `Recommendation`, `BasicRecommender`.

- [ ] **Step 2: Modify to optionally re-export `ClaudeRecommender`**

Replace contents with:

```python
"""rarecell.recommender — Recommender protocol + concrete implementations.

ClaudeRecommender is re-exported here for convenience but lives in
rarecell.agent.recommender. It requires the [agent] extra.
"""
from rarecell.recommender.base import Recommendation, Recommender
from rarecell.recommender.basic import BasicRecommender

try:
    from rarecell.agent.recommender import ClaudeRecommender
    _has_claude = True
except ImportError:
    _has_claude = False

__all__ = ["Recommender", "Recommendation", "BasicRecommender"]
if _has_claude:
    __all__.append("ClaudeRecommender")
```

- [ ] **Step 3: Write a small test**

`packages/rarecell/tests/recommender/test_reexport.py`:

```python
def test_basic_recommender_always_importable():
    from rarecell.recommender import BasicRecommender, Recommendation, Recommender
    assert BasicRecommender is not None
    assert Recommendation is not None
    assert Recommender is not None


def test_claude_recommender_importable_when_agent_installed():
    """If the [agent] extra is installed (it is in the dev env), ClaudeRecommender
    is re-exported from rarecell.recommender."""
    try:
        from rarecell.recommender import ClaudeRecommender
    except ImportError:
        # Acceptable in environments without [agent]
        return
    assert ClaudeRecommender is not None
```

- [ ] **Step 4: Run, expect 2 pass**

Run: `uv run pytest packages/rarecell/tests/recommender/test_reexport.py -v`

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check packages/rarecell/src/rarecell/recommender/ packages/rarecell/tests/recommender/test_reexport.py
git add packages/rarecell/src/rarecell/recommender/__init__.py packages/rarecell/tests/recommender/test_reexport.py
git commit -m "Re-export ClaudeRecommender from rarecell.recommender"
```

---

## Task 12: CI integration + README polish

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `README.md`
- Modify: `packages/rarecell/README.md` (if exists; else this is no-op)

- [ ] **Step 1: Verify CI test invocation**

Read `.github/workflows/test.yml`. The current `unit` job runs `uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py -v`. The new agent tests live under `packages/rarecell/tests/{rag,agent}/` and `packages/rarecell/tests/state_machine/test_isolate_gates.py` — all already covered by `packages/rarecell/tests`. The new `tests/integration/test_claude_recommender_e2e.py` is NOT covered. Add it to the unit job:

Append `tests/integration/test_claude_recommender_e2e.py` to the existing pytest command. Final line:

```yaml
      - run: uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py tests/integration/test_claude_recommender_e2e.py -v
```

- [ ] **Step 2: Verify lint already covers**

The lint workflow runs `ruff check .` and covers all new paths automatically. No changes needed.

- [ ] **Step 3: Update root `README.md` to mention the advisor agent**

Read `README.md`. Add (near the bottom or in a "Status" section) a one-paragraph note:

```markdown
## Advisor agent (Plan 3 — `rarecell.agent`)

The advisor experience lives in `rarecell.agent` and is gated by the
`[agent]` optional extra:

```bash
pip install 'rarecell[agent]'
```

This installs the Anthropic SDK and `rarecell-mcp-knowledge`. The agent
provides `ClaudeRecommender` (LLM-backed swap-in for the heuristic
`BasicRecommender`) and a profile drafting flow that turns natural-language
prompts into reviewable `TargetCellProfile` YAMLs.

Without the extra, `rarecell.core` works unchanged with `BasicRecommender`.
```

- [ ] **Step 4: Run the full suite locally**

```bash
uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py tests/integration/test_claude_recommender_e2e.py -v
```

Expected: 81 (existing) + ~12 (new from this plan) = ~93 tests pass.

- [ ] **Step 5: Lint clean check**

```bash
uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/test.yml README.md
git commit -m "Wire Plan 3 tests into CI; document advisor agent in README"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Plan coverage |
|---|---|
| §5.1 deterministic control flow, LLM reasoning | ClaudeRecommender implements Recommender Protocol (Task 7); state machine is unchanged in its determinism (Task 8 just adds two states already declared in Plan 1) |
| §5.2 state machine S0..S7 with 3 gates | Task 8 wires gates 2 + 3 (Plan 1 only had gate 1) |
| §5.3 structured per-cluster recommendations | Task 7 returns list[Recommendation] |
| §5.4 narrow agent tool surface | Tasks 4, 5 provide read-only retriever wrappers; no side-effecting `core` tools |
| §5.5 BasicRecommender fallback | Plan 1; not touched here |
| §5.6 three modes: profile_draft, isolate, review | Tasks 7+8 cover `isolate`. Task 10 covers `profile_draft`. `review` mode is **deferred** to Plan 4 (intentional scope cut — see notes below). |
| §6.1 Retriever abstraction | Task 2 |
| §6.2 Citation propagation | The Recommendation model has a `citations: list[str]` field (Plan 1). ClaudeRecommender populates it from the model's response. End-to-end propagation through to bibliography.bib is already in IsolationReport (Plan 1). |
| §6.3 offline + no-LLM modes | Plan 1's BasicRecommender + optional `[agent]` extra (Task 1) |

**Placeholder scan:** no TBD/TODO. Every step has either complete code or a precise modification instruction.

**Type consistency:**
- `Recommendation` model unchanged from Plan 1; ClaudeRecommender's parsed entries pass through `Recommendation(**entry)`.
- `KnowledgeSession.call(name, kwargs)` signature consistent across Tasks 3, 4, 5.
- `LiteratureRetriever.search(query, *, year_range, tissue, page_size)` consistent in Task 4.
- `MarkersDBRetriever.search(cell_type, *, tissue)` consistent in Task 5.

**Deliberate scope cuts:**
- `review` agent mode (replay a report, surface anomalies) — deferred to Plan 4 alongside the CLI/notebook front-ends where it's most natural.
- The full agentic tool-use loop (Claude calling `search_literature` itself mid-recommendation) — replaced with the simpler "RAG into prompt, structured-output back" pattern. Cheaper, more deterministic, and good enough for v0.1.
- Live Anthropic integration tests — all tests use mocked clients. Plan 4 may add a nightly cron with real API.
- `prior-runs RAG` (retrieve over past `IsolationReport` decisions) — listed as an open follow-up in the spec; not in Plan 3.
