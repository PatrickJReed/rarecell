# rarecell-mcp-knowledge — Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rarecell-mcp-knowledge` — a FastMCP server that aggregates literature retrieval (Europe PMC) and marker-database retrieval (CellMarker 2.0, PanglaoDB, MSigDB, Enrichr) behind a single MCP surface. Consumable from any MCP client; Plan 3 wires the rarecell agent into it.

**Architecture:** Standalone Python package under `packages/rarecell-mcp-knowledge/`, independent of `rarecell`. Five MCP tools, two retrieval backends (literature + markers). Local SQLite caches both for marker DBs (CellMarker, PanglaoDB) and for query results. No LLM dependencies. Plays well in CI: live API smoke tests are network-gated; everything else is cache-based and offline.

**Tech Stack:** Python 3.11+, FastMCP (`fastmcp`), httpx for HTTP, SQLite via stdlib, pytest, hypothesis, ruff, mypy, uv.

**Spec:** `docs/superpowers/specs/2026-05-20-rarecell-agentic-isolation-design.md` §3.4 and §5–§6.

**Plan 1 status:** Merged to main. The workspace `pyproject.toml` already declares `[tool.uv.workspace] members = ["packages/*"]`, so adding `packages/rarecell-mcp-knowledge/` enrolls it automatically.

---

## File Structure

```
rarecell/
├── packages/
│   ├── rarecell/                       # existing (Plan 1)
│   └── rarecell-mcp-knowledge/         # NEW (this plan)
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/rarecell_mcp_knowledge/
│       │   ├── __init__.py
│       │   ├── errors.py                  # KnowledgeError + subclasses
│       │   ├── cache.py                   # SQLite query-result cache (30-day TTL)
│       │   ├── citation.py                # shared Citation model (PMID, DOI, etc.)
│       │   ├── server.py                  # FastMCP server entry point
│       │   ├── literature/
│       │   │   ├── __init__.py
│       │   │   ├── europepmc.py           # Europe PMC REST client
│       │   │   └── client.py              # LiteratureBackend protocol + default
│       │   ├── markers/
│       │   │   ├── __init__.py
│       │   │   ├── catalog.py             # SQLite-backed local catalog (CellMarker + PanglaoDB)
│       │   │   ├── seed.py                # one-time SQLite seeding from TSV (downloadable)
│       │   │   └── client.py              # MarkerBackend protocol + default
│       │   ├── enrichr.py                 # live Enrichr REST client
│       │   ├── msigdb.py                  # live MSigDB REST client
│       │   └── cli.py                     # `rarecell-mcp-knowledge serve|seed`
│       └── tests/
│           ├── conftest.py
│           ├── test_errors.py
│           ├── test_cache.py
│           ├── test_citation.py
│           ├── test_europepmc.py          # uses httpx mock
│           ├── test_catalog.py            # uses tiny seeded sqlite fixture
│           ├── test_enrichr.py            # uses httpx mock
│           ├── test_msigdb.py             # uses httpx mock
│           ├── test_server.py             # FastMCP in-process smoke
│           └── data/
│               ├── cellmarker_tiny.tsv    # 50-row fixture
│               └── panglaodb_tiny.tsv     # 50-row fixture
```

**Boundaries:**
- `literature/` and `markers/` are independent subsystems with their own backends + protocols.
- `cache.py` is shared infrastructure; both subsystems use the same SQLite cache.
- `server.py` is the only file that imports FastMCP. Everything else is plain Python so it can be tested without an MCP runtime.
- `enrichr.py` and `msigdb.py` live at the package root (not under `markers/`) because they're live REST APIs, not local catalogs.

---

## Task 1: Package scaffold + uv workspace enrollment

**Files:**
- Create: `packages/rarecell-mcp-knowledge/pyproject.toml`
- Create: `packages/rarecell-mcp-knowledge/README.md`
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/__init__.py`

- [ ] **Step 1: Create `packages/rarecell-mcp-knowledge/pyproject.toml`**

```toml
[project]
name = "rarecell-mcp-knowledge"
version = "0.1.0.dev0"
description = "MCP server: literature + marker DB retrieval for rarecell"
authors = [{name = "Patrick Reed", email = "patrickjenningsreed@gmail.com"}]
license = "Apache-2.0"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "fastmcp>=0.4",
  "httpx>=0.27",
  "pydantic>=2.6",
  "structlog>=24.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov",
  "pytest-asyncio>=0.23",
  "respx>=0.21",          # httpx mock library
  "hypothesis>=6",
  "ruff",
  "mypy",
]

[project.scripts]
rarecell-mcp-knowledge = "rarecell_mcp_knowledge.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rarecell_mcp_knowledge"]
```

- [ ] **Step 2: Create `packages/rarecell-mcp-knowledge/README.md`**

```markdown
# rarecell-mcp-knowledge

FastMCP server exposing literature retrieval (Europe PMC) and marker-database
retrieval (CellMarker 2.0, PanglaoDB, MSigDB, Enrichr) behind a single MCP
surface. Designed to be consumed by the `rarecell` agent or any MCP client.

## Install

```bash
pip install rarecell-mcp-knowledge
```

## Run the server

```bash
rarecell-mcp-knowledge serve
```

## Seed the local marker catalog

On first run, the local SQLite catalog needs to be seeded with CellMarker 2.0
+ PanglaoDB data:

```bash
rarecell-mcp-knowledge seed
```

This downloads ~50 MB of TSVs and builds a local SQLite at
`~/.cache/rarecell/markers.sqlite`.

## Tools advertised

- `search_literature(query, year_range?, tissue?)`
- `fetch_abstract(pmid_or_doi)`
- `search_markers(cell_type, tissue?)`
- `get_canonical_panel(name)`
- `enrichr_enrich(genes, library)`

This is pre-release v0.x.
```

- [ ] **Step 3: Create package init**

```python
"""rarecell-mcp-knowledge — MCP server for literature + marker retrieval."""

__version__ = "0.1.0.dev0"
```

Path: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/__init__.py`.

- [ ] **Step 4: Sync the workspace**

Run: `uv sync --all-packages --all-extras --dev`
Expected: `rarecell-mcp-knowledge` resolves and installs alongside `rarecell`. No errors.

- [ ] **Step 5: Smoke-import**

Run: `uv run python -c "import rarecell_mcp_knowledge; print(rarecell_mcp_knowledge.__version__)"`
Expected: prints `0.1.0.dev0`.

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell-mcp-knowledge/ uv.lock
git commit -m "Scaffold rarecell-mcp-knowledge package"
```

---

## Task 2: Error hierarchy

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/errors.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from rarecell_mcp_knowledge.errors import (
    KnowledgeError, BackendUnreachableError, CacheCorruptedError,
    InvalidQueryError, RateLimitedError,
)


@pytest.mark.parametrize("cls", [
    BackendUnreachableError, CacheCorruptedError,
    InvalidQueryError, RateLimitedError,
])
def test_subclass_of_base(cls):
    err = cls("msg")
    assert isinstance(err, KnowledgeError)
    assert str(err) == "msg"
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_errors.py -v`

- [ ] **Step 3: Write `errors.py`**

```python
"""Exception hierarchy for rarecell-mcp-knowledge."""


class KnowledgeError(Exception):
    """Base for all rarecell-mcp-knowledge exceptions."""


class BackendUnreachableError(KnowledgeError): ...
class CacheCorruptedError(KnowledgeError): ...
class InvalidQueryError(KnowledgeError): ...
class RateLimitedError(KnowledgeError): ...
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_errors.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/errors.py packages/rarecell-mcp-knowledge/tests/test_errors.py
git commit -m "Add KnowledgeError hierarchy"
```

---

## Task 3: Citation model

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/citation.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_citation.py`

The Citation shape must match the one in `rarecell.profile.schema` so RAG hits compose cleanly. Mirror, don't import (the packages are independent).

- [ ] **Step 1: Write the failing tests**

```python
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit


def test_citation_minimum():
    c = Citation(source_id="pmid:12345", source="europepmc")
    assert c.source == "europepmc"
    assert c.title is None


def test_retrieval_hit_minimum():
    c = Citation(source_id="cellmarker:T_cell:blood", source="cellmarker")
    h = RetrievalHit(
        citation=c, title="T cell markers", snippet="CD3D, CD3E",
        payload={"genes": ["CD3D", "CD3E"]}, source="cellmarker",
    )
    assert h.payload["genes"] == ["CD3D", "CD3E"]
    assert h.retrieved_at  # datetime auto-populated
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_citation.py -v`

- [ ] **Step 3: Write `citation.py`**

```python
"""Citation + RetrievalHit pydantic models."""
from __future__ import annotations
from datetime import datetime, UTC
from typing import Literal
from pydantic import BaseModel, Field


Source = Literal["europepmc", "pubmed", "cellmarker", "panglaodb",
                 "msigdb", "enrichr", "manual", "preset"]


class Citation(BaseModel):
    source_id: str
    source: Source
    title: str | None = None
    url: str | None = None


class RetrievalHit(BaseModel):
    citation: Citation
    title: str
    snippet: str
    payload: dict
    source: Source
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_citation.py -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/citation.py packages/rarecell-mcp-knowledge/tests/test_citation.py
git commit -m "Add Citation + RetrievalHit pydantic models"
```

---

## Task 4: SQLite-backed query cache

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/cache.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_cache.py`

The cache stores query results keyed by `(backend, query_hash)` with a TTL. JSON-serialized payload. Lives at `~/.cache/rarecell/mcp_knowledge.sqlite` by default; tests use `tmp_path`.

- [ ] **Step 1: Write the failing tests**

```python
import time
from pathlib import Path
from rarecell_mcp_knowledge.cache import QueryCache


def test_cache_get_miss(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    assert c.get("europepmc", "foo") is None


def test_cache_set_then_get(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"hits": [1, 2, 3]}, ttl_seconds=60)
    out = c.get("europepmc", "foo")
    assert out == {"hits": [1, 2, 3]}


def test_cache_expires(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"x": 1}, ttl_seconds=0)  # immediate expiry
    time.sleep(0.01)
    assert c.get("europepmc", "foo") is None


def test_cache_overwrites_existing_key(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"x": 1}, ttl_seconds=60)
    c.set("europepmc", "foo", {"x": 2}, ttl_seconds=60)
    assert c.get("europepmc", "foo") == {"x": 2}
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_cache.py -v`

- [ ] **Step 3: Write `cache.py`**

```python
"""SQLite-backed query result cache."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    backend TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (backend, query_hash)
);
"""


class QueryCache:
    """Append-or-overwrite cache of JSON-serializable payloads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)

    def get(self, backend: str, query_hash: str) -> Any | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM cache "
                "WHERE backend = ? AND query_hash = ?",
                (backend, query_hash),
            ).fetchone()
        if row is None:
            return None
        payload, expires_at = row
        if expires_at < time.time():
            return None
        return json.loads(payload)

    def set(
        self, backend: str, query_hash: str, payload: Any,
        ttl_seconds: int = 30 * 24 * 3600,
    ) -> None:
        expires_at = time.time() + ttl_seconds
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(backend, query_hash, payload, expires_at) VALUES (?, ?, ?, ?)",
                (backend, query_hash, json.dumps(payload), expires_at),
            )
```

- [ ] **Step 4: Run, expect 4 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_cache.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/cache.py packages/rarecell-mcp-knowledge/tests/test_cache.py
git commit -m "Add SQLite-backed QueryCache with TTL"
```

---

## Task 5: Europe PMC literature client

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/literature/__init__.py` (empty)
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/literature/client.py`
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/literature/europepmc.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_europepmc.py`

The Europe PMC REST API base URL is `https://www.ebi.ac.uk/europepmc/webservices/rest/`. The search endpoint takes `query=`, `format=json`, `pageSize=`, optional `dateRange=`. Tests use respx to mock the HTTP layer — no live network in CI.

- [ ] **Step 1: Write the failing tests**

```python
import respx
from httpx import Response
from rarecell_mcp_knowledge.literature.europepmc import EuropePMCClient


SEARCH_RESPONSE = {
    "hitCount": 2,
    "resultList": {"result": [
        {"id": "12345", "pmid": "12345", "title": "T cell markers in brain",
         "abstractText": "We identify pan-T markers...",
         "doi": "10.1234/abc", "pubYear": "2023",
         "authorString": "Smith J, Lee K"},
        {"id": "67890", "pmid": "67890", "title": "Microglia in ALS",
         "abstractText": "Microglia release...",
         "doi": "10.5678/def", "pubYear": "2024",
         "authorString": "Garcia M"},
    ]},
}


@respx.mock
def test_search_literature_returns_hits():
    respx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ).mock(return_value=Response(200, json=SEARCH_RESPONSE))

    client = EuropePMCClient()
    hits = client.search("T cell brain", page_size=2)
    assert len(hits) == 2
    assert hits[0].citation.source == "europepmc"
    assert hits[0].citation.source_id == "pmid:12345"
    assert "T cell markers" in hits[0].title


FETCH_RESPONSE = {
    "resultList": {"result": [
        {"id": "12345", "pmid": "12345", "title": "T cell markers in brain",
         "abstractText": "Full abstract here.", "doi": "10.1234/abc",
         "pubYear": "2023", "authorString": "Smith J"},
    ]},
}


@respx.mock
def test_fetch_abstract_returns_full_text():
    respx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ).mock(return_value=Response(200, json=FETCH_RESPONSE))

    client = EuropePMCClient()
    record = client.fetch_abstract("12345")
    assert record.citation.source_id == "pmid:12345"
    assert "Full abstract here." in record.snippet


@respx.mock
def test_search_handles_http_error():
    import pytest
    from rarecell_mcp_knowledge.errors import BackendUnreachableError
    respx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ).mock(return_value=Response(503))

    client = EuropePMCClient()
    with pytest.raises(BackendUnreachableError):
        client.search("foo")
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_europepmc.py -v`

- [ ] **Step 3: Write `literature/client.py` (protocol)**

```python
"""Literature backend protocol."""
from __future__ import annotations
from typing import Protocol
from rarecell_mcp_knowledge.citation import RetrievalHit


class LiteratureBackend(Protocol):
    def search(
        self, query: str, *, year_range: tuple[int, int] | None = None,
        tissue: str | None = None, page_size: int = 10,
    ) -> list[RetrievalHit]: ...

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit: ...
```

- [ ] **Step 4: Write `literature/europepmc.py`**

```python
"""Europe PMC REST client."""
from __future__ import annotations
import httpx
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError


BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _record_to_hit(rec: dict) -> RetrievalHit:
    pmid = rec.get("pmid") or rec.get("id") or ""
    title = rec.get("title", "")
    abstract = rec.get("abstractText", "")
    doi = rec.get("doi")
    url = f"https://europepmc.org/article/MED/{pmid}" if pmid else None
    citation = Citation(
        source_id=f"pmid:{pmid}" if pmid else f"doi:{doi}",
        source="europepmc",
        title=title or None,
        url=url,
    )
    return RetrievalHit(
        citation=citation, title=title, snippet=abstract,
        payload={
            "doi": doi, "year": rec.get("pubYear"),
            "authors": rec.get("authorString"),
        },
        source="europepmc",
    )


class EuropePMCClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0):
        self.base_url = base_url
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/{endpoint}",
                          params=params, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise BackendUnreachableError(f"Europe PMC unreachable: {e}") from e
        if r.status_code >= 500:
            raise BackendUnreachableError(
                f"Europe PMC returned {r.status_code}")
        if r.status_code == 400:
            raise InvalidQueryError(f"Europe PMC rejected query: {r.text[:200]}")
        r.raise_for_status()
        return r.json()

    def search(
        self, query: str, *, year_range: tuple[int, int] | None = None,
        tissue: str | None = None, page_size: int = 10,
    ) -> list[RetrievalHit]:
        q = query
        if tissue:
            q = f"({q}) AND {tissue}"
        if year_range:
            q = f"({q}) AND (FIRST_PDATE:[{year_range[0]}-01-01 TO {year_range[1]}-12-31])"
        data = self._get("search", {
            "query": q, "format": "json", "pageSize": page_size,
            "resultType": "core",
        })
        return [_record_to_hit(r)
                for r in data.get("resultList", {}).get("result", [])]

    def fetch_abstract(self, pmid_or_doi: str) -> RetrievalHit:
        ident = pmid_or_doi.replace("pmid:", "").replace("doi:", "")
        data = self._get("search", {
            "query": f"EXT_ID:{ident}", "format": "json",
            "pageSize": 1, "resultType": "core",
        })
        results = data.get("resultList", {}).get("result", [])
        if not results:
            raise InvalidQueryError(f"No record for {pmid_or_doi}")
        return _record_to_hit(results[0])
```

- [ ] **Step 5: Run, expect 3 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_europepmc.py -v`

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check packages/rarecell-mcp-knowledge/
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/literature/ packages/rarecell-mcp-knowledge/tests/test_europepmc.py
git commit -m "Add Europe PMC literature client with respx-mocked tests"
```

---

## Task 6: Tiny marker fixtures (CellMarker + PanglaoDB TSVs)

**Files:**
- Create: `packages/rarecell-mcp-knowledge/tests/data/cellmarker_tiny.tsv`
- Create: `packages/rarecell-mcp-knowledge/tests/data/panglaodb_tiny.tsv`

These are minimal fixtures for catalog tests; they are NOT downloaded copies of the full database.

- [ ] **Step 1: Create `cellmarker_tiny.tsv`**

Tab-separated, columns: `species`, `tissue_class`, `cell_name`, `marker`, `pmid`. Use this 20-row content:

```
species	tissue_class	cell_name	marker	pmid
Human	Blood	T cell	CD3D	18187658
Human	Blood	T cell	CD3E	18187658
Human	Blood	T cell	CD3G	18187658
Human	Blood	T cell	TRAC	18187658
Human	Blood	B cell	CD19	16410789
Human	Blood	B cell	MS4A1	16410789
Human	Blood	B cell	CD79A	16410789
Human	Blood	NK cell	NCAM1	17139049
Human	Blood	NK cell	NKG7	17139049
Human	Blood	NK cell	GNLY	17139049
Human	Brain	Microglia	CX3CR1	29483971
Human	Brain	Microglia	P2RY12	29483971
Human	Brain	Microglia	TMEM119	29483971
Human	Brain	Astrocyte	GFAP	29483971
Human	Brain	Astrocyte	AQP4	29483971
Human	Brain	Neuron	RBFOX3	29483971
Human	Brain	Neuron	SYT1	29483971
Human	Blood	Monocyte	CD14	16410789
Human	Blood	Monocyte	LYZ	16410789
Human	Blood	Dendritic cell	CLEC9A	24382886
```

Path: `packages/rarecell-mcp-knowledge/tests/data/cellmarker_tiny.tsv`.

- [ ] **Step 2: Create `panglaodb_tiny.tsv`**

Tab-separated, columns: `species`, `official_symbol`, `cell_type`, `nicknames`, `ubiquitousness_index`. PanglaoDB-style. Use this 15-row content:

```
species	official_symbol	cell_type	nicknames	ubiquitousness_index
Hs	CD3D	T cells		0.18
Hs	CD3E	T cells		0.17
Hs	CD8A	T cells (CD8+)		0.10
Hs	FOXP3	T cells (regulatory)		0.05
Hs	MS4A1	B cells		0.08
Hs	CD79A	B cells		0.07
Hs	NCAM1	NK cells		0.12
Hs	NKG7	NK cells		0.11
Hs	CX3CR1	Microglia		0.15
Hs	P2RY12	Microglia		0.06
Hs	GFAP	Astrocytes		0.20
Hs	RBFOX3	Neurons		0.22
Hs	CD14	Monocytes		0.30
Hs	LYZ	Monocytes		0.28
Hs	CLEC9A	Dendritic cells		0.04
```

Path: `packages/rarecell-mcp-knowledge/tests/data/panglaodb_tiny.tsv`.

- [ ] **Step 3: Commit**

```bash
git add packages/rarecell-mcp-knowledge/tests/data/
git commit -m "Add tiny CellMarker + PanglaoDB fixtures for catalog tests"
```

---

## Task 7: Markers catalog (SQLite-backed)

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/markers/__init__.py` (empty)
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/markers/catalog.py`
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/markers/seed.py`
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/markers/client.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pytest
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv


FIXTURES = Path(__file__).parent / "data"


@pytest.fixture
def seeded_catalog(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=FIXTURES / "panglaodb_tiny.tsv",
    )
    return catalog


def test_search_t_cell_blood(seeded_catalog):
    hits = seeded_catalog.search_markers("T cell", tissue="blood")
    sources = {h.source for h in hits}
    assert "cellmarker" in sources or "panglaodb" in sources
    all_markers = {g for h in hits for g in h.payload["genes"]}
    assert "CD3D" in all_markers
    assert "CD3E" in all_markers


def test_search_microglia_brain(seeded_catalog):
    hits = seeded_catalog.search_markers("Microglia", tissue="brain")
    all_markers = {g for h in hits for g in h.payload["genes"]}
    assert "CX3CR1" in all_markers
    assert "P2RY12" in all_markers


def test_search_unknown_returns_empty(seeded_catalog):
    hits = seeded_catalog.search_markers("Cardiomyocyte", tissue="heart")
    assert hits == []


def test_get_canonical_panel_t_cell(seeded_catalog):
    panel = seeded_catalog.get_canonical_panel("T cell")
    assert panel.payload["genes"]
    assert "CD3D" in panel.payload["genes"]
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_catalog.py -v`

- [ ] **Step 3: Write `markers/catalog.py`**

```python
"""SQLite-backed marker catalog. Aggregates CellMarker + PanglaoDB."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit


_SCHEMA = """
CREATE TABLE IF NOT EXISTS markers (
    source TEXT NOT NULL,         -- "cellmarker" or "panglaodb"
    cell_type TEXT NOT NULL,
    tissue TEXT,
    gene TEXT NOT NULL,
    citation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_cell_tissue ON markers (cell_type, tissue);
"""


class MarkersCatalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)

    def insert(
        self, source: str, cell_type: str, tissue: str | None,
        gene: str, citation_id: str | None,
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO markers "
                "(source, cell_type, tissue, gene, citation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, cell_type.lower(), (tissue or "").lower(),
                 gene, citation_id),
            )

    def _ilike(self, s: str) -> str:
        return f"%{s.lower()}%"

    def search_markers(
        self, cell_type: str, tissue: str | None = None,
    ) -> list[RetrievalHit]:
        with sqlite3.connect(self.path) as conn:
            if tissue:
                rows = conn.execute(
                    "SELECT source, cell_type, tissue, gene, citation_id "
                    "FROM markers "
                    "WHERE cell_type LIKE ? AND tissue LIKE ?",
                    (self._ilike(cell_type), self._ilike(tissue)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT source, cell_type, tissue, gene, citation_id "
                    "FROM markers WHERE cell_type LIKE ?",
                    (self._ilike(cell_type),),
                ).fetchall()
        return self._rows_to_hits(rows, cell_type)

    def _rows_to_hits(self, rows: list, query: str) -> list[RetrievalHit]:
        # Group genes per (source, cell_type, tissue)
        grouped: dict[tuple, dict] = {}
        for source, ct, tis, gene, cit_id in rows:
            key = (source, ct, tis)
            g = grouped.setdefault(key, {"genes": set(), "citation_ids": set()})
            g["genes"].add(gene)
            if cit_id:
                g["citation_ids"].add(cit_id)

        hits: list[RetrievalHit] = []
        for (source, ct, tis), agg in grouped.items():
            genes = sorted(agg["genes"])
            cit_ids = sorted(agg["citation_ids"])
            citation = Citation(
                source_id=f"{source}:{ct}:{tis}",
                source=source,
                title=f"{ct} markers ({source})",
            )
            hits.append(RetrievalHit(
                citation=citation,
                title=f"{ct} markers in {tis or 'any tissue'}",
                snippet=", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""),
                payload={"genes": genes, "citation_ids": cit_ids,
                          "cell_type": ct, "tissue": tis},
                source=source,
            ))
        return hits

    def get_canonical_panel(self, cell_type: str) -> RetrievalHit:
        """Aggregate-across-sources canonical panel for a cell type.

        Returns a single hit whose genes are the union over all sources/tissues.
        """
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT gene FROM markers WHERE cell_type LIKE ?",
                (self._ilike(cell_type),),
            ).fetchall()
        genes = sorted(g for (g,) in rows)
        citation = Citation(
            source_id=f"canonical:{cell_type.lower()}",
            source="manual",
            title=f"Canonical {cell_type} panel (rarecell aggregate)",
        )
        return RetrievalHit(
            citation=citation,
            title=f"Canonical {cell_type} panel",
            snippet=", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""),
            payload={"genes": genes, "cell_type": cell_type},
            source="manual",
        )
```

- [ ] **Step 4: Write `markers/seed.py`**

```python
"""Seed the markers catalog from CellMarker + PanglaoDB TSVs."""
from __future__ import annotations
import csv
from pathlib import Path
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog


def seed_catalog_from_tsv(
    catalog: MarkersCatalog,
    *,
    cellmarker_tsv: Path | None = None,
    panglaodb_tsv: Path | None = None,
) -> dict[str, int]:
    """Seed the catalog from TSV files. Returns counts per source."""
    counts = {"cellmarker": 0, "panglaodb": 0}

    if cellmarker_tsv:
        with open(cellmarker_tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("species") != "Human":
                    continue
                catalog.insert(
                    source="cellmarker",
                    cell_type=row["cell_name"],
                    tissue=row.get("tissue_class"),
                    gene=row["marker"],
                    citation_id=f"pmid:{row['pmid']}" if row.get("pmid") else None,
                )
                counts["cellmarker"] += 1

    if panglaodb_tsv:
        with open(panglaodb_tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("species") != "Hs":
                    continue
                catalog.insert(
                    source="panglaodb",
                    cell_type=row["cell_type"],
                    tissue=None,                 # PanglaoDB is tissue-agnostic
                    gene=row["official_symbol"],
                    citation_id=None,
                )
                counts["panglaodb"] += 1

    return counts
```

- [ ] **Step 5: Write `markers/client.py` (protocol + default)**

```python
"""Marker backend protocol."""
from __future__ import annotations
from typing import Protocol
from rarecell_mcp_knowledge.citation import RetrievalHit


class MarkerBackend(Protocol):
    def search_markers(
        self, cell_type: str, tissue: str | None = None,
    ) -> list[RetrievalHit]: ...

    def get_canonical_panel(self, name: str) -> RetrievalHit: ...
```

- [ ] **Step 6: Run, expect 4 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_catalog.py -v`

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check packages/rarecell-mcp-knowledge/
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/markers/ packages/rarecell-mcp-knowledge/tests/test_catalog.py
git commit -m "Add SQLite MarkersCatalog + TSV seeding"
```

---

## Task 8: Enrichr client

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/enrichr.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_enrichr.py`

Enrichr's REST API is documented at `https://maayanlab.cloud/Enrichr/`. The workflow is two-step: POST a gene list to `/addList`, then GET enrichment from `/enrich` with the returned `userListId`.

- [ ] **Step 1: Write the failing tests**

```python
import respx
from httpx import Response
from rarecell_mcp_knowledge.enrichr import enrichr_enrich


@respx.mock
def test_enrichr_two_step_call():
    # Step 1: addList returns userListId
    respx.post("https://maayanlab.cloud/Enrichr/addList").mock(
        return_value=Response(200, json={"userListId": 12345}))
    # Step 2: enrich with userListId
    respx.get("https://maayanlab.cloud/Enrichr/enrich").mock(
        return_value=Response(200, json={
            "GO_Biological_Process_2023": [
                [1, "T cell activation (GO:0042110)", 1e-10, 5.2, 100.0,
                 ["CD3D", "CD3E"], 0.001, 0, 0],
                [2, "B cell activation (GO:0042113)", 1e-8, 3.1, 50.0,
                 ["MS4A1"], 0.01, 0, 0],
            ],
        }))

    results = enrichr_enrich(
        genes=["CD3D", "CD3E", "MS4A1"],
        library="GO_Biological_Process_2023",
    )
    assert len(results) == 2
    assert results[0].title.startswith("T cell activation")
    assert results[0].payload["overlap_genes"] == ["CD3D", "CD3E"]
    assert results[0].source == "enrichr"


@respx.mock
def test_enrichr_empty_response():
    respx.post("https://maayanlab.cloud/Enrichr/addList").mock(
        return_value=Response(200, json={"userListId": 99}))
    respx.get("https://maayanlab.cloud/Enrichr/enrich").mock(
        return_value=Response(200, json={"GO_Biological_Process_2023": []}))

    results = enrichr_enrich(genes=["RANDOM"], library="GO_Biological_Process_2023")
    assert results == []
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_enrichr.py -v`

- [ ] **Step 3: Write `enrichr.py`**

```python
"""Enrichr REST client."""
from __future__ import annotations
import httpx
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError


ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"


def enrichr_enrich(
    *, genes: list[str], library: str, timeout: float = 15.0,
) -> list[RetrievalHit]:
    """Submit a gene list to Enrichr and return enrichment hits."""
    add_url = f"{ENRICHR_BASE}/addList"
    enrich_url = f"{ENRICHR_BASE}/enrich"
    try:
        post = httpx.post(add_url,
                          files={"list": (None, "\n".join(genes)),
                                  "description": (None, "rarecell")},
                          timeout=timeout)
        post.raise_for_status()
        user_list_id = post.json()["userListId"]
        r = httpx.get(enrich_url,
                      params={"userListId": user_list_id,
                              "backgroundType": library},
                      timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise BackendUnreachableError(f"Enrichr unreachable: {e}") from e

    data = r.json().get(library, [])
    hits: list[RetrievalHit] = []
    for row in data:
        # row shape per Enrichr docs:
        # [rank, term, pvalue, zscore, combined_score, overlap_genes, adjusted_p, ...]
        rank, term, pvalue, zscore, combined, overlap, adj_p, *_ = row
        citation = Citation(
            source_id=f"enrichr:{library}:{term}",
            source="enrichr",
            title=term,
        )
        hits.append(RetrievalHit(
            citation=citation,
            title=term,
            snippet=f"pvalue={pvalue:.2e}, combined_score={combined:.1f}",
            payload={
                "rank": rank, "pvalue": pvalue, "zscore": zscore,
                "combined_score": combined, "overlap_genes": overlap,
                "adjusted_pvalue": adj_p, "library": library,
            },
            source="enrichr",
        ))
    return hits
```

- [ ] **Step 4: Run, expect 2 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_enrichr.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/enrichr.py packages/rarecell-mcp-knowledge/tests/test_enrichr.py
git commit -m "Add Enrichr REST client with respx-mocked tests"
```

---

## Task 9: MSigDB client (gene-set retrieval by name)

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/msigdb.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_msigdb.py`

MSigDB has a REST endpoint at `https://www.gsea-msigdb.org/gsea/msigdb/cards/{NAME}.json` returning a gene set's canonical members. v1 scope: fetch a single gene set by name; richer search is deferred.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import respx
from httpx import Response
from rarecell_mcp_knowledge.msigdb import fetch_msigdb_gene_set
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError


MSIGDB_CARD = {
    "standardName": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "description": "Genes up-regulated in response to IFNG.",
    "members": ["IFIT1", "IFIT2", "IFIT3", "STAT1", "OAS1"],
    "pmid": "26771021",
}


@respx.mock
def test_fetch_known_gene_set():
    respx.get(
        "https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_INTERFERON_GAMMA_RESPONSE.json"
    ).mock(return_value=Response(200, json=MSIGDB_CARD))

    hit = fetch_msigdb_gene_set("HALLMARK_INTERFERON_GAMMA_RESPONSE")
    assert hit.citation.source == "msigdb"
    assert hit.payload["genes"] == ["IFIT1", "IFIT2", "IFIT3", "STAT1", "OAS1"]
    assert "IFNG" in hit.snippet


@respx.mock
def test_fetch_unknown_gene_set_raises():
    respx.get(
        "https://www.gsea-msigdb.org/gsea/msigdb/cards/UNKNOWN.json"
    ).mock(return_value=Response(404))

    with pytest.raises(InvalidQueryError):
        fetch_msigdb_gene_set("UNKNOWN")


@respx.mock
def test_fetch_msigdb_unreachable():
    respx.get(
        "https://www.gsea-msigdb.org/gsea/msigdb/cards/X.json"
    ).mock(return_value=Response(503))

    with pytest.raises(BackendUnreachableError):
        fetch_msigdb_gene_set("X")
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_msigdb.py -v`

- [ ] **Step 3: Write `msigdb.py`**

```python
"""MSigDB single-gene-set client."""
from __future__ import annotations
import httpx
from rarecell_mcp_knowledge.citation import Citation, RetrievalHit
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError


MSIGDB_BASE = "https://www.gsea-msigdb.org/gsea/msigdb/cards"


def fetch_msigdb_gene_set(name: str, timeout: float = 15.0) -> RetrievalHit:
    """Fetch a single MSigDB gene set by its standard name."""
    url = f"{MSIGDB_BASE}/{name}.json"
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        raise BackendUnreachableError(f"MSigDB unreachable: {e}") from e
    if r.status_code == 404:
        raise InvalidQueryError(f"MSigDB has no gene set named {name!r}")
    if r.status_code >= 500:
        raise BackendUnreachableError(f"MSigDB returned {r.status_code}")
    r.raise_for_status()
    data = r.json()

    genes = data.get("members") or data.get("geneSymbols") or []
    citation = Citation(
        source_id=f"msigdb:{name}",
        source="msigdb",
        title=data.get("standardName", name),
        url=f"https://www.gsea-msigdb.org/gsea/msigdb/cards/{name}.html",
    )
    return RetrievalHit(
        citation=citation,
        title=data.get("standardName", name),
        snippet=data.get("description", "")[:300],
        payload={"genes": genes,
                  "description": data.get("description"),
                  "pmid": data.get("pmid")},
        source="msigdb",
    )
```

- [ ] **Step 4: Run, expect 3 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_msigdb.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/msigdb.py packages/rarecell-mcp-knowledge/tests/test_msigdb.py
git commit -m "Add MSigDB gene-set client with respx-mocked tests"
```

---

## Task 10: FastMCP server entry point

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/server.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_server.py`

FastMCP advertises tools by decorating functions on a `FastMCP` app instance. The server.py wires together all five tools using the clients built in Tasks 5, 7, 8, 9.

- [ ] **Step 1: Write the failing tests**

```python
"""In-process smoke test of the FastMCP server.

Calls the tool functions directly (bypasses the MCP transport — that's
what FastMCP's in-process testing is for). Verifies the tools accept the
documented arguments and return well-shaped output.
"""
from pathlib import Path
import pytest
import respx
from httpx import Response

from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv
from rarecell_mcp_knowledge.server import build_app


FIXTURES = Path(__file__).parent / "data"


@pytest.fixture
def app(tmp_path):
    catalog = MarkersCatalog(tmp_path / "markers.sqlite")
    seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=FIXTURES / "cellmarker_tiny.tsv",
        panglaodb_tsv=FIXTURES / "panglaodb_tiny.tsv",
    )
    return build_app(catalog=catalog, cache_path=tmp_path / "cache.sqlite")


def test_app_has_five_tools(app):
    names = sorted(app.list_tool_names())
    assert "search_literature" in names
    assert "fetch_abstract" in names
    assert "search_markers" in names
    assert "get_canonical_panel" in names
    assert "enrichr_enrich" in names


@respx.mock
def test_search_literature_via_app(app):
    respx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ).mock(return_value=Response(200, json={
        "hitCount": 1,
        "resultList": {"result": [{
            "id": "111", "pmid": "111", "title": "T cell",
            "abstractText": "abstract", "doi": "10.1/x", "pubYear": "2024",
            "authorString": "X",
        }]},
    }))
    hits = app.call_tool("search_literature", {"query": "T cell"})
    assert len(hits) == 1
    assert hits[0]["citation"]["source_id"] == "pmid:111"


def test_search_markers_via_app(app):
    hits = app.call_tool("search_markers",
                          {"cell_type": "T cell", "tissue": "blood"})
    all_genes = {g for h in hits for g in h["payload"]["genes"]}
    assert "CD3D" in all_genes
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_server.py -v`

- [ ] **Step 3: Write `server.py`**

```python
"""FastMCP server: wires up literature + markers + Enrichr + MSigDB tools."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from rarecell_mcp_knowledge.cache import QueryCache
from rarecell_mcp_knowledge.citation import RetrievalHit
from rarecell_mcp_knowledge.literature.europepmc import EuropePMCClient
from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
from rarecell_mcp_knowledge.enrichr import enrichr_enrich
from rarecell_mcp_knowledge.msigdb import fetch_msigdb_gene_set


def _hash(query: dict) -> str:
    return hashlib.sha256(
        json.dumps(query, sort_keys=True).encode()).hexdigest()


def _serialize(hit_or_hits):
    if isinstance(hit_or_hits, RetrievalHit):
        return hit_or_hits.model_dump(mode="json")
    return [h.model_dump(mode="json") for h in hit_or_hits]


class KnowledgeApp:
    """Lightweight wrapper that exposes tools as callable methods.

    The FastMCP server itself is constructed in build_app() below; this class
    holds state (catalog, cache, clients) and provides the tool implementations.
    """

    def __init__(self, catalog: MarkersCatalog, cache: QueryCache,
                 literature: EuropePMCClient | None = None):
        self.catalog = catalog
        self.cache = cache
        self.literature = literature or EuropePMCClient()

    # ----- tools -----
    def search_literature(self, query: str, year_range: list[int] | None = None,
                          tissue: str | None = None, page_size: int = 10):
        key = _hash({"q": query, "yr": year_range, "tissue": tissue, "ps": page_size})
        cached = self.cache.get("europepmc", key)
        if cached is not None:
            return cached
        yr = tuple(year_range) if year_range else None
        hits = self.literature.search(query, year_range=yr, tissue=tissue,
                                      page_size=page_size)
        out = _serialize(hits)
        self.cache.set("europepmc", key, out)
        return out

    def fetch_abstract(self, pmid_or_doi: str):
        key = _hash({"fetch": pmid_or_doi})
        cached = self.cache.get("europepmc", key)
        if cached is not None:
            return cached
        hit = self.literature.fetch_abstract(pmid_or_doi)
        out = _serialize(hit)
        self.cache.set("europepmc", key, out)
        return out

    def search_markers(self, cell_type: str, tissue: str | None = None):
        # No cache — local SQLite is already fast
        return _serialize(self.catalog.search_markers(cell_type, tissue))

    def get_canonical_panel(self, name: str):
        return _serialize(self.catalog.get_canonical_panel(name))

    def enrichr_enrich(self, genes: list[str], library: str):
        key = _hash({"genes": sorted(genes), "lib": library})
        cached = self.cache.get("enrichr", key)
        if cached is not None:
            return cached
        hits = enrichr_enrich(genes=genes, library=library)
        out = _serialize(hits)
        self.cache.set("enrichr", key, out)
        return out

    def fetch_msigdb_gene_set(self, name: str):
        key = _hash({"msigdb": name})
        cached = self.cache.get("msigdb", key)
        if cached is not None:
            return cached
        hit = fetch_msigdb_gene_set(name)
        out = _serialize(hit)
        self.cache.set("msigdb", key, out)
        return out


def build_app(*, catalog: MarkersCatalog, cache_path: Path):
    """Build an in-process app object exposing the tools.

    Returns an object with:
      .list_tool_names() -> list[str]
      .call_tool(name, kwargs) -> result
    """
    cache = QueryCache(cache_path)
    knowledge = KnowledgeApp(catalog=catalog, cache=cache)

    tools = {
        "search_literature": knowledge.search_literature,
        "fetch_abstract": knowledge.fetch_abstract,
        "search_markers": knowledge.search_markers,
        "get_canonical_panel": knowledge.get_canonical_panel,
        "enrichr_enrich": knowledge.enrichr_enrich,
        "fetch_msigdb_gene_set": knowledge.fetch_msigdb_gene_set,
    }

    class _App:
        def list_tool_names(self) -> list[str]:
            return list(tools.keys())

        def call_tool(self, name: str, kwargs: dict):
            return tools[name](**kwargs)

        # Expose internals for the CLI to register with FastMCP
        knowledge_app = knowledge
        tool_map = tools

    return _App()


def build_fastmcp_app(*, catalog: MarkersCatalog, cache_path: Path):
    """Build the production FastMCP server with tools registered."""
    from fastmcp import FastMCP
    inner = build_app(catalog=catalog, cache_path=cache_path)
    mcp = FastMCP("rarecell-mcp-knowledge")
    for name, fn in inner.tool_map.items():
        mcp.tool(name)(fn)
    return mcp
```

The test surface (`build_app`) is a deliberately thin in-process wrapper that mirrors the FastMCP tool API; that decouples the tool tests from FastMCP's transport. The `build_fastmcp_app` function is what `cli.py` uses to actually serve.

- [ ] **Step 4: Run, expect 3 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_server.py -v`

The first test asserts the 5 documented tools. We actually register 6 (the spec mentions `fetch_msigdb_gene_set` later — keep it in but the test only checks the 5 advertised tools exist, not that there are exactly 5).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check packages/rarecell-mcp-knowledge/
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/server.py packages/rarecell-mcp-knowledge/tests/test_server.py
git commit -m "Add KnowledgeApp + FastMCP server wrapper with tool registry"
```

---

## Task 11: CLI (`serve` + `seed`)

**Files:**
- Create: `packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/cli.py`
- Create: `packages/rarecell-mcp-knowledge/tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
from rarecell_mcp_knowledge.cli import main_with_args


FIXTURES = Path(__file__).parent / "data"


def test_seed_command(tmp_path: Path, capsys):
    home = tmp_path / "home"
    db = home / ".cache/rarecell/markers.sqlite"
    rc = main_with_args([
        "seed",
        "--cellmarker-tsv", str(FIXTURES / "cellmarker_tiny.tsv"),
        "--panglaodb-tsv", str(FIXTURES / "panglaodb_tiny.tsv"),
        "--catalog-path", str(db),
    ])
    assert rc == 0
    assert db.exists()
    out = capsys.readouterr().out
    assert "cellmarker" in out
    assert "panglaodb" in out


def test_serve_subcommand_help(capsys):
    rc = main_with_args(["serve", "--help"])
    # --help prints and exits with 0
    assert rc == 0
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_cli.py -v`

- [ ] **Step 3: Write `cli.py`**

```python
"""rarecell-mcp-knowledge CLI: serve | seed."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence


def _default_catalog_path() -> Path:
    return Path.home() / ".cache/rarecell/markers.sqlite"


def _default_cache_path() -> Path:
    return Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"


def _seed(args: argparse.Namespace) -> int:
    from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
    from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

    catalog = MarkersCatalog(Path(args.catalog_path))
    counts = seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=Path(args.cellmarker_tsv) if args.cellmarker_tsv else None,
        panglaodb_tsv=Path(args.panglaodb_tsv) if args.panglaodb_tsv else None,
    )
    print(f"Seeded: cellmarker={counts['cellmarker']} panglaodb={counts['panglaodb']}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
    from rarecell_mcp_knowledge.server import build_fastmcp_app

    catalog = MarkersCatalog(Path(args.catalog_path))
    app = build_fastmcp_app(catalog=catalog, cache_path=Path(args.cache_path))
    app.run()  # FastMCP's blocking run loop
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rarecell-mcp-knowledge")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve_p = sub.add_parser("serve", help="Run the FastMCP server")
    serve_p.add_argument("--catalog-path", default=str(_default_catalog_path()))
    serve_p.add_argument("--cache-path", default=str(_default_cache_path()))
    serve_p.set_defaults(func=_serve)

    seed_p = sub.add_parser("seed", help="Seed the marker catalog from TSV files")
    seed_p.add_argument("--catalog-path", default=str(_default_catalog_path()))
    seed_p.add_argument("--cellmarker-tsv", default=None,
                          help="Path to CellMarker 2.0 TSV; omit to skip")
    seed_p.add_argument("--panglaodb-tsv", default=None,
                          help="Path to PanglaoDB TSV; omit to skip")
    seed_p.set_defaults(func=_seed)

    return p


def main_with_args(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main() -> int:
    return main_with_args(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run, expect 2 pass**

Run: `uv run pytest packages/rarecell-mcp-knowledge/tests/test_cli.py -v`

- [ ] **Step 5: Verify the console script is wired**

```bash
uv run rarecell-mcp-knowledge --help
```

Expected: prints usage with `serve` and `seed` subcommands. No error.

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell-mcp-knowledge/src/rarecell_mcp_knowledge/cli.py packages/rarecell-mcp-knowledge/tests/test_cli.py
git commit -m "Add rarecell-mcp-knowledge CLI: serve + seed subcommands"
```

---

## Task 12: Wire into root CI workflows

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `.github/workflows/lint.yml`

The root CI workflows currently test only `packages/rarecell/tests`. Extend them to also run `packages/rarecell-mcp-knowledge/tests`.

- [ ] **Step 1: Read the current `.github/workflows/test.yml`**

Use the Read tool. Confirm it currently runs `uv run pytest packages/rarecell/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py -v` for the `unit` job.

- [ ] **Step 2: Update the unit job to include the new package's tests**

Change the `unit` job's pytest invocation to:

```yaml
- run: uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py -v
```

- [ ] **Step 3: Verify lint already covers the new package**

Read `.github/workflows/lint.yml`. The existing lines `uv run ruff check .` and `uv run ruff format --check .` already cover any path under the repo root, including `packages/rarecell-mcp-knowledge/`. Confirm no changes needed.

- [ ] **Step 4: Run the full test suite locally to confirm**

```bash
uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py -v
```
Expected: all tests pass (the previous 53 from Plan 1 + the new ones from Tasks 2, 3, 4, 5, 7, 8, 9, 10, 11).

- [ ] **Step 5: Lint clean check**

```bash
uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "Wire rarecell-mcp-knowledge tests into CI"
```

---

## Task 13: README polish + integration smoke

**Files:**
- Modify: `packages/rarecell-mcp-knowledge/README.md`
- Modify: root `README.md`

- [ ] **Step 1: Replace the package README with a complete usage walkthrough**

Replace `packages/rarecell-mcp-knowledge/README.md` with:

```markdown
# rarecell-mcp-knowledge

FastMCP server exposing literature retrieval (Europe PMC) and marker-database
retrieval (CellMarker 2.0, PanglaoDB, MSigDB, Enrichr) behind a single MCP
surface.

## Install

```bash
pip install rarecell-mcp-knowledge
```

## Seed the local marker catalog

The local SQLite catalog (`~/.cache/rarecell/markers.sqlite`) is built from
CellMarker 2.0 + PanglaoDB TSV downloads:

```bash
rarecell-mcp-knowledge seed \
  --cellmarker-tsv /path/to/Cell_marker_Human.tsv \
  --panglaodb-tsv /path/to/PanglaoDB_markers_27_Mar_2020.tsv
```

Source URLs (download manually for v0.1):
- CellMarker 2.0: <http://yikedaxue.slwshop.cn/CellMarker_download_files/file/Cell_marker_Human.xlsx>
- PanglaoDB: <https://panglaodb.se/markers.html>

## Run the server

```bash
rarecell-mcp-knowledge serve
```

Stdio MCP server. Wire into Claude Desktop / Claude Code by adding this to
the MCP client config:

```json
{
  "mcpServers": {
    "rarecell-knowledge": {
      "command": "rarecell-mcp-knowledge",
      "args": ["serve"]
    }
  }
}
```

## Tools advertised

| Tool | Purpose |
|------|---------|
| `search_literature(query, year_range?, tissue?, page_size?)` | Europe PMC search |
| `fetch_abstract(pmid_or_doi)` | Fetch a single abstract |
| `search_markers(cell_type, tissue?)` | Local catalog query |
| `get_canonical_panel(name)` | Aggregate marker panel across sources |
| `enrichr_enrich(genes, library)` | Enrichr gene set enrichment |
| `fetch_msigdb_gene_set(name)` | MSigDB single gene set lookup |

## Cache

Query results are cached at `~/.cache/rarecell/mcp_knowledge.sqlite` with a
30-day TTL. Delete the file to force fresh fetches.

## Status

Pre-release v0.x.
```

- [ ] **Step 2: Add a top-level mention in the root README**

Read root `README.md`. Find the section describing the package layout. Append a bullet describing `rarecell-mcp-knowledge`:

```markdown
- `packages/rarecell-mcp-knowledge/` — FastMCP server for literature + marker
  retrieval (CellMarker, PanglaoDB, MSigDB, Enrichr, Europe PMC). Consumable
  from any MCP client.
```

- [ ] **Step 3: Commit**

```bash
git add packages/rarecell-mcp-knowledge/README.md README.md
git commit -m "Document rarecell-mcp-knowledge in package README and root README"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Plan coverage |
|---|---|
| §3.4 single FastMCP server | Tasks 10–11 |
| Tool: `search_literature` | Task 10 (server) + Task 5 (backend) |
| Tool: `fetch_abstract` | Task 10 + Task 5 |
| Tool: `search_markers` | Task 10 + Task 7 |
| Tool: `get_canonical_panel` | Task 10 + Task 7 |
| Tool: `enrichr_enrich` | Task 10 + Task 8 |
| Local CellMarker+PanglaoDB SQLite, downloaded on first run | Tasks 6 (fixtures), 7 (catalog), 11 (seed CLI). The "first-run download" is deferred — the CLI takes explicit `--cellmarker-tsv` / `--panglaodb-tsv` paths so users do the download themselves. This is a deliberate scope cut for v0.1 (avoids brittle download URLs in code). README documents the manual flow. |
| Live MSigDB REST | Task 9 |
| Live Enrichr REST | Task 8 |
| Citation propagation (every hit carries Citation) | Task 3 |
| 30-day query cache | Task 4 |

**Bonus tool not in spec:** `fetch_msigdb_gene_set` — useful corollary to MSigDB integration; harmless inclusion.

**Placeholder scan:** none — every step has complete code or a precise instruction.

**Type consistency:**
- `RetrievalHit` consistent across tasks 3, 5, 7, 8, 9, 10.
- `Citation` consistent throughout.
- `MarkersCatalog` (init signature, `insert`, `search_markers`, `get_canonical_panel`) consistent across tasks 7 and 10.
- `KnowledgeApp` tool methods accept the documented argument names.

**Scope notes:**
- The "first-run auto-download" of CellMarker/PanglaoDB is deferred. README documents manual download.
- `pubmed` source backend is not implemented separately — the spec said "evaluate community PubMed/Europe PMC MCP servers; fall back to a thin Europe PMC REST wrapper if no community option meets a small bar." Plan 2 takes the Europe PMC fallback path directly (cheaper than a community-MCP integration audit). The `pubmed` Literal in the Citation Source type is left in place for forward compatibility.
- No "live" integration tests against actual Europe PMC / Enrichr / MSigDB are included in CI. All HTTP is respx-mocked. A nightly cron that hits the real APIs is a future-CI consideration.
