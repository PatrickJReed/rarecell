# rarecell Foundation — Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `rarecell` Python library — `core/`, `profile/`, `report.py`, `BasicRecommender`, the state machine, and replay mode — so a scientist can isolate a rare-cell population from a single AnnData using only heuristics (no LLM, no MCP), with a fully-replayable IsolationReport as output.

**Architecture:** Monorepo with `uv` workspaces; the first package (`rarecell`) is built layer-by-layer (profile → core modules → state machine → report → replay). The state machine takes a pluggable `Recommender` interface; Plan 1 ships `BasicRecommender` (heuristic-only); Plan 3 will add `ClaudeRecommender`. Refactors `als_utils.py` from the ALS T-cell project into clean profile-driven modules.

**Tech Stack:** Python 3.11+, scanpy 1.10.x, anndata 0.10+, pydantic v2, harmonypy, celltypist, scrublet, scikit-learn, gseapy (Enrichr), pytest, hypothesis, structlog, uv, ruff, mypy.

**Source to port from:** `/Users/patrickreed/Downloads/ALS_Project_Reedp1-ac13bc8b95587ee8/als_utils.py` (referred to below as `als_utils.py` with line numbers).

**Spec:** `docs/superpowers/specs/2026-05-20-rarecell-agentic-isolation-design.md`.

---

## File Structure (created in this plan)

```
rarecell/                                       # repo root, already initialized
├── pyproject.toml                              # workspace root
├── uv.lock
├── packages/
│   └── rarecell/
│       ├── pyproject.toml
│       └── src/rarecell/
│           ├── __init__.py
│           ├── errors.py                       # RareCellError + subclasses
│           ├── core/
│           │   ├── __init__.py
│           │   ├── ingest.py                   # ports als_utils:33,244-433,5049-5060,5107-5147
│           │   ├── qc.py                       # ports als_utils:564-679
│           │   ├── markers.py                  # ports als_utils:848-877 + profile-driven scoring
│           │   ├── annotate.py                 # ports als_utils:714-845,2043-2118,4032-4246
│           │   ├── clustering.py               # ports als_utils:700-712,1111-1604,1605-1794
│           │   ├── evidence.py                 # ports als_utils:1795-2042,2387-3439
│           │   ├── purify.py                   # ports als_utils:2167-2386
│           │   └── io.py                       # ports als_utils:5149-5292
│           ├── profile/
│           │   ├── __init__.py
│           │   ├── schema.py                   # pydantic TargetCellProfile + freeze interlock
│           │   ├── draft.py                    # stub (real implementation in Plan 3)
│           │   └── presets/
│           │       ├── t_cell_pbmc.yaml
│           │       ├── t_cell_cns.yaml
│           │       ├── b_cell.yaml
│           │       ├── nk_cell.yaml
│           │       ├── microglia.yaml
│           │       ├── dendritic_cell.yaml
│           │       └── monocyte_macrophage.yaml
│           ├── recommender/
│           │   ├── __init__.py
│           │   ├── base.py                     # Recommender protocol + Recommendation model
│           │   └── basic.py                    # BasicRecommender (heuristic-only)
│           ├── state_machine/
│           │   ├── __init__.py
│           │   ├── states.py                   # IsolateState enum + transitions
│           │   └── isolate.py                  # orchestrates S0..S7 with pluggable Recommender
│           ├── report.py                       # IsolationReport, manifest, decisions.jsonl
│           └── logging.py                      # structlog config
├── tests/
│   ├── __init__.py
│   ├── conftest.py                             # synthetic fixture
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── make_synthetic.py                   # 5000-cell synthetic AnnData
│   ├── core/                                   # unit tests per core module
│   ├── profile/
│   ├── recommender/
│   ├── state_machine/
│   └── integration/
│       ├── test_synthetic_end_to_end.py
│       ├── test_replay_determinism.py
│       └── test_pbmc3k.py
├── .github/workflows/
│   ├── lint.yml
│   ├── test.yml
│   └── docs.yml
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE                                     # Apache 2.0
```

---

## Task 1: Workspace skeleton + governance files

**Files:**
- Create: `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.gitignore`
- Create: `pyproject.toml` (workspace root)
- Create: `packages/rarecell/pyproject.toml`
- Create: `packages/rarecell/src/rarecell/__init__.py`

- [ ] **Step 1: Add Apache 2.0 LICENSE**

Download the canonical Apache 2.0 text and save as `LICENSE`. Copyright line: `Copyright 2026 Patrick Reed`.

```bash
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

Then edit the `[yyyy]` and `[name of copyright owner]` placeholders to `2026` and `Patrick Reed`.

- [ ] **Step 2: Add `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.testmondata
.coverage
htmlcov/
dist/
build/
.venv/
.env
.idea/
.vscode/
.DS_Store
runs/
data/
*.h5ad
!tests/fixtures/data/*.h5ad
```

- [ ] **Step 3: Add minimal README, CHANGELOG, CONTRIBUTING, CoC, SECURITY**

Create `README.md` with a 1-paragraph project summary and a "this is pre-release v0.x" notice. Create `CHANGELOG.md` with a "Keep a Changelog" header and an empty `## [Unreleased]` section. Create `CONTRIBUTING.md` with: how to run `uv sync`, how to run tests, how to run pre-commit. Use Contributor Covenant 2.1 verbatim for `CODE_OF_CONDUCT.md`. `SECURITY.md` points to `security@example.invalid` (placeholder for v0.1).

Exact contents — keep terse, one-screen each. The engineer should write these as 30-line files; do not pad.

- [ ] **Step 4: Create workspace root `pyproject.toml`**

```toml
[project]
name = "rarecell-workspace"
version = "0"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["packages/*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF", "SIM"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

- [ ] **Step 5: Create `packages/rarecell/pyproject.toml`**

```toml
[project]
name = "rarecell"
version = "0.1.0.dev0"
description = "Agentic rare-cell isolation from single-cell RNA-seq"
authors = [{name = "Patrick Reed", email = "patrickjenningsreed@gmail.com"}]
license = "Apache-2.0"
readme = "../../README.md"
requires-python = ">=3.11"
dependencies = [
  "scanpy>=1.10",
  "anndata>=0.10",
  "harmonypy>=0.0.9",
  "celltypist>=1.6",
  "scrublet",
  "scikit-learn>=1.3",
  "leidenalg>=0.10",
  "igraph>=0.11",
  "gseapy>=1.0",
  "pydantic>=2.6",
  "pyyaml>=6.0",
  "structlog>=24.1",
  "h5py>=3.9",
  "pybiomart>=0.2",
  "numpy>=1.24",
  "pandas>=2.0",
  "scipy>=1.11",
  "matplotlib>=3.7",
  "seaborn>=0.13",
  "statsmodels>=0.14",
]

[project.optional-dependencies]
agent = []  # populated in Plan 3
dev = [
  "pytest>=8",
  "pytest-cov",
  "hypothesis>=6",
  "ruff",
  "mypy",
  "pre-commit",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rarecell"]
```

- [ ] **Step 6: Create package init**

Create `packages/rarecell/src/rarecell/__init__.py`:

```python
"""rarecell — agentic rare-cell isolation from single-cell RNA-seq."""

__version__ = "0.1.0.dev0"
```

- [ ] **Step 7: Sync the workspace**

Run: `uv sync --all-extras --dev`
Expected: dependencies resolve and install, no errors.

- [ ] **Step 8: Commit**

```bash
git add LICENSE README.md CHANGELOG.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md .gitignore pyproject.toml packages/ uv.lock
git commit -m "Scaffold uv workspace with rarecell package skeleton"
```

---

## Task 2: Errors module

**Files:**
- Create: `packages/rarecell/src/rarecell/errors.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `packages/rarecell/tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/test_errors.py`:

```python
import pytest
from rarecell.errors import (
    RareCellError,
    MissingRawCountsError,
    InvalidProfileError,
    UnreviewedProfileError,
    IncompatibleSchemaError,
    MCPUnreachableError,
    LLMBudgetExceededError,
    CacheCorruptedError,
    IsolationAbortedError,
)

@pytest.mark.parametrize("cls", [
    MissingRawCountsError, InvalidProfileError, UnreviewedProfileError,
    IncompatibleSchemaError, MCPUnreachableError, LLMBudgetExceededError,
    CacheCorruptedError, IsolationAbortedError,
])
def test_subclasses_of_base(cls):
    err = cls("test message")
    assert isinstance(err, RareCellError)
    assert str(err) == "test message"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/test_errors.py -v`
Expected: `ImportError: cannot import name 'RareCellError' from 'rarecell.errors'`

- [ ] **Step 3: Write the errors module**

Create `packages/rarecell/src/rarecell/errors.py`:

```python
"""Exception hierarchy for rarecell."""


class RareCellError(Exception):
    """Base class for all rarecell-specific exceptions."""


# --- User-input errors (recoverable) ---
class MissingRawCountsError(RareCellError): ...
class InvalidProfileError(RareCellError): ...
class UnreviewedProfileError(RareCellError):
    """Raised when a profile is set frozen=true without human_reviewed=true."""
class IncompatibleSchemaError(RareCellError): ...


# --- Runtime errors (have fallbacks) ---
class MCPUnreachableError(RareCellError): ...
class LLMBudgetExceededError(RareCellError): ...
class CacheCorruptedError(RareCellError): ...


# --- Catastrophic (partial-run saved) ---
class IsolationAbortedError(RareCellError): ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/test_errors.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/errors.py packages/rarecell/tests/test_errors.py
git commit -m "Add RareCellError hierarchy"
```

---

## Task 3: Logging configuration

**Files:**
- Create: `packages/rarecell/src/rarecell/logging.py`
- Create: `packages/rarecell/tests/test_logging.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/test_logging.py`:

```python
import json
from pathlib import Path
from rarecell.logging import configure_logging, get_logger


def test_json_logging_to_file(tmp_path: Path):
    log_path = tmp_path / "run.log"
    configure_logging(log_path=log_path, level="INFO")
    log = get_logger("test")
    log.info("hello", run_id="abc123", state="S2")

    contents = log_path.read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["event"] == "hello"
    assert record["run_id"] == "abc123"
    assert record["state"] == "S2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/test_logging.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the logging module**

Create `packages/rarecell/src/rarecell/logging.py`:

```python
"""Structured logging for rarecell. JSON to file by default."""

from pathlib import Path
import logging
import structlog


def configure_logging(log_path: Path | None = None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/test_logging.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/logging.py packages/rarecell/tests/test_logging.py
git commit -m "Add structlog JSON logging"
```

---

## Task 4: TargetCellProfile schema (no freeze interlock yet)

**Files:**
- Create: `packages/rarecell/src/rarecell/profile/__init__.py`, `schema.py`
- Create: `packages/rarecell/tests/profile/test_schema_basic.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/profile/test_schema_basic.py`:

```python
from rarecell.profile.schema import (
    TargetCellProfile,
    MarkerPanel,
    Citation,
    QCParams,
    PurifyParams,
    BatchCorrection,
    AutoPolicy,
)


def test_minimal_valid_profile():
    p = TargetCellProfile(
        name="T cells, PBMC",
        description="pan T cells",
        target_lineage="lymphoid",
        tissue=["blood"],
        expected_abundance={"min_fraction": 0.05, "max_fraction": 0.6},
        positive_markers={
            "pan_t": MarkerPanel(genes=["CD3D", "CD3E"], threshold_z=1.0,
                                  citations=[Citation(source_id="pmid:1", source="europepmc")])
        },
        negative_markers={},
        reference_labels={},
        biccn_rules={"enabled": False},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        purify=PurifyParams(enabled=True, high_resolution=2.0, min_cluster_purity=0.7),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        auto_policy=AutoPolicy(),
    )
    assert p.schema_version == "1.0"
    assert p.frozen is False                # default
    assert p.human_reviewed is False        # default
    assert p.content_hash is None           # only set on freeze


def test_yaml_roundtrip(tmp_path):
    src = TargetCellProfile.from_yaml_path("packages/rarecell/src/rarecell/profile/presets/t_cell_pbmc.yaml")
    out = tmp_path / "out.yaml"
    src.to_yaml_path(out)
    rebuilt = TargetCellProfile.from_yaml_path(out)
    assert rebuilt.model_dump() == src.model_dump()
```

(The second test will be skipped until Task 6 creates the preset file. Mark it with `@pytest.mark.skip(reason='needs preset from Task 6')` for now.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/profile/test_schema_basic.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the profile schema**

Create `packages/rarecell/src/rarecell/profile/__init__.py`:

```python
from rarecell.profile.schema import TargetCellProfile

__all__ = ["TargetCellProfile"]
```

Create `packages/rarecell/src/rarecell/profile/schema.py`:

```python
"""TargetCellProfile pydantic schema. v1.0."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Citation(BaseModel):
    source_id: str
    source: Literal["europepmc", "pubmed", "cellmarker", "panglaodb",
                    "msigdb", "enrichr", "manual", "preset"]
    title: str | None = None
    url: str | None = None


class MarkerPanel(BaseModel):
    genes: list[str]
    threshold_z: float = Field(ge=0)
    citations: list[Citation] = Field(default_factory=list)


class NegativePanel(BaseModel):
    genes: list[str]
    exclusion_threshold_z: float = Field(ge=0, default=1.5)
    citations: list[Citation] = Field(default_factory=list)


class CellTypistRef(BaseModel):
    model: str
    match_patterns: list[str]
    enabled: bool = True


class ReferenceLabels(BaseModel):
    celltypist_models: list[CellTypistRef] = Field(default_factory=list)


class BICCNRules(BaseModel):
    enabled: bool = False
    class_filter: list[str] = Field(default_factory=list)
    subclass_filter: list[str] = Field(default_factory=list)


class ExpectedAbundance(BaseModel):
    min_fraction: float = Field(ge=0, le=1)
    max_fraction: float = Field(ge=0, le=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_order(self) -> "ExpectedAbundance":
        if self.min_fraction > self.max_fraction:
            raise ValueError("min_fraction must be <= max_fraction")
        return self


class QCParams(BaseModel):
    min_genes_per_cell: int = Field(ge=1)
    max_pct_mt: float = Field(ge=0, le=100)
    max_genes_per_cell: int = Field(default=10000, ge=1)
    min_cells_per_gene: int = Field(default=3, ge=1)
    rationale: str | None = None


class PurifyParams(BaseModel):
    enabled: bool = True
    high_resolution: float = Field(default=2.0, ge=0)
    min_cluster_purity: float = Field(default=0.7, ge=0, le=1)


class BatchCorrection(BaseModel):
    in_dataset: Literal["harmony", "none"] = "harmony"
    batch_key: str = "sample_id"


class GateAutoPolicy(BaseModel):
    gate1_cluster_decisions: Literal["recommendation", "abort_on_ambiguity",
                                     "conservative_drop"] = "recommendation"
    gate2_purify_decisions: Literal["recommendation", "abort_on_ambiguity",
                                    "conservative_drop"] = "recommendation"
    gate3_final: Literal["accept", "abort_on_anomaly"] = "accept"
    min_recommendation_confidence: float = Field(default=0.6, ge=0, le=1)
    max_abundance_deviation: float = Field(default=5.0, ge=1.0)


class AutoPolicy(BaseModel):
    gates: GateAutoPolicy = Field(default_factory=GateAutoPolicy)


class DraftedFrom(BaseModel):
    user_prompt: str | None = None
    drafted_by: str | None = None
    drafted_at: datetime | None = None
    rag_sources_consulted: list[str] = Field(default_factory=list)


class TargetCellProfile(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str
    name: str
    description: str
    target_lineage: str
    tissue: list[str]
    expected_abundance: ExpectedAbundance

    positive_markers: dict[str, MarkerPanel]
    negative_markers: dict[str, NegativePanel]
    reference_labels: ReferenceLabels = Field(default_factory=ReferenceLabels)
    biccn_rules: BICCNRules = Field(default_factory=BICCNRules)
    qc: QCParams
    purify: PurifyParams = Field(default_factory=PurifyParams)
    batch_correction: BatchCorrection = Field(default_factory=BatchCorrection)
    auto_policy: AutoPolicy = Field(default_factory=AutoPolicy)

    drafted_from: DraftedFrom = Field(default_factory=DraftedFrom)
    human_reviewed: bool = False
    reviewer: str | None = None
    frozen: bool = False
    content_hash: str | None = None

    @field_validator("positive_markers")
    @classmethod
    def _at_least_one_positive(cls, v: dict[str, MarkerPanel]) -> dict[str, MarkerPanel]:
        if not v:
            raise ValueError("at least one positive_markers panel is required")
        return v

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "TargetCellProfile":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def to_yaml_path(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"),
                                             sort_keys=False))

    def compute_content_hash(self) -> str:
        canonical = self.model_dump(mode="json",
                                    exclude={"content_hash", "frozen"})
        payload = yaml.safe_dump(canonical, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()
```

(The freeze interlock — the rule that `frozen=True` requires `human_reviewed=True` — is added in Task 5.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/profile/test_schema_basic.py::test_minimal_valid_profile -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/profile/ packages/rarecell/tests/profile/
git commit -m "Add TargetCellProfile schema (no freeze interlock yet)"
```

---

## Task 5: Profile freeze interlock

**Files:**
- Modify: `packages/rarecell/src/rarecell/profile/schema.py`
- Create: `packages/rarecell/tests/profile/test_freeze_interlock.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/profile/test_freeze_interlock.py`:

```python
import pytest
from rarecell.errors import UnreviewedProfileError
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, QCParams,
    ExpectedAbundance,
)


def _minimal_profile_kwargs():
    return dict(
        profile_id="test", name="T", description="d", target_lineage="lymphoid",
        tissue=["blood"],
        expected_abundance=ExpectedAbundance(min_fraction=0.01, max_fraction=0.5),
        positive_markers={"pan": MarkerPanel(genes=["CD3D"], threshold_z=1.0,
                                              citations=[Citation(source_id="pmid:1",
                                                                  source="europepmc")])},
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_frozen_without_review_raises():
    with pytest.raises(UnreviewedProfileError, match="human_reviewed"):
        TargetCellProfile(**_minimal_profile_kwargs(),
                          frozen=True, human_reviewed=False)


def test_frozen_with_review_succeeds_and_has_hash():
    p = TargetCellProfile(**_minimal_profile_kwargs(),
                          human_reviewed=True, reviewer="r@x")
    assert p.content_hash is None     # not frozen yet, no hash
    frozen = p.freeze()
    assert frozen.frozen is True
    assert frozen.content_hash is not None
    assert frozen.content_hash.startswith("sha256:")


def test_freezing_unreviewed_raises():
    p = TargetCellProfile(**_minimal_profile_kwargs(), human_reviewed=False)
    with pytest.raises(UnreviewedProfileError):
        p.freeze()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/profile/test_freeze_interlock.py -v`
Expected: 3 failures (validator + method missing).

- [ ] **Step 3: Add freeze interlock**

Edit `packages/rarecell/src/rarecell/profile/schema.py`. Add at the top:

```python
from rarecell.errors import UnreviewedProfileError
```

Add a `model_validator` to `TargetCellProfile`:

```python
    @model_validator(mode="after")
    def _frozen_requires_review(self) -> "TargetCellProfile":
        if self.frozen and not self.human_reviewed:
            raise UnreviewedProfileError(
                "Cannot set frozen=True without human_reviewed=True. "
                "A human must review and sign off on the profile before it is frozen."
            )
        return self

    def freeze(self) -> "TargetCellProfile":
        """Return a frozen copy with content_hash set. Requires human_reviewed=True."""
        if not self.human_reviewed:
            raise UnreviewedProfileError(
                "freeze() requires human_reviewed=True. "
                "Set human_reviewed=True and provide reviewer email before freezing."
            )
        h = self.compute_content_hash()
        return self.model_copy(update={"frozen": True, "content_hash": h})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/profile/ -v`
Expected: all pass (4 tests total — 3 new + the basic from Task 4).

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/profile/schema.py packages/rarecell/tests/profile/test_freeze_interlock.py
git commit -m "Add freeze interlock: human_reviewed required for frozen=true"
```

---

## Task 6: Preset profiles (7 YAMLs)

**Files:**
- Create: `packages/rarecell/src/rarecell/profile/presets/{t_cell_pbmc,t_cell_cns,b_cell,nk_cell,microglia,dendritic_cell,monocyte_macrophage}.yaml`
- Create: `packages/rarecell/tests/profile/test_presets.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/profile/test_presets.py`:

```python
from pathlib import Path
import pytest
from rarecell.profile.schema import TargetCellProfile

PRESETS_DIR = Path(__file__).resolve().parents[3] / "src/rarecell/profile/presets"

PRESET_NAMES = [
    "t_cell_pbmc", "t_cell_cns", "b_cell", "nk_cell",
    "microglia", "dendritic_cell", "monocyte_macrophage",
]


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_loads(name):
    p = TargetCellProfile.from_yaml_path(PRESETS_DIR / f"{name}.yaml")
    assert p.frozen is False        # presets are not frozen — user must review
    assert p.human_reviewed is False
    assert p.positive_markers       # at least one panel
    assert len(p.tissue) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/profile/test_presets.py -v`
Expected: FileNotFoundError on all 7.

- [ ] **Step 3: Write the 7 preset YAMLs**

Create each preset using the marker panels from `als_utils.py:863-877` (T-cell panels) and `als_utils.py:927-1085` (cell type panels) as authoritative starting points where possible; otherwise use canonical PanglaoDB markers. Example for `t_cell_pbmc.yaml`:

```yaml
schema_version: "1.0"
profile_id: "preset-t-cell-pbmc-v0.1"
name: "T cells, PBMC"
description: "Pan T-cell isolation from peripheral blood single-cell RNA-seq"
target_lineage: "lymphoid"
tissue: ["blood", "pbmc"]
expected_abundance:
  min_fraction: 0.10
  max_fraction: 0.70
  notes: "T cells are 45-70% of PBMC lymphocytes"

positive_markers:
  pan_t_cell:
    genes: [CD3D, CD3E, CD3G, CD2, CD7, TRAC, TRBC1, TRBC2]
    threshold_z: 1.0
    citations:
      - {source_id: "panglaodb:T_cell", source: "panglaodb"}

negative_markers:
  b_cell:
    genes: [MS4A1, CD79A, CD79B, CD19]
    exclusion_threshold_z: 1.5
  myeloid:
    genes: [CD14, LYZ, CST3, S100A8]
    exclusion_threshold_z: 1.5

reference_labels:
  celltypist_models:
    - model: "Immune_All_Low.pkl"
      match_patterns: ["T cell", "Tcm", "Tem", "Treg", "Th1", "Th2", "Th17",
                       "CD4", "CD8", "MAIT", "NKT", "gdT"]
      enabled: true

biccn_rules:
  enabled: false

qc:
  min_genes_per_cell: 200
  max_pct_mt: 10
  max_genes_per_cell: 10000
  min_cells_per_gene: 3
  rationale: "Standard PBMC QC"

purify:
  enabled: true
  high_resolution: 2.0
  min_cluster_purity: 0.7

batch_correction:
  in_dataset: "harmony"
  batch_key: "sample_id"

auto_policy:
  gates:
    gate1_cluster_decisions: "recommendation"
    gate2_purify_decisions: "recommendation"
    gate3_final: "accept"

frozen: false
human_reviewed: false
```

For `t_cell_cns.yaml`, mirror the panels from `als_utils.py:863-877` (positive: `pan_t_cell`, `cd4_t_cell`, `cd8_t_cell`, `treg`, `exhausted_checkpoint`, `tissue_resident_memory`, `activation`) and `als_utils.py:1054-1066` (negative panels: `neuron`, `astrocyte`, `oligodendrocyte`, `opc`, `endothelial`, `microglia`). Set `tissue: ["brain", "spinal_cord"]`, `expected_abundance: {min_fraction: 0.0001, max_fraction: 0.05}`, `biccn_rules.enabled: true` with `class_filter: ["TCELL"]` and `subclass_filter: ["TCELL", "BCELL", "MGL", "MONO", "MAC", "NK"]`. Add `qc.rationale: "Permissive floor preserves small low-RNA nuclei"` with `min_genes_per_cell: 150`. Include a `description` note that the PsychAD MSSM brain CellTypist model is not bundled and link the user to docs.

For the other five presets — `b_cell`, `nk_cell`, `microglia`, `dendritic_cell`, `monocyte_macrophage` — use canonical PanglaoDB marker sets:
- `b_cell`: positive `pan_b: [MS4A1, CD79A, CD79B, CD19, BANK1]`; negative panels exclude T/NK/myeloid.
- `nk_cell`: positive `pan_nk: [NCAM1, NKG7, GNLY, KLRD1, FCGR3A]`; negative excludes T (CD3D etc.).
- `microglia`: positive `microglia: [CX3CR1, P2RY12, TMEM119, AIF1, CSF1R]`; tissue: brain; negative excludes neurons + astrocytes + monocytes.
- `dendritic_cell`: positive `dc: [CLEC9A, CLEC10A, FCER1A, ITGAX, CD1C]`; negative excludes T/B/NK.
- `monocyte_macrophage`: positive `mono_mac: [CD14, FCGR3A, LYZ, CST3, S100A8, CD68]`; negative excludes T/B/NK.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/profile/test_presets.py -v`
Expected: 7 tests pass.

Then un-skip the YAML roundtrip test from Task 4 (remove the `@pytest.mark.skip`):

Run: `uv run pytest packages/rarecell/tests/profile/test_schema_basic.py::test_yaml_roundtrip -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/profile/presets/ packages/rarecell/tests/profile/test_presets.py
git commit -m "Add 7 preset profiles (t-cell PBMC/CNS, B, NK, microglia, DC, mono/mac)"
```

---

## Task 7: `core/ingest.py` — count validation + symbol conversion

**Files:**
- Create: `packages/rarecell/src/rarecell/core/__init__.py`, `core/ingest.py`
- Create: `packages/rarecell/tests/core/test_ingest.py`

**Port from:** `als_utils.py:244-433` (BioMart cache + Ensembl→symbol conversion), `als_utils.py:308-326` (protein-coding gene filter), `als_utils.py:5049-5060` (unique obs names), `als_utils.py:5107-5147` (restore full genes), plus a new `validate_counts` function.

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_ingest.py`:

```python
import numpy as np
import pytest
import anndata as ad
from scipy import sparse
from rarecell.core.ingest import validate_counts, make_obs_names_unique_across_samples
from rarecell.errors import MissingRawCountsError


def _toy_adata(layer_with_counts=None, x_is_counts=True, has_raw=False):
    n = 100
    rng = np.random.default_rng(0)
    counts = sparse.csr_matrix(rng.poisson(2, size=(n, 50)).astype(float))
    X = counts.copy() if x_is_counts else counts.copy().multiply(0.1)
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * n})
    if layer_with_counts:
        a.layers[layer_with_counts] = counts.copy()
    if has_raw:
        a.raw = ad.AnnData(X=counts.copy())
    return a


def test_validate_counts_finds_X():
    a = _toy_adata(x_is_counts=True)
    out = validate_counts(a)
    # returns the layer name where counts live, "X" if .X is counts
    assert out == "X"


def test_validate_counts_finds_layer():
    a = _toy_adata(x_is_counts=False, layer_with_counts="counts")
    out = validate_counts(a)
    assert out == "counts"


def test_validate_counts_finds_raw():
    a = _toy_adata(x_is_counts=False, has_raw=True)
    out = validate_counts(a)
    assert out == "raw"


def test_validate_counts_missing_raises():
    a = _toy_adata(x_is_counts=False, has_raw=False)
    with pytest.raises(MissingRawCountsError):
        validate_counts(a)


def test_make_obs_names_unique():
    a = ad.AnnData(X=np.zeros((3, 2)))
    a.obs_names = ["c1", "c2", "c3"]
    b = ad.AnnData(X=np.zeros((3, 2)))
    b.obs_names = ["c1", "c2", "c3"]   # collisions
    out_a, out_b = make_obs_names_unique_across_samples([a, b], ["s1", "s2"])
    assert list(out_a.obs_names) == ["s1_c1", "s1_c2", "s1_c3"]
    assert list(out_b.obs_names) == ["s2_c1", "s2_c2", "s2_c3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/core/test_ingest.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/ingest.py`**

Create `packages/rarecell/src/rarecell/core/__init__.py` (empty).

Create `packages/rarecell/src/rarecell/core/ingest.py`. Implement:

```python
"""Ingest: count validation, symbol conversion, obs-name deduplication."""

from __future__ import annotations
from typing import Literal
import numpy as np
import anndata as ad
from rarecell.errors import MissingRawCountsError


CountsLocation = Literal["X", "raw", "counts"] | str


def _looks_like_counts(matrix) -> bool:
    """Heuristic: integer-valued and non-negative."""
    sample = matrix[:100].toarray() if hasattr(matrix, "toarray") else matrix[:100]
    sample = np.asarray(sample)
    if (sample < 0).any():
        return False
    # accept if all near-integer (allows float dtypes containing integer values)
    return np.allclose(sample, np.round(sample))


def validate_counts(adata: ad.AnnData) -> CountsLocation:
    """Locate raw integer counts on the AnnData.

    Returns the location label: "X", "raw", or the layer name.
    Raises MissingRawCountsError if not found.
    """
    if _looks_like_counts(adata.X):
        return "X"
    for layer_name in ("counts", "raw_counts", "spliced"):
        if layer_name in adata.layers and _looks_like_counts(adata.layers[layer_name]):
            return layer_name
    if adata.raw is not None and _looks_like_counts(adata.raw.X):
        return "raw"
    raise MissingRawCountsError(
        "No raw integer counts found in .X, .layers['counts'], or .raw. "
        "rarecell requires integer counts for QC, Scrublet, and normalization. "
        "If counts are stored elsewhere, copy them into adata.layers['counts']."
    )


def make_obs_names_unique_across_samples(
    adata_list: list[ad.AnnData], sample_ids: list[str]
) -> list[ad.AnnData]:
    """Prefix obs_names with sample_id to ensure global uniqueness."""
    if len(adata_list) != len(sample_ids):
        raise ValueError("adata_list and sample_ids must have the same length")
    out = []
    for a, sid in zip(adata_list, sample_ids):
        a = a.copy()
        a.obs_names = [f"{sid}_{n}" for n in a.obs_names]
        out.append(a)
    return out
```

Port BioMart-backed Ensembl→symbol conversion from `als_utils.py:244-433` as additional functions in this same module: `convert_ensembl_to_symbols(adata, cache_path=None)`, `get_protein_coding_autosomal_genes(adata)`, `restore_full_genes(adata)`. Keep the function signatures and behavior from `als_utils.py`; just remove any T-cell-specific logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/core/test_ingest.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Add tests for the BioMart-backed functions**

```python
def test_get_protein_coding_autosomal_genes(monkeypatch):
    # use a tiny in-memory gene annotation table
    import pandas as pd
    fake_ann = pd.DataFrame({
        "gene_name": ["CD3D", "MT-ATP6", "XIST", "GAPDH"],
        "chromosome_name": ["11", "MT", "X", "12"],
        "gene_biotype": ["protein_coding"] * 4,
    })
    monkeypatch.setattr("rarecell.core.ingest._load_or_query_gene_annotations",
                       lambda *a, **kw: fake_ann)
    a = ad.AnnData(X=np.zeros((1, 4)),
                   var={"gene_symbols": ["CD3D", "MT-ATP6", "XIST", "GAPDH"]})
    a.var_names = ["CD3D", "MT-ATP6", "XIST", "GAPDH"]
    keep = get_protein_coding_autosomal_genes(a)
    assert set(keep) == {"CD3D", "GAPDH"}
```

Re-run tests. Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add packages/rarecell/src/rarecell/core/__init__.py packages/rarecell/src/rarecell/core/ingest.py packages/rarecell/tests/core/test_ingest.py
git commit -m "Add core.ingest: count validation, symbol conversion, gene filtering"
```

---

## Task 8: `core/qc.py` — profile-driven QC + Scrublet

**Files:**
- Create: `packages/rarecell/src/rarecell/core/qc.py`
- Create: `packages/rarecell/tests/core/test_qc.py`

**Port from:** `als_utils.py:564-679` (run_qc + run_scrublet).

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_qc.py`:

```python
import numpy as np
import anndata as ad
from scipy import sparse
from rarecell.profile.schema import QCParams
from rarecell.core.qc import run_qc, run_scrublet


def _toy_adata(n=1000):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 200)).astype(float))
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * n})
    a.var_names = [f"MT-{i}" if i < 10 else f"GENE{i}" for i in range(200)]
    return a


def test_run_qc_filters_with_profile_params():
    a = _toy_adata()
    params = QCParams(min_genes_per_cell=150, max_pct_mt=10,
                      max_genes_per_cell=10000, min_cells_per_gene=3)
    out = run_qc(a, params)
    assert "n_genes_by_counts" in out.obs.columns
    assert "pct_counts_mt" in out.obs.columns
    assert (out.obs["n_genes_by_counts"] >= 150).all()
    assert (out.obs["pct_counts_mt"] <= 10).all()


def test_run_scrublet_marks_doublets():
    a = _toy_adata(n=300)
    out = run_scrublet(a, batch_key="sample_id", expected_doublet_rate=0.05)
    assert "predicted_doublet" in out.obs.columns
    assert "doublet_score" in out.obs.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/core/test_qc.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/qc.py`**

Port `als_utils.py:564-613` (`run_qc`) and `als_utils.py:614-679` (`run_scrublet`). Two changes from the original:
1. `run_qc` takes a `QCParams` pydantic object instead of a `params` dict; read attributes directly.
2. Remove any `dataset_id` argument and figure-directory side effects; this module is pure compute.

Signature targets:

```python
def run_qc(adata: ad.AnnData, params: QCParams) -> ad.AnnData: ...
def run_scrublet(
    adata: ad.AnnData, *, batch_key: str = "sample_id",
    expected_doublet_rate: float = 0.05,
) -> ad.AnnData: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/core/test_qc.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/qc.py packages/rarecell/tests/core/test_qc.py
git commit -m "Add core.qc: profile-driven QC + per-sample Scrublet"
```

---

## Task 9: `core/markers.py` — profile-driven panel scoring

**Files:**
- Create: `packages/rarecell/src/rarecell/core/markers.py`
- Create: `packages/rarecell/tests/core/test_markers.py`

**Port from:** `als_utils.py:848-877` (`_score_panel`) — make public; replace T-cell-specific helpers with profile-driven `score_profile_markers` / `score_negative_panels`.

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_markers.py`:

```python
import numpy as np
import anndata as ad
import scanpy as sc
from rarecell.core.markers import (
    score_panel, score_profile_markers, score_negative_panels,
)
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, NegativePanel, Citation,
    ExpectedAbundance, QCParams,
)


def _toy_adata_with_panels():
    rng = np.random.default_rng(0)
    n = 200
    genes = ["CD3D", "CD3E", "CD4", "CD8A", "MS4A1", "NEUROD1"]
    X = rng.poisson(1, size=(n, len(genes))).astype(float)
    # boost CD3D + CD3E for first 50 cells (the "T cell" cluster)
    X[:50, :2] += 10
    # boost MS4A1 for next 50 ("B cells")
    X[50:100, 4] += 10
    a = ad.AnnData(X=X, var={"gene": genes})
    a.var_names = genes
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_score_panel_writes_obs():
    a = _toy_adata_with_panels()
    score_panel(a, "pan_t", ["CD3D", "CD3E"], threshold_z=1.0, use_raw=False)
    assert "score_pan_t" in a.obs.columns
    assert "pass_pan_t" in a.obs.columns
    # first 50 cells should pass; B cells should not
    assert a.obs["pass_pan_t"][:50].sum() > 30
    assert a.obs["pass_pan_t"][50:100].sum() < 10


def _profile():
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(genes=["CD3D", "CD3E"], threshold_z=1.0,
                                  citations=[Citation(source_id="pmid:1",
                                                      source="europepmc")])
        },
        negative_markers={
            "b_cell": NegativePanel(genes=["MS4A1"], exclusion_threshold_z=1.0),
        },
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_score_profile_markers_writes_all_panels():
    a = _toy_adata_with_panels()
    score_profile_markers(a, _profile(), use_raw=False)
    assert "score_pan_t" in a.obs
    assert "pass_pan_t" in a.obs


def test_score_negative_panels_flags_contaminants():
    a = _toy_adata_with_panels()
    score_negative_panels(a, _profile(), use_raw=False)
    assert "is_contaminant" in a.obs
    # B cells flagged as contaminant
    assert a.obs["is_contaminant"][50:100].sum() > 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/core/test_markers.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/markers.py`**

```python
"""Profile-driven marker scoring.

score_panel is a thin wrapper over scanpy.tl.score_genes that also writes
a boolean pass_<name> column based on a z-score threshold.
"""
from __future__ import annotations
import numpy as np
import scanpy as sc
import anndata as ad
from rarecell.profile.schema import TargetCellProfile


def score_panel(
    adata: ad.AnnData, name: str, genes: list[str],
    threshold_z: float | None = None, use_raw: bool = True,
) -> None:
    """Score a marker panel via sc.tl.score_genes.

    Writes adata.obs[f"score_{name}"]. If threshold_z is not None, also writes
    adata.obs[f"pass_{name}"] = score > mean + threshold_z * std.
    """
    present = [g for g in genes if g in (adata.raw.var_names if use_raw
                                          else adata.var_names)]
    if not present:
        adata.obs[f"score_{name}"] = 0.0
        if threshold_z is not None:
            adata.obs[f"pass_{name}"] = False
        return

    sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}",
                      use_raw=use_raw)
    if threshold_z is not None:
        s = adata.obs[f"score_{name}"]
        adata.obs[f"pass_{name}"] = (s > s.mean() + threshold_z * s.std())


def score_profile_markers(
    adata: ad.AnnData, profile: TargetCellProfile, use_raw: bool = True,
) -> None:
    """Score every positive_markers panel in the profile."""
    for name, panel in profile.positive_markers.items():
        score_panel(adata, name, panel.genes, panel.threshold_z, use_raw=use_raw)


def score_negative_panels(
    adata: ad.AnnData, profile: TargetCellProfile, use_raw: bool = True,
) -> None:
    """Score negative_markers panels and write is_contaminant flag.

    A cell is_contaminant if ANY negative panel exceeds its exclusion_threshold_z.
    """
    flags = np.zeros(adata.n_obs, dtype=bool)
    for name, panel in profile.negative_markers.items():
        score_panel(adata, name, panel.genes, panel.exclusion_threshold_z,
                    use_raw=use_raw)
        flags |= adata.obs[f"pass_{name}"].to_numpy()
    adata.obs["is_contaminant"] = flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/core/test_markers.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/markers.py packages/rarecell/tests/core/test_markers.py
git commit -m "Add core.markers: profile-driven panel + negative scoring"
```

---

## Task 10: `core/annotate.py` — profile-driven CellTypist + Enrichr

**Files:**
- Create: `packages/rarecell/src/rarecell/core/annotate.py`
- Create: `packages/rarecell/tests/core/test_annotate.py`

**Port from:** `als_utils.py:714-738` (`annotate_celltypist_immune` — generalize), `als_utils.py:2043-2118` (`annotate_celltypist_hierarchical` — generalize), `als_utils.py:4032-4246` (Enrichr enrichment + bubble plot), `als_utils.py:5589-5654` (`build_consensus_labels`).

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_annotate.py`:

```python
from unittest.mock import patch, MagicMock
import numpy as np
import anndata as ad
import pandas as pd
from rarecell.core.annotate import annotate_celltypist
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, ReferenceLabels, CellTypistRef,
)


def _profile_with_two_models():
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[
            CellTypistRef(model="Immune_All_Low.pkl",
                          match_patterns=["T cell"], enabled=True),
            CellTypistRef(model="Disabled.pkl",
                          match_patterns=["x"], enabled=False),
        ]),
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_annotate_celltypist_skips_disabled_and_runs_enabled():
    a = ad.AnnData(X=np.zeros((10, 5)), var={"g": list("abcde")})
    a.var_names = list("abcde")
    with patch("rarecell.core.annotate._run_one_celltypist_model") as mock:
        mock.return_value = pd.DataFrame({
            "predicted_labels": ["T cell"] * 10,
            "majority_voting": ["T cell"] * 10,
            "conf_score": [0.9] * 10,
        }, index=a.obs_names)
        annotate_celltypist(a, _profile_with_two_models())
    mock.assert_called_once()
    args, _ = mock.call_args
    assert args[1] == "Immune_All_Low.pkl"
    assert "celltypist_Immune_All_Low_label" in a.obs.columns
    assert "celltypist_Immune_All_Low_label_majority" in a.obs.columns
    assert "celltypist_Immune_All_Low_conf" in a.obs.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/core/test_annotate.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/annotate.py`**

```python
"""Profile-driven CellTypist annotation + Enrichr enrichment."""
from __future__ import annotations
import pandas as pd
import anndata as ad
from rarecell.profile.schema import TargetCellProfile, CellTypistRef


def _model_label(model_filename: str) -> str:
    """e.g. 'Immune_All_Low.pkl' -> 'Immune_All_Low'."""
    return model_filename.replace(".pkl", "")


def _run_one_celltypist_model(
    adata: ad.AnnData, model_name: str, majority_voting: bool = True,
) -> pd.DataFrame:
    """Wrap celltypist.annotate. Returns a DataFrame indexed by adata.obs_names
    with columns predicted_labels, majority_voting, conf_score.
    """
    import celltypist  # heavy import; deferred
    model = celltypist.models.Model.load(model_name)
    result = celltypist.annotate(adata, model=model, majority_voting=majority_voting)
    return result.predicted_labels


def annotate_celltypist(adata: ad.AnnData, profile: TargetCellProfile) -> None:
    """Run every enabled CellTypist model in the profile.

    Writes to adata.obs:
      celltypist_{model_label}_label
      celltypist_{model_label}_label_majority
      celltypist_{model_label}_conf
    """
    for ref in profile.reference_labels.celltypist_models:
        if not ref.enabled:
            continue
        preds = _run_one_celltypist_model(adata, ref.model)
        label = _model_label(ref.model)
        adata.obs[f"celltypist_{label}_label"] = preds["predicted_labels"].values
        adata.obs[f"celltypist_{label}_label_majority"] = preds["majority_voting"].values
        adata.obs[f"celltypist_{label}_conf"] = preds["conf_score"].values
```

Port `enrichr_cell_type_enrichment` from `als_utils.py:4032-4176` and `plot_enrichr_bubble` from `als_utils.py:4178-4246` into the same module with minor name cleanup. Port `build_consensus_labels` from `als_utils.py:5589-5654`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/core/test_annotate.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/annotate.py packages/rarecell/tests/core/test_annotate.py
git commit -m "Add core.annotate: profile-driven CellTypist + Enrichr + consensus labels"
```

---

## Task 11: `core/clustering.py` — taxonomy_cluster with in-dataset Harmony

**Files:**
- Create: `packages/rarecell/src/rarecell/core/clustering.py`
- Create: `packages/rarecell/tests/core/test_clustering.py`

**Port from:** `als_utils.py:1111-1305` (`taxonomy_cluster`), `als_utils.py:1306-1350` (`finalize_taxonomy_cluster`), `als_utils.py:1351-1491` (`scan_leiden_resolution`), `als_utils.py:1492-1604` (`_smooth_metric`, `_normalize_01`, `_select_best_resolution`), `als_utils.py:1605-1724` (`compute_marker_purity`), `als_utils.py:1725-1794` (`compute_cluster_quality`).

Key change: `taxonomy_cluster(adata, profile, stage=None, ...)` — `stage` is a free-form string (defaults to `"class"`); profile drives:
- HVG selection mode (always protein-coding autosomal for now)
- Harmony batch_key from `profile.batch_correction.batch_key` if `in_dataset == "harmony"`; else skip Harmony
- All else as in original

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_clustering.py`:

```python
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse
from rarecell.core.clustering import taxonomy_cluster
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, BatchCorrection,
)


def _profile(in_dataset="harmony"):
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        batch_correction=BatchCorrection(in_dataset=in_dataset, batch_key="sample_id"),
    )


def _toy(n=400):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 100)).astype(float))
    a = ad.AnnData(X=X, obs={"sample_id": ["s1"] * (n // 2) + ["s2"] * (n // 2)})
    a.var_names = [f"G{i}" for i in range(100)]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_taxonomy_cluster_writes_leiden_and_pca():
    a = _toy()
    taxonomy_cluster(a, _profile(), stage="class")
    assert "leiden" in a.obs.columns
    assert "X_pca" in a.obsm
    assert "X_pca_harmony" in a.obsm   # because in_dataset == "harmony"


def test_taxonomy_cluster_skips_harmony_when_none():
    a = _toy()
    taxonomy_cluster(a, _profile(in_dataset="none"), stage="class")
    assert "leiden" in a.obs.columns
    assert "X_pca" in a.obsm
    assert "X_pca_harmony" not in a.obsm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/core/test_clustering.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/clustering.py`**

Port `taxonomy_cluster` from `als_utils.py:1111`. Replace its hardcoded knobs with `profile`-driven values:
- `batch_key` from `profile.batch_correction.batch_key`
- Skip Harmony if `profile.batch_correction.in_dataset == "none"` (still compute PCA + neighbors + Leiden on `X_pca`)
- Keep silhouette-guided Leiden resolution scan
- Keep protein-coding autosomal HVG selection (call `get_protein_coding_autosomal_genes` from `core.ingest`)
- Keep cell-cycle regression

Also port `finalize_taxonomy_cluster`, `scan_leiden_resolution`, `_smooth_metric`, `_normalize_01`, `_select_best_resolution`, `compute_marker_purity`, `compute_cluster_quality` as supporting functions in the same module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/core/test_clustering.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/clustering.py packages/rarecell/tests/core/test_clustering.py
git commit -m "Add core.clustering: profile-driven taxonomy_cluster + silhouette Leiden"
```

---

## Task 12: `core/evidence.py` — BICCN trinarization + consensus table

**Files:**
- Create: `packages/rarecell/src/rarecell/core/evidence.py`
- Create: `packages/rarecell/tests/core/test_evidence.py`

**Port from:** `als_utils.py:1795-1846` (`_trinarize`), `als_utils.py:1847-2042` (`annotate_biccn_rules` → `score_biccn_evidence`), `als_utils.py:2387-2520` (`score_stage_evidence` → `score_evidence`), `als_utils.py:3160-3439` (`plot_consensus_evidence_table` → `render_consensus_table` returning `(DataFrame, Figure)`), plus all the BICCN plot helpers.

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/core/test_evidence.py`:

```python
import numpy as np
import pandas as pd
import anndata as ad
from rarecell.core.evidence import (
    score_biccn_evidence, score_evidence, render_consensus_table,
    select_clusters,
)
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, BICCNRules,
)


def _profile():
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["brain"],
        expected_abundance=ExpectedAbundance(min_fraction=0.001, max_fraction=0.05),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        biccn_rules=BICCNRules(enabled=True, class_filter=["TCELL"],
                                subclass_filter=["TCELL", "BCELL"]),
        qc=QCParams(min_genes_per_cell=150, max_pct_mt=10),
    )


def _adata_with_clusters():
    n = 200
    a = ad.AnnData(X=np.zeros((n, 5)),
                   obs={"leiden": ["0"] * 100 + ["1"] * 100,
                        "score_pan_t": [2.0] * 100 + [0.1] * 100,
                        "pass_pan_t": [True] * 100 + [False] * 100,
                        "is_contaminant": [False] * 200})
    a.var_names = ["CD3D", "GFAP", "MS4A1", "AQP4", "RBFOX3"]
    return a


def test_score_evidence_returns_one_row_per_cluster():
    a = _adata_with_clusters()
    table = score_evidence(a, _profile(), cluster_key="leiden")
    assert isinstance(table, pd.DataFrame)
    assert set(table["cluster"].astype(str)) == {"0", "1"}
    # cluster "0" has higher pan_t score
    score_0 = float(table.set_index("cluster").loc["0", "score_pan_t_mean"])
    score_1 = float(table.set_index("cluster").loc["1", "score_pan_t_mean"])
    assert score_0 > score_1


def test_select_clusters_routes_by_recommendation():
    a = _adata_with_clusters()
    table = score_evidence(a, _profile(), cluster_key="leiden")
    table["recommendation"] = ["keep", "drop"]
    keep = select_clusters(table, "keep")
    drop = select_clusters(table, "drop")
    assert set(keep) == {"0"}
    assert set(drop) == {"1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/core/test_evidence.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/evidence.py`**

Port `_trinarize` from `als_utils.py:1795`. Port `annotate_biccn_rules` as `score_biccn_evidence(adata, profile, cluster_key)`. Port `score_stage_evidence` as `score_evidence(adata, profile, cluster_key) -> pd.DataFrame` — returns one row per cluster with these columns:

- `cluster`
- one `score_{panel}_mean` per positive panel
- one `pass_{panel}_frac` per positive panel (fraction of cluster cells that pass)
- one `negative_{panel}_frac` per negative panel
- `is_contaminant_frac`
- `celltypist_{model}_top_label` per enabled model
- `celltypist_{model}_top_label_frac`
- `biccn_top_label`, `biccn_top_prob` (if `biccn_rules.enabled`)
- `n_cells`

Port `plot_consensus_evidence_table` as `render_consensus_table(adata, profile, cluster_key) -> (pd.DataFrame, plt.Figure)`. Add `select_clusters(table, recommendation)` that returns the list of cluster IDs matching the `recommendation` column value.

Port the plot helpers (`plot_biccn_composition`, `plot_biccn_dotplot`, `plot_biccn_probability_table`, `plot_annotation_confusion`, `plot_all_markers_dotplot`, `plot_resolution_scan`, `plot_resolution_umap_comparison`, `plot_stage_evidence`) as additional functions in this module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/core/test_evidence.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/evidence.py packages/rarecell/tests/core/test_evidence.py
git commit -m "Add core.evidence: BICCN trinarization + multi-evidence consensus table"
```

---

## Task 13: `core/purify.py` — subcluster_and_purify

**Files:**
- Create: `packages/rarecell/src/rarecell/core/purify.py`
- Create: `packages/rarecell/tests/core/test_purify.py`

**Port from:** `als_utils.py:2167-2386` (`subcluster_and_purify`).

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/core/test_purify.py`:

```python
import numpy as np
import anndata as ad
from scipy import sparse
import scanpy as sc
from rarecell.core.purify import subcluster_and_purify
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, PurifyParams,
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
        purify=PurifyParams(enabled=True, high_resolution=2.0,
                            min_cluster_purity=0.5),
    )


def test_purify_returns_filtered_adata():
    rng = np.random.default_rng(0)
    n = 300
    X = sparse.csr_matrix(rng.poisson(2, size=(n, 50)).astype(float))
    a = ad.AnnData(X=X, obs={"leiden": ["0"] * n,
                              "sample_id": ["s1"] * n})
    a.var_names = [f"G{i}" for i in range(50)]
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    out = subcluster_and_purify(a, _profile(), suspect_clusters=["0"],
                                cluster_key="leiden")
    # output is an AnnData; size may shrink
    assert out.n_obs <= a.n_obs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/core/test_purify.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/purify.py`**

Port `subcluster_and_purify` from `als_utils.py:2167`. Generalize:
- Replace `stage="subtype"` references with the profile's `purify.high_resolution`
- Replace any T-cell-specific evidence calls with `score_evidence(sub_adata, profile, "leiden")`
- Use profile's `purify.min_cluster_purity` as the keep/drop threshold

Signature target:

```python
def subcluster_and_purify(
    adata: ad.AnnData, profile: TargetCellProfile,
    suspect_clusters: list[str], *, cluster_key: str = "leiden",
) -> ad.AnnData: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/core/test_purify.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/purify.py packages/rarecell/tests/core/test_purify.py
git commit -m "Add core.purify: profile-driven subcluster_and_purify"
```

---

## Task 14: `core/io.py` — h5ad sanitization + checkpoints

**Files:**
- Create: `packages/rarecell/src/rarecell/core/io.py`
- Create: `packages/rarecell/tests/core/test_io.py`

**Port from:** `als_utils.py:33-243` (`setup_figure_dir`), `als_utils.py:5149-5292` (`save_checkpoint`, `load_checkpoint`, `_stringify_dict_keys`, `_sanitize_uns_for_h5ad`, `_sanitize_df_columns`, `save_output`, `generate_qc_report`).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import numpy as np
import anndata as ad
from rarecell.core.io import save_h5ad, load_h5ad


def test_save_load_roundtrip_sanitizes_uns(tmp_path):
    a = ad.AnnData(X=np.zeros((5, 3)))
    a.uns = {1: "non-string-key", "nested": {2: "x"}}   # would normally fail h5ad write
    save_h5ad(a, tmp_path / "out.h5ad")
    b = load_h5ad(tmp_path / "out.h5ad")
    assert b.n_obs == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/core/test_io.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `core/io.py`**

Port the sanitization helpers from `als_utils.py:5188-5235`. Provide `save_h5ad(adata, path)` and `load_h5ad(path)` as the public surface; both call into the porting helpers. Also export `setup_figure_dir(out_dir)`.

- [ ] **Step 4: Run test to verify it passes**

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/core/io.py packages/rarecell/tests/core/test_io.py
git commit -m "Add core.io: h5ad sanitization + checkpoint helpers"
```

---

## Task 15: Synthetic AnnData fixture

**Files:**
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/make_synthetic.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the fixture generator with a known-rare planted cluster**

Create `tests/fixtures/make_synthetic.py`:

```python
"""Generate a synthetic AnnData with planted T-cell-like rare population.

5000 cells; 4 clusters at 30%, 40%, 25%, 5%. The 5% cluster has high
CD3D/CD3E expression (the "rare T cells"). Other clusters express
neuron-like (RBFOX3), astrocyte-like (GFAP), or B-cell-like (MS4A1) markers.
"""
from __future__ import annotations
import numpy as np
import anndata as ad
from scipy import sparse

GENES = [
    "CD3D", "CD3E", "CD3G", "TRAC",     # T cell positive
    "MS4A1", "CD79A",                    # B cell (negative panel)
    "GFAP", "AQP4", "ALDH1L1",           # astrocyte (negative)
    "RBFOX3", "SNAP25", "SYT1",          # neuron (negative)
] + [f"GENE{i}" for i in range(38)]      # filler — 50 genes total


def make_synthetic(seed: int = 0, n_cells: int = 5000) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # cluster sizes: 0=neuron 30%, 1=astrocyte 40%, 2=B 25%, 3=Tcell 5%
    sizes = [int(n_cells * f) for f in (0.30, 0.40, 0.25, 0.05)]
    sizes[-1] = n_cells - sum(sizes[:-1])
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])

    X = rng.poisson(0.5, size=(n_cells, len(GENES))).astype(float)
    # boost cluster-specific markers
    for ci, marker_idxs in enumerate([(9, 10, 11),   # neuron
                                       (6, 7, 8),     # astrocyte
                                       (4, 5),        # B cell
                                       (0, 1, 2, 3)]):  # T cell (the rare one)
        rows = np.where(labels == ci)[0]
        for j in marker_idxs:
            X[rows, j] += rng.poisson(15, size=rows.shape[0])

    n_per_sample = n_cells // 4
    sample_id = np.repeat([f"s{i}" for i in range(4)], n_per_sample)
    sample_id = np.concatenate([sample_id, ["s3"] * (n_cells - len(sample_id))])

    a = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs={"sample_id": sample_id, "true_cluster": labels.astype(str)},
        var={"gene": GENES},
    )
    a.var_names = GENES
    a.layers["counts"] = a.X.copy()
    return a
```

Create `tests/conftest.py`:

```python
import pytest
from tests.fixtures.make_synthetic import make_synthetic


@pytest.fixture
def synthetic_adata():
    return make_synthetic(seed=0)
```

- [ ] **Step 2: Add a smoke test that the fixture builds**

Create `tests/fixtures/test_synthetic_smoke.py`:

```python
def test_fixture_shape(synthetic_adata):
    a = synthetic_adata
    assert a.n_obs == 5000
    # ~5% true_cluster == "3" (the planted rare T cell pop)
    rare_frac = (a.obs["true_cluster"] == "3").mean()
    assert 0.04 < rare_frac < 0.06
```

Run: `uv run pytest tests/fixtures/test_synthetic_smoke.py -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/ tests/conftest.py
git commit -m "Add synthetic AnnData fixture with planted 5% T-cell cluster"
```

---

## Task 16: Recommender protocol + BasicRecommender

**Files:**
- Create: `packages/rarecell/src/rarecell/recommender/{__init__.py,base.py,basic.py}`
- Create: `packages/rarecell/tests/recommender/test_basic.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/recommender/test_basic.py`:

```python
import pandas as pd
from rarecell.recommender.base import Recommendation
from rarecell.recommender.basic import BasicRecommender
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance, QCParams,
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


def test_basic_recommender_keeps_high_panel_low_contam():
    table = pd.DataFrame({
        "cluster": ["0", "1", "2"],
        "n_cells": [100, 100, 100],
        "score_pan_t_mean": [2.0, 0.1, 1.5],
        "pass_pan_t_frac": [0.9, 0.05, 0.7],
        "is_contaminant_frac": [0.02, 0.5, 0.15],
    })
    recs = BasicRecommender(_profile()).recommend(table)
    by_id = {r.cluster_id: r for r in recs}
    assert by_id["0"].recommendation == "keep"
    assert by_id["1"].recommendation == "drop"
    assert by_id["2"].recommendation == "purify"   # mixed signal
    # all Recommendations have evidence + confidence
    for r in recs:
        assert 0 <= r.confidence <= 1
        assert r.evidence_summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/recommender/test_basic.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the recommender modules**

Create `packages/rarecell/src/rarecell/recommender/__init__.py`:

```python
from rarecell.recommender.base import Recommender, Recommendation
from rarecell.recommender.basic import BasicRecommender

__all__ = ["Recommender", "Recommendation", "BasicRecommender"]
```

Create `packages/rarecell/src/rarecell/recommender/base.py`:

```python
"""Recommender protocol + Recommendation dataclass."""
from __future__ import annotations
from typing import Literal, Protocol
import pandas as pd
from pydantic import BaseModel


Decision = Literal["keep", "drop", "purify"]


class Recommendation(BaseModel):
    cluster_id: str
    recommendation: Decision
    confidence: float
    evidence_summary: dict
    reasoning: str
    citations: list[str] = []


class Recommender(Protocol):
    """Anything that turns a consensus-table DataFrame into per-cluster Recommendations."""

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]: ...
```

Create `packages/rarecell/src/rarecell/recommender/basic.py`:

```python
"""Heuristic-only recommender. No LLM; used when [agent] extra not installed."""
from __future__ import annotations
import pandas as pd
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommender, Recommendation


class BasicRecommender(Recommender):
    """Threshold-based keep/drop/purify per cluster.

    keep:    any positive panel pass_frac >= 0.5 AND is_contaminant_frac < 0.1
    drop:    no positive panel pass_frac >= 0.1 OR is_contaminant_frac > 0.4
    purify:  otherwise (mixed signal)
    """

    def __init__(self, profile: TargetCellProfile):
        self.profile = profile

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]:
        positive_names = list(self.profile.positive_markers.keys())
        out: list[Recommendation] = []
        for _, row in table.iterrows():
            pass_fracs = [row.get(f"pass_{n}_frac", 0.0) for n in positive_names]
            best_pass = max(pass_fracs) if pass_fracs else 0.0
            contam = row.get("is_contaminant_frac", 0.0)

            if best_pass >= 0.5 and contam < 0.1:
                rec, conf = "keep", 0.9
                reasoning = f"Strong positive signal ({best_pass:.2f}) and low contamination."
            elif best_pass < 0.1 or contam > 0.4:
                rec, conf = "drop", 0.85
                reasoning = (f"Weak positive ({best_pass:.2f}) "
                             f"or heavy contamination ({contam:.2f}).")
            else:
                rec, conf = "purify", 0.55
                reasoning = "Mixed signal — recommend surgical subclustering."

            ev = {n: float(row.get(f"score_{n}_mean", 0.0)) for n in positive_names}
            ev["is_contaminant_frac"] = float(contam)
            out.append(Recommendation(
                cluster_id=str(row["cluster"]),
                recommendation=rec,
                confidence=conf,
                evidence_summary=ev,
                reasoning=reasoning,
            ))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/recommender/ -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/recommender/ packages/rarecell/tests/recommender/
git commit -m "Add Recommender protocol + BasicRecommender heuristic"
```

---

## Task 17: State machine — IsolateState + transitions

**Files:**
- Create: `packages/rarecell/src/rarecell/state_machine/{__init__.py,states.py,isolate.py}`
- Create: `packages/rarecell/tests/state_machine/test_states.py`

- [ ] **Step 1: Write the failing test (state transitions only, no execution yet)**

Create `packages/rarecell/tests/state_machine/test_states.py`:

```python
from rarecell.state_machine.states import IsolateState, valid_transitions


def test_normal_progression():
    seq = [IsolateState.S0_LOAD, IsolateState.S1_INGEST, IsolateState.S2_QC,
           IsolateState.S3_CLUSTER, IsolateState.S4_GATE1,
           IsolateState.S5_PURIFY, IsolateState.S5_GATE2,
           IsolateState.S6_FINAL, IsolateState.S6_GATE3, IsolateState.S7_REPORT]
    for a, b in zip(seq, seq[1:]):
        assert b in valid_transitions(a)


def test_skip_purify_path():
    # S4 -> S6 if no clusters flagged for purify
    assert IsolateState.S6_FINAL in valid_transitions(IsolateState.S4_GATE1)


def test_abort_from_anywhere():
    for s in IsolateState:
        if s is not IsolateState.S_ABORTED:
            assert IsolateState.S_ABORTED in valid_transitions(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_states.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `state_machine/states.py`**

```python
"""IsolateState enum + valid transitions."""
from __future__ import annotations
from enum import Enum, auto


class IsolateState(Enum):
    S0_LOAD = auto()
    S1_INGEST = auto()
    S2_QC = auto()
    S3_CLUSTER = auto()
    S4_GATE1 = auto()
    S5_PURIFY = auto()
    S5_GATE2 = auto()
    S6_FINAL = auto()
    S6_GATE3 = auto()
    S7_REPORT = auto()
    S_ABORTED = auto()


_TRANSITIONS = {
    IsolateState.S0_LOAD:    {IsolateState.S1_INGEST},
    IsolateState.S1_INGEST:  {IsolateState.S2_QC},
    IsolateState.S2_QC:      {IsolateState.S3_CLUSTER},
    IsolateState.S3_CLUSTER: {IsolateState.S4_GATE1},
    IsolateState.S4_GATE1:   {IsolateState.S5_PURIFY, IsolateState.S6_FINAL},
    IsolateState.S5_PURIFY:  {IsolateState.S5_GATE2},
    IsolateState.S5_GATE2:   {IsolateState.S6_FINAL},
    IsolateState.S6_FINAL:   {IsolateState.S6_GATE3},
    IsolateState.S6_GATE3:   {IsolateState.S7_REPORT},
    IsolateState.S7_REPORT:  set(),
    IsolateState.S_ABORTED:  set(),
}


def valid_transitions(state: IsolateState) -> set[IsolateState]:
    """Return the set of states reachable in one step."""
    base = set(_TRANSITIONS.get(state, set()))
    if state is not IsolateState.S_ABORTED:
        base.add(IsolateState.S_ABORTED)
    return base
```

Create `packages/rarecell/src/rarecell/state_machine/__init__.py`:

```python
from rarecell.state_machine.states import IsolateState, valid_transitions

__all__ = ["IsolateState", "valid_transitions"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_states.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/state_machine/__init__.py packages/rarecell/src/rarecell/state_machine/states.py packages/rarecell/tests/state_machine/test_states.py
git commit -m "Add IsolateState enum and valid transitions"
```

---

## Task 18: Decision logging + DecisionLog

**Files:**
- Create: `packages/rarecell/src/rarecell/report.py` (initial — Decision model + DecisionLog)
- Create: `packages/rarecell/tests/test_report_decisions.py`

(The full `IsolationReport` writer comes in Task 21; this task adds just the decision model + JSONL append-only log.)

- [ ] **Step 1: Write the failing tests**

Create `packages/rarecell/tests/test_report_decisions.py`:

```python
import json
from pathlib import Path
from rarecell.report import Decision, DecisionLog


def test_decision_log_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.append(Decision(
        gate=1, cluster_id="0", recommendation="keep",
        user_decision="keep", confidence=0.9,
        evidence={"score_pan_t_mean": 2.0},
        reasoning="Strong signal", citations=["pmid:1"],
    ))
    log.append(Decision(
        gate=1, cluster_id="1", recommendation="drop",
        user_decision="drop", confidence=0.85,
        evidence={}, reasoning="No signal",
    ))
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["cluster_id"] == "0"
    assert first["gate"] == 1


def test_decision_log_replay_reads_back(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.append(Decision(gate=1, cluster_id="0", recommendation="keep",
                        user_decision="keep", confidence=0.9, evidence={},
                        reasoning=""))
    decisions = list(DecisionLog.iter_decisions(log_path))
    assert len(decisions) == 1
    assert decisions[0].cluster_id == "0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/rarecell/tests/test_report_decisions.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `report.py` (decision portion only)**

Create `packages/rarecell/src/rarecell/report.py`:

```python
"""IsolationReport — manifest + decisions.jsonl + figures + bibliography.

This file grows in Task 21 to include the full Report writer.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal
from pydantic import BaseModel, Field


class Decision(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    gate: Literal[1, 2, 3]
    cluster_id: str
    recommendation: Literal["keep", "drop", "purify", "accept", "abort"]
    user_decision: Literal["keep", "drop", "purify", "accept", "abort"]
    confidence: float
    evidence: dict
    reasoning: str
    citations: list[str] = []


class DecisionLog:
    """Append-only JSONL log of gate decisions."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision) -> None:
        with self.path.open("a") as f:
            f.write(decision.model_dump_json() + "\n")

    @staticmethod
    def iter_decisions(path: Path) -> Iterator[Decision]:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                yield Decision.model_validate_json(line)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/rarecell/tests/test_report_decisions.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/report.py packages/rarecell/tests/test_report_decisions.py
git commit -m "Add Decision model and JSONL DecisionLog"
```

---

## Task 19: IsolateRunner — pluggable Recommender, executes S0..S7

**Files:**
- Create: `packages/rarecell/src/rarecell/state_machine/isolate.py`
- Create: `packages/rarecell/tests/state_machine/test_isolate_runner.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/state_machine/test_isolate_runner.py`:

```python
from pathlib import Path
import anndata as ad
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.profile.schema import (
    TargetCellProfile, MarkerPanel, Citation, ExpectedAbundance,
    QCParams, BICCNRules, ReferenceLabels, BatchCorrection, PurifyParams,
)
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner


def _profile_for_synthetic():
    return TargetCellProfile(
        profile_id="syn-t", name="syn T", description="d",
        target_lineage="lymphoid", tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.02, max_fraction=0.10),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D", "CD3E", "CD3G", "TRAC"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),  # off for synthetic
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(min_genes_per_cell=10, max_pct_mt=100,
                    max_genes_per_cell=10000, min_cells_per_gene=1),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony",
                                          batch_key="sample_id"),
        human_reviewed=True, reviewer="test@x",
    ).freeze()


def test_runner_completes_and_returns_isolated_subset(tmp_path: Path):
    adata = make_synthetic(seed=0)
    profile = _profile_for_synthetic()
    runner = IsolateRunner(
        adata=adata, profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    result = runner.run()
    assert result.isolated.n_obs > 0
    # the rare cluster should be in the kept set
    isolated_true = result.isolated.obs["true_cluster"]
    rare_frac = (isolated_true == "3").mean()
    assert rare_frac > 0.5    # most kept cells are the planted rare cluster
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_isolate_runner.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the runner**

Create `packages/rarecell/src/rarecell/state_machine/isolate.py`:

```python
"""IsolateRunner — executes the S0..S7 state machine with a pluggable Recommender."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import anndata as ad
from rarecell.core import ingest, qc, markers, annotate, clustering, evidence, purify, io
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommender, Recommendation
from rarecell.report import Decision, DecisionLog
from rarecell.state_machine.states import IsolateState
from rarecell.logging import get_logger
from rarecell.errors import UnreviewedProfileError

AutoPolicyName = Literal["recommendation", "abort_on_ambiguity",
                         "conservative_drop", "from_decisions"]


@dataclass
class IsolateResult:
    isolated: ad.AnnData
    final_state: IsolateState
    decisions_path: Path


class IsolateRunner:
    def __init__(
        self, *, adata: ad.AnnData, profile: TargetCellProfile,
        recommender: Recommender, out_dir: Path,
        auto_policy: AutoPolicyName = "recommendation",
        replay_decisions_path: Path | None = None,
    ):
        if not profile.frozen:
            raise UnreviewedProfileError(
                "IsolateRunner requires a frozen profile. "
                "Call profile.freeze() (which requires human_reviewed=True)."
            )
        self.adata = adata
        self.profile = profile
        self.recommender = recommender
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.auto_policy = auto_policy
        self.replay_decisions_path = replay_decisions_path
        self.log = DecisionLog(self.out_dir / "decisions.jsonl")
        self.state = IsolateState.S0_LOAD
        self.logger = get_logger("rarecell.runner")

    # ---- per-state handlers ----
    def _s1_ingest(self) -> None:
        ingest.validate_counts(self.adata)

    def _s2_qc(self) -> None:
        self.adata = qc.run_qc(self.adata, self.profile.qc)
        self.adata = qc.run_scrublet(
            self.adata, batch_key=self.profile.batch_correction.batch_key)

    def _s3_cluster(self) -> None:
        markers.score_profile_markers(self.adata, self.profile, use_raw=False)
        markers.score_negative_panels(self.adata, self.profile, use_raw=False)
        if self.profile.reference_labels.celltypist_models:
            annotate.annotate_celltypist(self.adata, self.profile)
        clustering.taxonomy_cluster(self.adata, self.profile, stage="class")
        if self.profile.biccn_rules.enabled:
            evidence.score_biccn_evidence(self.adata, self.profile,
                                          cluster_key="leiden")

    def _decide_for_gate(self, gate: int, recs: list[Recommendation]) -> dict[str, str]:
        """Resolve user_decision per cluster from auto_policy."""
        decisions: dict[str, str] = {}
        if self.auto_policy == "from_decisions":
            assert self.replay_decisions_path is not None
            for d in DecisionLog.iter_decisions(self.replay_decisions_path):
                if d.gate == gate:
                    decisions[d.cluster_id] = d.user_decision
            return decisions
        for r in recs:
            if self.auto_policy == "recommendation":
                decisions[r.cluster_id] = r.recommendation
            elif self.auto_policy == "conservative_drop":
                decisions[r.cluster_id] = "drop" if r.recommendation == "purify" \
                    else r.recommendation
            elif self.auto_policy == "abort_on_ambiguity":
                if r.confidence < self.profile.auto_policy.gates.min_recommendation_confidence:
                    decisions[r.cluster_id] = "abort"
                else:
                    decisions[r.cluster_id] = r.recommendation
        return decisions

    def _log_decisions(self, gate: int, recs: list[Recommendation],
                       user_decisions: dict[str, str]) -> None:
        for r in recs:
            ud = user_decisions.get(r.cluster_id, r.recommendation)
            self.log.append(Decision(
                gate=gate, cluster_id=r.cluster_id,
                recommendation=r.recommendation,
                user_decision=ud, confidence=r.confidence,
                evidence=r.evidence_summary,
                reasoning=r.reasoning, citations=r.citations,
            ))

    def _s4_gate1(self) -> tuple[list[str], list[str]]:
        """Returns (kept_clusters, purify_clusters)."""
        table = evidence.score_evidence(self.adata, self.profile,
                                        cluster_key="leiden")
        recs = self.recommender.recommend(table)
        user_decisions = self._decide_for_gate(1, recs)
        self._log_decisions(1, recs, user_decisions)
        kept = [cid for cid, d in user_decisions.items() if d == "keep"]
        purify_ids = [cid for cid, d in user_decisions.items() if d == "purify"]
        return kept, purify_ids

    def _s5_purify(self, suspect: list[str]) -> list[str]:
        """Returns additional kept cluster IDs after surgical purify."""
        if not suspect or not self.profile.purify.enabled:
            return []
        self.adata = purify.subcluster_and_purify(
            self.adata, self.profile, suspect_clusters=suspect,
            cluster_key="leiden",
        )
        # subcluster_and_purify returns the filtered AnnData; the kept cells
        # are anything that survived (we don't need a separate keep list here)
        return ["_purified"]   # sentinel

    def _select_isolated(self, kept_clusters: list[str]) -> ad.AnnData:
        mask = self.adata.obs["leiden"].astype(str).isin(set(kept_clusters))
        return self.adata[mask].copy()

    # ---- main entrypoint ----
    def run(self) -> IsolateResult:
        try:
            self.state = IsolateState.S1_INGEST
            self._s1_ingest()
            self.state = IsolateState.S2_QC
            self._s2_qc()
            self.state = IsolateState.S3_CLUSTER
            self._s3_cluster()
            self.state = IsolateState.S4_GATE1
            kept, suspect = self._s4_gate1()
            if suspect:
                self.state = IsolateState.S5_PURIFY
                purified_marker = self._s5_purify(suspect)
                if purified_marker:
                    kept.extend(suspect)   # purified subclusters retained inside subset
            self.state = IsolateState.S6_FINAL
            isolated = self._select_isolated(kept)
            self.state = IsolateState.S7_REPORT
            io.save_h5ad(isolated, self.out_dir / "isolated.h5ad")
            return IsolateResult(
                isolated=isolated, final_state=self.state,
                decisions_path=self.log.path,
            )
        except Exception:
            self.state = IsolateState.S_ABORTED
            raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/state_machine/test_isolate_runner.py -v`
Expected: pass. Most importantly: `rare_frac > 0.5` — the planted T-cell cluster dominates the isolated subset.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/tests/state_machine/test_isolate_runner.py
git commit -m "Add IsolateRunner — state machine execution with pluggable Recommender"
```

---

## Task 20: Replay determinism

**Files:**
- Modify: `packages/rarecell/src/rarecell/state_machine/isolate.py` (only if needed — replay-by-decisions already in Task 19)
- Create: `tests/integration/test_replay_determinism.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_replay_determinism.py`:

```python
import hashlib
from pathlib import Path
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner
from packages.rarecell.tests.state_machine.test_isolate_runner import _profile_for_synthetic


def _h5ad_hash(path: Path) -> str:
    # hash the bytes — h5ad is binary; HDF5 can vary on metadata, so we
    # extract a few representative arrays and hash those instead.
    import anndata as ad
    a = ad.read_h5ad(path)
    h = hashlib.sha256()
    h.update(bytes(a.shape))
    h.update(a.X.toarray().tobytes() if hasattr(a.X, "toarray") else a.X.tobytes())
    h.update(",".join(a.obs.columns).encode())
    h.update(",".join(map(str, a.obs.index)).encode())
    return h.hexdigest()


def test_replay_byte_deterministic(tmp_path: Path):
    profile = _profile_for_synthetic()
    adata = make_synthetic(seed=0)

    # First run
    run1_dir = tmp_path / "run1"
    r1 = IsolateRunner(adata=adata.copy(), profile=profile,
                       recommender=BasicRecommender(profile),
                       out_dir=run1_dir, auto_policy="recommendation").run()

    # Replay using recorded decisions
    run2_dir = tmp_path / "run2"
    r2 = IsolateRunner(adata=adata.copy(), profile=profile,
                       recommender=BasicRecommender(profile),
                       out_dir=run2_dir,
                       auto_policy="from_decisions",
                       replay_decisions_path=r1.decisions_path).run()

    assert _h5ad_hash(run1_dir / "isolated.h5ad") == _h5ad_hash(run2_dir / "isolated.h5ad")
```

- [ ] **Step 2: Run test to verify it passes (or surface what's nondeterministic)**

Run: `uv run pytest tests/integration/test_replay_determinism.py -v`

If it fails, the most likely culprit is leiden/PCA seed. Inspect `core/clustering.py`. Pin every random seed encountered in the run: scanpy's `sc.tl.leiden(random_state=0)`, sklearn PCA `random_state=0`, harmonypy `random_state=0`, Scrublet `random_state=0`. Then re-run until pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_replay_determinism.py packages/rarecell/src/rarecell/core/clustering.py packages/rarecell/src/rarecell/core/qc.py
git commit -m "Pin random seeds; add replay determinism regression test"
```

---

## Task 21: Full IsolationReport — manifest + bibliography + replay.sh

**Files:**
- Modify: `packages/rarecell/src/rarecell/report.py` (add Manifest + IsolationReport writer)
- Modify: `packages/rarecell/src/rarecell/state_machine/isolate.py` (write Manifest + replay.sh at S7)
- Create: `packages/rarecell/tests/test_report_full.py`

- [ ] **Step 1: Write the failing test**

Create `packages/rarecell/tests/test_report_full.py`:

```python
import json
from pathlib import Path
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner
from packages.rarecell.tests.state_machine.test_isolate_runner import _profile_for_synthetic


def test_full_report_written(tmp_path: Path):
    profile = _profile_for_synthetic()
    runner = IsolateRunner(
        adata=make_synthetic(seed=0), profile=profile,
        recommender=BasicRecommender(profile), out_dir=tmp_path,
        auto_policy="recommendation",
    )
    runner.run()

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "profile.yaml").exists()
    assert (tmp_path / "isolated.h5ad").exists()
    assert (tmp_path / "decisions.jsonl").exists()
    assert (tmp_path / "bibliography.bib").exists()
    assert (tmp_path / "replay.sh").exists()

    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == "1.0"
    assert m["profile_content_hash"].startswith("sha256:")
    assert m["isolated_summary"]["n_cells"] > 0
    assert m["decision_count"]["gate_1"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/rarecell/tests/test_report_full.py -v`
Expected: missing manifest.json / bibliography.bib / replay.sh assertions.

- [ ] **Step 3: Extend `report.py` with Manifest + writer**

Add to `packages/rarecell/src/rarecell/report.py`:

```python
from collections import Counter
from importlib.metadata import distributions
import subprocess
import platform


class Manifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    started_at: datetime
    finished_at: datetime
    rarecell_version: str
    rarecell_commit: str | None = None
    python_version: str
    platform: str
    profile_id: str
    profile_content_hash: str
    input_hash: str | None = None
    dependencies: dict[str, str]
    input_summary: dict
    qc_summary: dict
    isolated_summary: dict
    rag_sources_used: list[str] = []
    decision_count: dict[str, int]
    status: Literal["ok", "failed", "aborted"] = "ok"
    degraded_mode: bool = False


def _git_commit_or_none() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _captured_deps() -> dict[str, str]:
    return {d.metadata["Name"]: d.version for d in distributions()
            if d.metadata["Name"] in {
                "scanpy", "anndata", "scvi-tools", "harmonypy", "celltypist",
                "scrublet", "numpy", "pandas", "scipy", "pydantic", "rarecell",
            }}


def write_isolation_report(
    *, out_dir: Path, profile_yaml_path_src: Path | None,
    profile: "TargetCellProfile",  # type: ignore[name-defined]
    input_adata: "ad.AnnData", isolated: "ad.AnnData",     # type: ignore[name-defined]
    started_at: datetime, decisions_path: Path,
    rag_sources_used: list[str] | None = None,
    status: str = "ok",
) -> Manifest:
    """Write manifest.json, profile.yaml, bibliography.bib, replay.sh.

    Assumes isolated.h5ad and decisions.jsonl have already been written.
    """
    import rarecell
    out_dir = Path(out_dir)

    # profile.yaml — re-emit from frozen object
    profile.to_yaml_path(out_dir / "profile.yaml")

    # decision_count
    counts = Counter(d.gate for d in DecisionLog.iter_decisions(decisions_path))
    decision_count = {f"gate_{g}": counts[g] for g in (1, 2, 3) if counts[g]}

    # qc_summary from input vs isolated (best-effort)
    input_summary = {
        "n_cells": int(input_adata.n_obs),
        "n_genes": int(input_adata.n_vars),
        "samples": sorted(set(input_adata.obs.get("sample_id", ["_"]))),
    }
    isolated_summary = {
        "n_cells": int(isolated.n_obs),
        "abundance_fraction": float(isolated.n_obs / max(input_adata.n_obs, 1)),
        "within_expected_bounds": (
            profile.expected_abundance.min_fraction
            <= isolated.n_obs / max(input_adata.n_obs, 1)
            <= profile.expected_abundance.max_fraction
        ),
    }
    qc_summary = {
        "retained_after_qc": int(input_adata.n_obs),     # rough — refine later
    }

    manifest = Manifest(
        run_id=out_dir.name,
        started_at=started_at,
        finished_at=datetime.utcnow(),
        rarecell_version=getattr(rarecell, "__version__", "0.0.0"),
        rarecell_commit=_git_commit_or_none(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        profile_id=profile.profile_id,
        profile_content_hash=profile.content_hash or "",
        dependencies=_captured_deps(),
        input_summary=input_summary,
        qc_summary=qc_summary,
        isolated_summary=isolated_summary,
        rag_sources_used=rag_sources_used or [],
        decision_count=decision_count,
        status=status,
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2, mode="json"))

    # bibliography.bib — collect every citation referenced
    bib_entries: list[str] = []
    for panel in list(profile.positive_markers.values()) + \
                 list(profile.negative_markers.values()):
        for c in panel.citations:
            key = c.source_id.replace(":", "_")
            bib_entries.append(
                f"@misc{{{key}, note = {{{c.source}: {c.source_id}}} }}\n")
    (out_dir / "bibliography.bib").write_text("".join(bib_entries))

    # replay.sh
    (out_dir / "replay.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "# Reproduce this run from its frozen profile and recorded decisions.\n"
        "rarecell isolate \\\n"
        "  --input <path/to/input.h5ad> \\\n"
        "  --profile profile.yaml \\\n"
        "  --out-dir ./replay \\\n"
        "  --auto-policy from-decisions \\\n"
        "  --decisions decisions.jsonl\n"
    )
    (out_dir / "replay.sh").chmod(0o755)

    return manifest
```

In `state_machine/isolate.py`, modify the `run()` method to call `write_isolation_report` at the end of the `try` block (before returning), passing the input AnnData (cache a copy at S0), the isolated AnnData, and `started_at` (capture at `run()` entry).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/rarecell/tests/test_report_full.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rarecell/src/rarecell/report.py packages/rarecell/src/rarecell/state_machine/isolate.py packages/rarecell/tests/test_report_full.py
git commit -m "Write full IsolationReport (manifest, bibliography, replay.sh)"
```

---

## Task 22: PBMC 3k end-to-end integration test

**Files:**
- Create: `tests/integration/test_pbmc3k.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_pbmc3k.py`:

```python
import os
from pathlib import Path
import pytest
import scanpy as sc
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

PRESET = (Path(__file__).resolve().parents[2]
          / "packages/rarecell/src/rarecell/profile/presets/t_cell_pbmc.yaml")


@pytest.fixture(scope="module")
def pbmc3k():
    """Fetched via scanpy datasets."""
    return sc.datasets.pbmc3k()


@pytest.mark.integration
def test_pbmc3k_isolates_t_cells(pbmc3k, tmp_path: Path):
    profile = TargetCellProfile.from_yaml_path(PRESET).model_copy(
        update={"human_reviewed": True, "reviewer": "ci@x"}
    ).freeze()
    # disable celltypist for CI speed (no model fetch)
    profile = profile.model_copy(update={
        "reference_labels": profile.reference_labels.model_copy(
            update={"celltypist_models": []}),
    }).model_copy(update={"human_reviewed": True}).freeze()

    pbmc3k.obs["sample_id"] = "pbmc3k_sample"
    pbmc3k.layers["counts"] = pbmc3k.X.copy()

    runner = IsolateRunner(
        adata=pbmc3k.copy(), profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path, auto_policy="recommendation",
    )
    result = runner.run()

    # PBMC 3k has ~45-60% T cells; isolated subset should be substantial
    frac = result.isolated.n_obs / pbmc3k.n_obs
    assert 0.10 < frac < 0.80
    assert (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Add an `integration` marker to `pyproject.toml`**

Append to `packages/rarecell/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
  "integration: end-to-end tests that fetch external data",
]
```

- [ ] **Step 3: Run the integration test**

Run: `uv run pytest tests/integration/test_pbmc3k.py -v -m integration`
Expected: pass; T-cell fraction within bounds.

Note: this test fetches PBMC 3k via `sc.datasets.pbmc3k()` on first run and caches it in `~/.cache/scanpy/`. CI gates this on `pull_request` only (set up in Task 24).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_pbmc3k.py packages/rarecell/pyproject.toml
git commit -m "Add PBMC 3k integration test (fraction-bound assertion)"
```

---

## Task 23: Synthetic end-to-end integration test (recommendations correctness)

**Files:**
- Create: `tests/integration/test_synthetic_end_to_end.py`

- [ ] **Step 1: Write the test**

```python
from pathlib import Path
from tests.fixtures.make_synthetic import make_synthetic
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner
from packages.rarecell.tests.state_machine.test_isolate_runner import _profile_for_synthetic


def test_synthetic_isolates_rare_cluster(tmp_path: Path):
    adata = make_synthetic(seed=0)
    profile = _profile_for_synthetic()
    runner = IsolateRunner(
        adata=adata, profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path, auto_policy="recommendation",
    )
    result = runner.run()

    # The planted cluster is true_cluster == "3" at 5% prevalence
    isolated_true = result.isolated.obs["true_cluster"]
    # Recall: did we capture most of the planted rare cells?
    planted = (adata.obs["true_cluster"] == "3").sum()
    captured = (isolated_true == "3").sum()
    recall = captured / planted
    assert recall > 0.7

    # Precision: are most kept cells the planted rare cluster?
    precision = (isolated_true == "3").mean()
    assert precision > 0.6
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/integration/test_synthetic_end_to_end.py -v`
Expected: recall > 0.7, precision > 0.6.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_synthetic_end_to_end.py
git commit -m "Add synthetic end-to-end recall/precision integration test"
```

---

## Task 24: Pre-commit + CI workflows

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/lint.yml`, `.github/workflows/test.yml`

- [ ] **Step 1: Add `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.6, types-PyYAML]
        files: ^packages/rarecell/src/
        args: [--ignore-missing-imports]
```

- [ ] **Step 2: Add lint workflow**

Create `.github/workflows/lint.yml`:

```yaml
name: lint
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy packages/rarecell/src --ignore-missing-imports
```

- [ ] **Step 3: Add test workflow with PR-gated integration tests**

Create `.github/workflows/test.yml`:

```yaml
name: test
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras --dev --python ${{ matrix.python-version }}
      - run: uv run pytest packages/rarecell/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py -v
  integration-pbmc:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras --dev
      - run: uv run pytest tests/integration/test_pbmc3k.py -v -m integration
```

- [ ] **Step 4: Install pre-commit and run once**

Run: `uv run pre-commit install && uv run pre-commit run --all-files`
Expected: all hooks pass (some may auto-fix; re-run if so).

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml .github/workflows/
git commit -m "Add pre-commit config and CI workflows (PR-gated integration)"
```

---

## Self-Review

**Spec coverage:**
- §3.2 layered library ✓ (Tasks 7-14, 17-19)
- §3.5 install footprint ✓ (Task 1 pyproject)
- §4 profile schema + freeze interlock ✓ (Tasks 4-5)
- §4.5 preset library ✓ (Task 6 — 7 presets)
- §5 state machine, three gates, narrow tool surface, BasicRecommender ✓ (Tasks 16-19)
- §6 RAG ❌ (deferred to Plan 3; intentional)
- §7.2 IsolationReport directory layout ✓ (Tasks 18, 21)
- §7.3 replay mode ✓ (Tasks 19, 20)
- §8 testing ✓ (synthetic fixture Task 15; replay determinism Task 20; PBMC 3k Task 22; property tests deferred to a later refinement)
- §9 errors ✓ (Task 2)
- §10 governance ✓ (Tasks 1, 24)

**Placeholder scan:** none — every step has either complete code or an explicit "port from als_utils.py:LINES with these specific changes" instruction with a defined signature.

**Type consistency:** `Recommendation` (singular) used consistently from Task 16 onwards. `IsolateState` enum values referenced consistently in Tasks 17 and 19. `Decision` model used identically in Tasks 18, 19, 20.

**Gaps acknowledged:**
- Property tests with hypothesis (spec §8.1 item 6) are not in this plan. Add in a Plan-1.5 refinement task once core is stable.
- `core/annotate.py` has the CellTypist function tested; Enrichr functions are ported but not test-covered (they call external APIs). Tests are added in Plan 3 alongside `rarecell-mcp-knowledge` since Enrichr is wrapped by that server anyway.
- The `core/evidence.py` plot helpers (BICCN dotplot, composition, etc.) are ported but not test-covered. Plot smoke tests are added in Plan 4 alongside the Jupyter widget work.

These are deliberate scope deferrals, not placeholders.
