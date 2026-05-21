# rarecell CLI + Exposed MCP Server — Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the two highest-impact front-ends that make rarecell v0.1 publicly usable: a `rarecell` Typer CLI exposing `draft | isolate | review`, and a standalone `rarecell-mcp` package — a FastMCP server wrapping the workflow for Claude Desktop / Claude Code / Cursor users.

**Architecture:** `rarecell.cli` is a thin Typer wrapper around the public `rarecell` API (IsolateRunner, profile drafting, IsolationReport reader). `rarecell-mcp` is a new sibling package under `packages/rarecell-mcp/` advertising four high-level tools (`draft_profile`, `validate_input`, `run_isolation`, `inspect_report`) — it depends on `rarecell` and `rarecell-mcp-knowledge`. Both front-ends share the same library; no duplicated business logic.

**Tech Stack:** Python 3.11+, Typer (already in dep tree via rich), `rarecell` (Plan 1 + Plan 3), `rarecell-mcp-knowledge` (Plan 2), FastMCP, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-rarecell-agentic-isolation-design.md` §3.3 (exposed MCP) and §7.1 (front-ends).

**Plans 1, 2, 3 status:** All merged to main. 98 unit/integration tests pass; 1 pre-existing flake noted (synthetic recall threshold), 1 network-gated PBMC test only runs in CI on PRs.

**Out of Plan 4 (deferred to Plan 5):**
- Jupyter widgets / cell magics (`%%rarecell isolate`)
- Full `review` agent mode (replay + anomaly surfacing). Plan 4 ships a basic `rarecell review` CLI subcommand that reads a manifest and prints a summary; agentic anomaly analysis is Plan 5.

---

## File Structure

```
packages/rarecell/src/rarecell/
├── cli.py                            # NEW — Typer CLI entry point

packages/rarecell-mcp/                # NEW package
├── pyproject.toml
├── README.md
├── src/rarecell_mcp/
│   ├── __init__.py
│   ├── server.py                     # FastMCP server with 4 workflow tools
│   └── cli.py                        # `rarecell-mcp serve` entry point
└── tests/
    ├── test_server.py
    └── test_cli.py
```

Tests:

```
packages/rarecell/tests/
└── test_cli.py                       # CLI subcommand tests
```

`packages/rarecell/pyproject.toml` gets:
- `typer>=0.12` added to `dependencies`
- A `[project.scripts]` entry: `rarecell = "rarecell.cli:app"`

---

## Task 1: Add `typer` and the `rarecell` console script

**Files:**
- Modify: `packages/rarecell/pyproject.toml`

- [ ] **Step 1: Read existing `packages/rarecell/pyproject.toml`**

Locate the `dependencies` block. Locate the existing entries (currently no `[project.scripts]` section for `rarecell` itself).

- [ ] **Step 2: Add `typer` dep and the console script**

Add `"typer>=0.12"` to the dependencies list (alphabetically: between `structlog` and the next entry).

Add a new section near the end:

```toml
[project.scripts]
rarecell = "rarecell.cli:app"
```

- [ ] **Step 3: Sync**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv sync --all-packages --all-extras --dev
```

Expected: typer resolves; no errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add packages/rarecell/pyproject.toml uv.lock && git commit -m "Add typer dep + rarecell console script entry"
```

---

## Task 2: `rarecell` CLI — `isolate` subcommand

**Files:**
- Create: `packages/rarecell/src/rarecell/cli.py`
- Create: `packages/rarecell/tests/test_cli.py`

The CLI's primary subcommand. Loads a frozen profile YAML, opens an AnnData, picks `BasicRecommender` (or `ClaudeRecommender` if `--use-claude` and `[agent]` installed), runs `IsolateRunner`, prints summary.

- [ ] **Step 1: Write the failing test**

`packages/rarecell/tests/test_cli.py`:

```python
"""Tests for the rarecell CLI."""
from pathlib import Path
from typer.testing import CliRunner
from rarecell.cli import app


runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "isolate" in result.stdout
    assert "draft" in result.stdout
    assert "review" in result.stdout


def test_isolate_subcommand_runs_on_synthetic(tmp_path: Path):
    """Smoke test: write a tiny synthetic AnnData + minimal profile, invoke
    `rarecell isolate`, expect exit 0 and an isolated.h5ad output."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic
    import yaml

    adata = make_synthetic(seed=0)
    adata_path = tmp_path / "input.h5ad"
    adata.write_h5ad(adata_path)

    # Minimal profile YAML (frozen) — mirror the synthetic test's profile
    profile_yaml = {
        "schema_version": "1.0",
        "profile_id": "cli-test-tcell",
        "name": "CLI test T cells",
        "description": "Smoke test profile",
        "target_lineage": "lymphoid",
        "tissue": ["pbmc"],
        "expected_abundance": {"min_fraction": 0.02, "max_fraction": 0.10},
        "positive_markers": {
            "pan_t": {
                "genes": ["CD3D", "CD3E", "CD3G", "TRAC"],
                "threshold_z": 1.0,
                "citations": [{"source_id": "pmid:1", "source": "europepmc"}],
            }
        },
        "negative_markers": {},
        "qc": {"min_genes_per_cell": 10, "max_pct_mt": 100,
                "max_genes_per_cell": 10000, "min_cells_per_gene": 1},
        "purify": {"enabled": False},
        "batch_correction": {"in_dataset": "harmony", "batch_key": "sample_id"},
        "human_reviewed": True,
        "reviewer": "ci@x",
        "frozen": False,                  # will be frozen via CLI freeze step
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_yaml))

    # Freeze the profile in-place so `rarecell isolate` can use it
    from rarecell.profile.schema import TargetCellProfile
    frozen = TargetCellProfile.from_yaml_path(profile_path).freeze()
    frozen.to_yaml_path(profile_path)

    out_dir = tmp_path / "run"
    result = runner.invoke(app, [
        "isolate",
        "--input", str(adata_path),
        "--profile", str(profile_path),
        "--out-dir", str(out_dir),
        "--auto-policy", "recommendation",
    ])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "isolated.h5ad").exists()
    assert (out_dir / "manifest.json").exists()
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest packages/rarecell/tests/test_cli.py -v
```

- [ ] **Step 3: Write `cli.py`**

```python
"""rarecell CLI — Typer commands wrapping the public API."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import anndata as ad

from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner


app = typer.Typer(
    help="rarecell — profile-driven rare-cell isolation from single-cell RNA-seq.",
    no_args_is_help=True,
)


def _load_recommender(profile, use_claude: bool):
    """Pick BasicRecommender (default) or ClaudeRecommender (if --use-claude)."""
    if not use_claude:
        return BasicRecommender(profile)
    try:
        from rarecell.agent.client import AnthropicClient
        from rarecell.agent.recommender import ClaudeRecommender
    except ImportError as e:
        raise typer.BadParameter(
            "--use-claude requires the [agent] extra. "
            "Install with: pip install 'rarecell[agent]'"
        ) from e
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise typer.BadParameter(
            "ANTHROPIC_API_KEY env var is required with --use-claude."
        )
    client = AnthropicClient(api_key=api_key)
    return ClaudeRecommender(profile=profile, client=client)


@app.command()
def isolate(
    input: Annotated[Path, typer.Option("--input", help="Path to input AnnData (.h5ad)")],
    profile_path: Annotated[Path, typer.Option("--profile", help="Path to frozen profile YAML")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory for IsolationReport")],
    auto_policy: Annotated[str, typer.Option("--auto-policy", help="recommendation | abort_on_ambiguity | conservative_drop | from_decisions")] = "recommendation",
    use_claude: Annotated[bool, typer.Option("--use-claude", help="Use ClaudeRecommender instead of BasicRecommender")] = False,
    decisions: Annotated[Optional[Path], typer.Option("--decisions", help="Path to decisions.jsonl for replay (with --auto-policy from_decisions)")] = None,
):
    """Run rare-cell isolation on an AnnData and write an IsolationReport."""
    typer.echo(f"Loading profile from {profile_path}")
    profile = TargetCellProfile.from_yaml_path(profile_path)
    if not profile.frozen:
        raise typer.BadParameter(
            f"Profile at {profile_path} is not frozen. "
            "Call profile.freeze() (requires human_reviewed=True) before running."
        )
    typer.echo(f"Loading AnnData from {input}")
    adata = ad.read_h5ad(input)
    typer.echo(f"  → {adata.n_obs} cells, {adata.n_vars} genes")

    recommender = _load_recommender(profile, use_claude)
    typer.echo(f"Recommender: {type(recommender).__name__}")

    runner = IsolateRunner(
        adata=adata, profile=profile, recommender=recommender,
        out_dir=out_dir, auto_policy=auto_policy,
        replay_decisions_path=decisions,
    )
    result = runner.run()
    typer.echo(f"Isolated {result.isolated.n_obs} cells "
               f"({result.isolated.n_obs / max(adata.n_obs, 1):.2%} of input)")
    typer.echo(f"Report written to {out_dir}")


@app.command()
def draft(
    prompt: Annotated[str, typer.Option("--prompt", help="Natural-language description of the target population")],
    out: Annotated[Path, typer.Option("--out", help="Output path for draft profile YAML")],
    catalog_path: Annotated[Optional[Path], typer.Option("--catalog-path", help="Path to markers SQLite (default: ~/.cache/rarecell/markers.sqlite)")] = None,
):
    """Draft a TargetCellProfile from a natural-language prompt (requires [agent] extra)."""
    import os
    try:
        from rarecell.agent.client import AnthropicClient
        from rarecell.agent.draft import draft_profile_from_prompt
        from rarecell.rag.knowledge import build_knowledge_session
    except ImportError as e:
        raise typer.BadParameter(
            "draft requires the [agent] extra. "
            "Install with: pip install 'rarecell[agent]'"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY env var is required for draft.")

    catalog = catalog_path or (Path.home() / ".cache/rarecell/markers.sqlite")
    cache = Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"
    session = build_knowledge_session(catalog_path=catalog, cache_path=cache)
    client = AnthropicClient(api_key=api_key)
    profile = draft_profile_from_prompt(prompt=prompt, client=client, session=session)
    profile.to_yaml_path(out)
    typer.echo(f"Draft profile written to {out}")
    typer.echo("Review it, set human_reviewed=true and provide reviewer, then freeze.")


@app.command()
def review(
    report: Annotated[Path, typer.Option("--report", help="Path to an IsolationReport directory")],
):
    """Print a summary of an IsolationReport (Plan 4 v1 — agentic anomaly analysis is Plan 5)."""
    manifest_path = report / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"No manifest.json at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    typer.echo(f"Report: {manifest['run_id']}")
    typer.echo(f"  Started:  {manifest['started_at']}")
    typer.echo(f"  Finished: {manifest['finished_at']}")
    typer.echo(f"  Profile:  {manifest['profile_id']} ({manifest['profile_content_hash']})")
    typer.echo(f"  Input:    {manifest['input_summary']['n_cells']} cells")
    typer.echo(f"  Isolated: {manifest['isolated_summary']['n_cells']} cells "
               f"({manifest['isolated_summary']['abundance_fraction']:.4f} fraction)")
    typer.echo(f"  Within expected bounds: {manifest['isolated_summary']['within_expected_bounds']}")
    typer.echo(f"  Decisions: {manifest['decision_count']}")
    typer.echo(f"  Status:    {manifest['status']}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run, expect 2 pass**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest packages/rarecell/tests/test_cli.py -v
```

- [ ] **Step 5: Verify the console script**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run rarecell --help
```
Expected: prints the Typer-generated help with `isolate`, `draft`, `review` subcommands.

- [ ] **Step 6: Lint + commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run ruff check packages/rarecell/src/rarecell/cli.py packages/rarecell/tests/test_cli.py
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add packages/rarecell/src/rarecell/cli.py packages/rarecell/tests/test_cli.py && git commit -m "Add rarecell CLI with isolate, draft, review subcommands"
```

---

## Task 3: `rarecell-mcp` package scaffold

**Files:**
- Create: `packages/rarecell-mcp/pyproject.toml`
- Create: `packages/rarecell-mcp/README.md`
- Create: `packages/rarecell-mcp/src/rarecell_mcp/__init__.py`

- [ ] **Step 1: `pyproject.toml`**

```toml
[project]
name = "rarecell-mcp"
version = "0.1.0.dev0"
description = "Exposed FastMCP workflow server for rarecell (drives isolation from any MCP client)"
authors = [{name = "Patrick Reed", email = "patrickjenningsreed@gmail.com"}]
license = "Apache-2.0"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "fastmcp>=0.4",
  "rarecell",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "respx>=0.21",
  "ruff",
]

[project.scripts]
rarecell-mcp = "rarecell_mcp.cli:main"

[tool.uv.sources]
rarecell = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rarecell_mcp"]
```

- [ ] **Step 2: `README.md`**

```markdown
# rarecell-mcp

Exposed FastMCP workflow server for rarecell. Lets any MCP client
(Claude Desktop, Claude Code, Cursor) drive a rarecell isolation
end-to-end with four high-level tools:

- `draft_profile(prompt, output_path)` — NL → draft profile YAML
- `validate_input(adata_path)` — verify counts + gene IDs
- `run_isolation(input_path, profile_path, out_dir, auto_policy?)` — drive IsolateRunner
- `inspect_report(report_path, question?)` — summarize a past run

## Install

```bash
pip install rarecell-mcp
```

## Run

```bash
rarecell-mcp serve
```

## Wire into Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "rarecell": {
      "command": "rarecell-mcp",
      "args": ["serve"]
    }
  }
}
```

This is pre-release v0.x.
```

- [ ] **Step 3: Package init**

```python
"""rarecell-mcp — exposed FastMCP workflow server for rarecell."""

__version__ = "0.1.0.dev0"
```

- [ ] **Step 4: Sync + smoke-import**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv sync --all-packages --all-extras --dev
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run python -c "import rarecell_mcp; print(rarecell_mcp.__version__)"
```
Expected: prints `0.1.0.dev0`.

- [ ] **Step 5: Commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add packages/rarecell-mcp/ uv.lock && git commit -m "Scaffold rarecell-mcp package"
```

---

## Task 4: `rarecell-mcp` server — four workflow tools

**Files:**
- Create: `packages/rarecell-mcp/src/rarecell_mcp/server.py`
- Create: `packages/rarecell-mcp/tests/test_server.py`

The server exposes high-level workflow tools (no raw clustering/QC). It composes the rarecell library directly — no subprocess to a CLI.

- [ ] **Step 1: Write the failing test**

`packages/rarecell-mcp/tests/test_server.py`:

```python
"""In-process smoke test of rarecell-mcp server."""
import json
from pathlib import Path
import pytest

from rarecell_mcp.server import build_app


@pytest.fixture
def app():
    return build_app()


def test_app_advertises_four_tools(app):
    names = sorted(app.list_tool_names())
    assert "draft_profile" in names
    assert "validate_input" in names
    assert "run_isolation" in names
    assert "inspect_report" in names


def test_validate_input_finds_counts(app, tmp_path: Path):
    """validate_input on a synthetic AnnData returns the counts location."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/fixtures"))
    from make_synthetic import make_synthetic
    adata = make_synthetic(seed=0)
    p = tmp_path / "input.h5ad"
    adata.write_h5ad(p)
    result = app.call_tool("validate_input", {"adata_path": str(p)})
    assert result["counts_location"] in ("X", "counts", "raw")
    assert result["n_obs"] == 5000


def test_inspect_report_reads_manifest(app, tmp_path: Path):
    """inspect_report reads a manifest.json and returns its key fields."""
    manifest = {
        "schema_version": "1.0",
        "run_id": "test-run",
        "started_at": "2026-05-20T10:00:00+00:00",
        "finished_at": "2026-05-20T10:02:00+00:00",
        "profile_id": "test-profile",
        "profile_content_hash": "sha256:abc",
        "isolated_summary": {"n_cells": 100, "abundance_fraction": 0.05,
                              "within_expected_bounds": True},
        "input_summary": {"n_cells": 2000, "n_genes": 500, "samples": ["s1"]},
        "decision_count": {"gate_1": 5, "gate_2": 2},
        "status": "ok",
    }
    report_dir = tmp_path / "run"
    report_dir.mkdir()
    (report_dir / "manifest.json").write_text(json.dumps(manifest))

    result = app.call_tool("inspect_report", {"report_path": str(report_dir)})
    assert result["run_id"] == "test-run"
    assert result["isolated_summary"]["n_cells"] == 100
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest packages/rarecell-mcp/tests/test_server.py -v
```

- [ ] **Step 3: Write `server.py`**

```python
"""rarecell-mcp server — exposes 4 high-level workflow tools."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import anndata as ad

from rarecell.core import ingest
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner


class RarecellWorkflowApp:
    """Tool implementations for the exposed MCP server."""

    def draft_profile(self, prompt: str, output_path: str) -> dict:
        """Draft a TargetCellProfile from a natural-language prompt.

        Requires the [agent] extra of rarecell + ANTHROPIC_API_KEY env var
        + a seeded markers SQLite. Writes the un-frozen draft to output_path.
        """
        import os
        try:
            from rarecell.agent.client import AnthropicClient
            from rarecell.agent.draft import draft_profile_from_prompt
            from rarecell.rag.knowledge import build_knowledge_session
        except ImportError as e:
            return {"error": "draft_profile requires rarecell[agent]",
                    "detail": str(e)}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY env var not set"}

        catalog = Path.home() / ".cache/rarecell/markers.sqlite"
        cache = Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"
        session = build_knowledge_session(catalog_path=catalog, cache_path=cache)
        client = AnthropicClient(api_key=api_key)
        profile = draft_profile_from_prompt(
            prompt=prompt, client=client, session=session)
        profile.to_yaml_path(output_path)
        return {"output_path": output_path, "profile_id": profile.profile_id,
                "frozen": profile.frozen, "human_reviewed": profile.human_reviewed}

    def validate_input(self, adata_path: str) -> dict:
        """Check raw counts and basic shape of an AnnData."""
        adata = ad.read_h5ad(adata_path)
        try:
            counts_loc = ingest.validate_counts(adata)
        except Exception as e:
            return {"error": str(e)}
        return {
            "adata_path": adata_path,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "counts_location": counts_loc,
            "samples": sorted(set(map(str, adata.obs.get("sample_id", ["_"])))),
        }

    def run_isolation(
        self, input_path: str, profile_path: str, out_dir: str,
        auto_policy: str = "recommendation",
    ) -> dict:
        """Drive IsolateRunner with BasicRecommender. Returns isolated cell count + report dir."""
        profile = TargetCellProfile.from_yaml_path(profile_path)
        if not profile.frozen:
            return {"error": "Profile must be frozen (call .freeze() with human_reviewed=True)."}
        adata = ad.read_h5ad(input_path)
        runner = IsolateRunner(
            adata=adata, profile=profile,
            recommender=BasicRecommender(profile),
            out_dir=Path(out_dir), auto_policy=auto_policy,
        )
        result = runner.run()
        return {
            "out_dir": out_dir,
            "isolated_n_cells": int(result.isolated.n_obs),
            "input_n_cells": int(adata.n_obs),
        }

    def inspect_report(self, report_path: str, question: str | None = None) -> dict:
        """Return the manifest.json contents (and ignore `question` for v1)."""
        manifest_path = Path(report_path) / "manifest.json"
        if not manifest_path.exists():
            return {"error": f"No manifest.json at {manifest_path}"}
        return json.loads(manifest_path.read_text())


def build_app() -> Any:
    """Build the in-process workflow app.

    Returns an object with .list_tool_names() and .call_tool(name, kwargs)
    that mirrors the FastMCP tool API for testing.
    """
    workflow = RarecellWorkflowApp()
    tools = {
        "draft_profile": workflow.draft_profile,
        "validate_input": workflow.validate_input,
        "run_isolation": workflow.run_isolation,
        "inspect_report": workflow.inspect_report,
    }

    class _App:
        knowledge_app = workflow
        tool_map = tools

        def list_tool_names(self) -> list[str]:
            return list(tools.keys())

        def call_tool(self, name: str, kwargs: dict):
            return tools[name](**kwargs)

    return _App()


def build_fastmcp_app() -> Any:
    """Build the production FastMCP server with tools registered."""
    from fastmcp import FastMCP
    inner = build_app()
    mcp = FastMCP("rarecell-mcp")
    for name, fn in inner.tool_map.items():
        mcp.tool(name)(fn)
    return mcp
```

- [ ] **Step 4: Run, expect 3 pass**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest packages/rarecell-mcp/tests/test_server.py -v
```

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run ruff check packages/rarecell-mcp/
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add packages/rarecell-mcp/src/rarecell_mcp/server.py packages/rarecell-mcp/tests/test_server.py && git commit -m "Add rarecell-mcp server with 4 workflow tools"
```

---

## Task 5: `rarecell-mcp` CLI (`serve` subcommand)

**Files:**
- Create: `packages/rarecell-mcp/src/rarecell_mcp/cli.py`
- Create: `packages/rarecell-mcp/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`packages/rarecell-mcp/tests/test_cli.py`:

```python
import pytest
from rarecell_mcp.cli import main_with_args


def test_serve_help_exits_clean(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main_with_args(["serve", "--help"])
    assert exc_info.value.code == 0


def test_top_level_help_exits_clean(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main_with_args(["--help"])
    assert exc_info.value.code == 0
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Write `cli.py`**

```python
"""rarecell-mcp CLI: serve."""
from __future__ import annotations
import argparse
import sys
from typing import Sequence


def _serve(args: argparse.Namespace) -> int:
    from rarecell_mcp.server import build_fastmcp_app
    app = build_fastmcp_app()
    app.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rarecell-mcp")
    sub = p.add_subparsers(dest="cmd", required=True)
    serve_p = sub.add_parser("serve", help="Run the FastMCP server")
    serve_p.set_defaults(func=_serve)
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

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest packages/rarecell-mcp/tests/test_cli.py -v
```

- [ ] **Step 5: Verify console script**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run rarecell-mcp --help
```
Expected: prints usage with `serve` subcommand.

- [ ] **Step 6: Commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add packages/rarecell-mcp/src/rarecell_mcp/cli.py packages/rarecell-mcp/tests/test_cli.py && git commit -m "Add rarecell-mcp CLI with serve subcommand"
```

---

## Task 6: CI wiring + root README polish

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `pyproject.toml` (root — add new package's tests to `testpaths`)
- Modify: `README.md` (root)

- [ ] **Step 1: Update root `pyproject.toml` testpaths**

Edit the `[tool.pytest.ini_options]` block: change `testpaths` to include `packages/rarecell-mcp/tests`:

```toml
testpaths = [
  "tests",
  "packages/rarecell/tests",
  "packages/rarecell-mcp-knowledge/tests",
  "packages/rarecell-mcp/tests",
]
```

- [ ] **Step 2: Update `.github/workflows/test.yml`**

In the `unit` job's pytest command, add `packages/rarecell-mcp/tests` to the list (after `packages/rarecell-mcp-knowledge/tests`):

```yaml
      - run: uv run pytest packages/rarecell/tests packages/rarecell-mcp-knowledge/tests packages/rarecell-mcp/tests tests/fixtures tests/integration/test_replay_determinism.py tests/integration/test_synthetic_end_to_end.py tests/integration/test_claude_recommender_e2e.py -v
```

- [ ] **Step 3: Update root README**

Append (or insert under the existing "Monorepo layout" section) a bullet:

```markdown
- `packages/rarecell-mcp/` — exposed FastMCP workflow server. Drive `draft |
  validate | isolate | inspect` from Claude Desktop / Claude Code / Cursor.
```

Also append a new top-level section near the bottom:

```markdown
## CLI

After `pip install rarecell`:

```bash
rarecell isolate --input adata.h5ad --profile profile.yaml --out-dir runs/run1
rarecell draft --prompt "rare T cells in PBMC" --out draft.yaml
rarecell review --report runs/run1
```

Drafting requires the `[agent]` extra and `ANTHROPIC_API_KEY` in the environment.

## Driving from an MCP client

Install `rarecell-mcp` and wire it into Claude Desktop or Claude Code:

```bash
pip install rarecell-mcp
```

```json
{"mcpServers": {"rarecell": {"command": "rarecell-mcp", "args": ["serve"]}}}
```
```

- [ ] **Step 4: Run the full suite**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run pytest 2>&1 | tail -5
```
Expected: ~105 tests pass (98 from Plans 1-3 + new from Plan 4); 1-2 pre-existing flaky tests (synthetic recall threshold, PBMC integration if no network).

- [ ] **Step 5: Lint clean check**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 6: Commit**

```bash
cd /Users/patrickreed/Sandbox/rarecell/.claude/worktrees/plan-4-frontends && git add .github/workflows/test.yml pyproject.toml README.md && git commit -m "Wire Plan 4 tests into CI; document CLI and rarecell-mcp"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Plan coverage |
|---|---|
| §3.3 exposed MCP server with 4 tools | Tasks 3-5 |
| §7.1 CLI front-end with isolate/draft/review | Task 2 |
| §7.1 Jupyter widgets | **Deferred to Plan 5** (documented in plan header) |
| Manifest-driven `review` mode | Task 2 ships a basic CLI summary; agentic review is **Plan 5** |
| Cross-package isolation | rarecell-mcp depends on rarecell + rarecell-mcp-knowledge, no reach into internals |

**Placeholder scan:** no TBD/TODO. Every step has complete code or precise modification.

**Type consistency:**
- `IsolateRunner` constructor (adata=, profile=, recommender=, out_dir=, auto_policy=, replay_decisions_path=) consistent in Task 2 CLI and Task 4 MCP server.
- `TargetCellProfile.from_yaml_path` consistent across Tasks 2, 4.
- `build_app()` mirrors the rarecell-mcp-knowledge pattern from Plan 2 (`list_tool_names`, `call_tool`, `tool_map`).

**Deliberate scope cuts (Plan 5 candidates):**
- Jupyter widgets / `%%rarecell` cell magics
- Agentic `review` mode (anomaly surfacing, decision-vs-recommendation deltas)
- Conda packaging
- Docker images
- Live Anthropic / Europe PMC integration tests on a nightly CI cron
- Tightening the `Retriever` protocol (deferred from Plan 3)
- Splitting `core/evidence.py` into `score.py` + `plots.py` (deferred from Plan 1)
- The 24 mypy strict-mode errors (deferred from Plan 1)
- Wiring the CellTypist-from-BICCN companion workflow ([[rarecell-celltypist-from-biccn]] memory)
