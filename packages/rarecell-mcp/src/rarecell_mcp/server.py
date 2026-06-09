"""rarecell-mcp server — exposes 4 high-level workflow tools."""

from __future__ import annotations

import json
import os
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
        """Draft a TargetCellProfile from a natural-language prompt."""
        try:
            from rarecell.agent.client import AnthropicClient
            from rarecell.agent.draft import draft_profile_from_prompt
            from rarecell.rag.knowledge import build_knowledge_session
        except ImportError as e:
            return {"error": "draft_profile requires rarecell[agent]", "detail": str(e)}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY env var not set"}

        catalog = Path.home() / ".cache/rarecell/markers.sqlite"
        cache = Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"
        session = build_knowledge_session(catalog_path=catalog, cache_path=cache)
        client = AnthropicClient(api_key=api_key)
        profile = draft_profile_from_prompt(prompt=prompt, client=client, session=session)
        profile.to_yaml_path(output_path)
        return {
            "output_path": output_path,
            "profile_id": profile.profile_id,
            "frozen": profile.frozen,
            "human_reviewed": profile.human_reviewed,
        }

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
        self,
        input_path: str,
        profile_path: str,
        out_dir: str,
        auto_policy: str = "recommendation",
    ) -> dict:
        """Drive IsolateRunner with BasicRecommender. Returns isolated cell count + report dir."""
        profile = TargetCellProfile.from_yaml_path(profile_path)
        if not profile.frozen:
            return {"error": "Profile must be frozen (call .freeze() with human_reviewed=True)."}
        adata = ad.read_h5ad(input_path)
        runner = IsolateRunner(
            adata=adata,
            profile=profile,
            recommender=BasicRecommender(profile),
            out_dir=Path(out_dir),
            auto_policy=auto_policy,
        )
        result = runner.run()
        return {
            "out_dir": out_dir,
            "isolated_n_cells": int(result.isolated.n_obs),
            "input_n_cells": int(adata.n_obs),
        }

    def inspect_report(self, report_path: str, question: str | None = None) -> dict:
        """Return the manifest.json contents."""
        manifest_path = Path(report_path) / "manifest.json"
        if not manifest_path.exists():
            return {"error": f"No manifest.json at {manifest_path}"}
        return json.loads(manifest_path.read_text())


def build_app() -> Any:
    """Build the in-process workflow app for testing."""
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
