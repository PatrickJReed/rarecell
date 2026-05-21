"""IsolationReport — manifest + decisions.jsonl + figures + bibliography.

This file grows in Task 21 to include the full Report writer.
"""

from __future__ import annotations

import platform
import subprocess
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Decision(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


_TRACKED_DEPS = {
    "scanpy",
    "anndata",
    "scvi-tools",
    "harmonypy",
    "celltypist",
    "scrublet",
    "numpy",
    "pandas",
    "scipy",
    "pydantic",
    "rarecell",
}


def _captured_deps() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in distributions():
        name = d.metadata["Name"]
        if name in _TRACKED_DEPS:
            out[name] = d.version
    return out


def write_isolation_report(
    *,
    out_dir: Path,
    profile,
    input_n_obs: int,
    input_n_vars: int,
    input_sample_ids: list[str],
    isolated,
    started_at: datetime,
    decisions_path: Path,
    rag_sources_used: list[str] | None = None,
    status: str = "ok",
) -> Manifest:
    """Write manifest.json, profile.yaml, bibliography.bib, replay.sh into out_dir.

    isolated.h5ad and decisions.jsonl are written elsewhere.
    """
    import rarecell

    out_dir = Path(out_dir)

    # profile.yaml — re-emit frozen object
    profile.to_yaml_path(out_dir / "profile.yaml")

    # decision_count from decisions.jsonl
    counts = Counter(d.gate for d in DecisionLog.iter_decisions(decisions_path))
    decision_count = {f"gate_{g}": counts.get(g, 0) for g in (1, 2, 3) if counts.get(g, 0) > 0}

    # input_summary
    input_summary = {
        "n_cells": int(input_n_obs),
        "n_genes": int(input_n_vars),
        "samples": list(input_sample_ids),
    }

    # isolated_summary
    iso_frac = isolated.n_obs / max(input_n_obs, 1)
    isolated_summary = {
        "n_cells": int(isolated.n_obs),
        "abundance_fraction": float(iso_frac),
        "within_expected_bounds": bool(
            profile.expected_abundance.min_fraction
            <= iso_frac
            <= profile.expected_abundance.max_fraction
        ),
    }

    qc_summary = {"retained_after_qc": int(input_n_obs)}

    manifest = Manifest(
        run_id=out_dir.name,
        started_at=started_at,
        finished_at=datetime.now(UTC),
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
        status=status,  # type: ignore[arg-type]
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    # bibliography.bib — every citation referenced in the profile
    bib_entries: list[str] = []
    seen: set[tuple[str, str]] = set()
    panels = list(profile.positive_markers.values()) + list(profile.negative_markers.values())
    for panel in panels:
        for c in panel.citations:
            key_id = (c.source, c.source_id)
            if key_id in seen:
                continue
            seen.add(key_id)
            key = c.source_id.replace(":", "_").replace("/", "_")
            bib_entries.append(f"@misc{{{key}, note = {{{c.source}: {c.source_id}}} }}\n")
    (out_dir / "bibliography.bib").write_text("".join(bib_entries))

    # replay.sh — bash recipe to reproduce
    replay_path = out_dir / "replay.sh"
    replay_path.write_text(
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
    replay_path.chmod(0o755)

    return manifest
