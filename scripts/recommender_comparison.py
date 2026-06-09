"""LLM-vs-heuristic recommender comparison harness.

Runs BasicRecommender (threshold-only) and ClaudeRecommender (LLM-backed) on
the same synthetic consensus table and prints a side-by-side comparison.

Usage (mock mode — no API key required):
    uv run python scripts/recommender_comparison.py

Real mode (requires rarecell[agent] and ANTHROPIC_API_KEY):
    from anthropic import Anthropic
    client = Anthropic()
    # Wrap in a thin shim: client.messages_create(messages=...) -> dict with
    #   {"content": [{"type": "text", "text": "```json\\n{...}\\n```"}]}
    # Then call compare(table, profile, llm_client=client) directly.

The comparison table written to stdout is markdown-formatted so it can be
pasted verbatim into docs/recommender-comparison.md under the Results section.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import pandas as pd
from rarecell.agent.recommender import ClaudeRecommender
from rarecell.profile.schema import (
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    QCParams,
    TargetCellProfile,
)
from rarecell.recommender.basic import BasicRecommender

# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class FakeClaudeClient:
    """Deterministic mock that mimics the shape ClaudeRecommender expects.

    Parses the table embedded in the user message and applies rule-driven
    logic that intentionally differs from BasicRecommender on AMBIGUOUS rows
    (mid pass-fraction + mid contamination), giving the harness a real
    disagreement to surface.

    Decision logic (deliberately distinct from BasicRecommender thresholds so
    the harness is not a trivial echo):

      keep:    best_pass >= 0.6 AND contam < 0.15
      drop:    best_pass < 0.15 OR contam >= 0.5
      purify:  otherwise (includes the ambiguous mid-range)

    The LLM mock adds a reasoning string that explains the evidence conflict
    on ambiguous rows — illustrating what the real LLM would draft.
    """

    def messages_create(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> dict[str, Any]:
        user_text: str = messages[-1]["content"]

        # Extract table lines between the header and the next blank line after it.
        recs = self._parse_and_decide(user_text)
        payload = json.dumps({"recommendations": recs}, indent=2)
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"```json\n{payload}\n```",
                }
            ]
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_and_decide(self, user_text: str) -> list[dict[str, Any]]:
        """Read the table block from the user message and emit decisions."""
        # Locate the table block between "Consensus table" header and the
        # "Return a single" instruction line.
        lines = user_text.splitlines()
        in_table = False
        table_lines: list[str] = []
        for line in lines:
            if "Consensus table" in line:
                in_table = True
                continue
            if in_table:
                if line.startswith("Return a single"):
                    break
                table_lines.append(line)

        if not table_lines:
            return []

        # Parse whitespace-separated table (pandas .to_string output).
        import io

        raw = "\n".join(table_lines).strip()
        try:
            df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python")
        except Exception:
            return []

        # Identify pass_*_frac and score_*_mean columns.
        pass_cols = [c for c in df.columns if c.startswith("pass_") and c.endswith("_frac")]
        contam_col = "is_contaminant_frac" if "is_contaminant_frac" in df.columns else None

        out: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            cluster_id = str(row["cluster"])
            best_pass = max((float(row.get(c, 0.0)) for c in pass_cols), default=0.0)
            contam = float(row.get(contam_col, 0.0)) if contam_col else 0.0

            if best_pass >= 0.6 and contam < 0.15:
                rec = "keep"
                conf = 0.92
                reasoning = (
                    f"Strong positive-panel signal (best pass-frac {best_pass:.2f}) "
                    f"with negligible contamination ({contam:.2f}). Confident keep."
                )
            elif best_pass < 0.15 or contam >= 0.5:
                rec = "drop"
                conf = 0.88
                reasoning = (
                    f"Positive signal absent or marginal (best pass-frac {best_pass:.2f}); "
                    f"contamination fraction {contam:.2f} disqualifying. Recommend drop."
                )
            else:
                rec = "purify"
                conf = 0.58
                reasoning = (
                    f"CONFLICTING EVIDENCE: moderate positive signal ({best_pass:.2f}) "
                    f"alongside non-trivial contamination ({contam:.2f}). "
                    "Cannot resolve to keep/drop on thresholds alone — "
                    "recommend subclustering to separate the pure fraction."
                )

            ev_summary = {c: float(row.get(c, 0.0)) for c in pass_cols}
            if contam_col:
                ev_summary["is_contaminant_frac"] = contam

            out.append(
                {
                    "cluster_id": cluster_id,
                    "recommendation": rec,
                    "confidence": conf,
                    "evidence_summary": ev_summary,
                    "reasoning": reasoning,
                    "citations": [],
                }
            )
        return out


# ---------------------------------------------------------------------------
# Demo table builder
# ---------------------------------------------------------------------------


def build_demo_table() -> pd.DataFrame:
    """Construct a synthetic consensus table with three illustrative cluster types.

    Cluster A — clean keep:    high pass-frac, low contamination.
    Cluster B — clean drop:    low pass-frac, moderate contamination.
    Cluster C — AMBIGUOUS:     mid pass-frac + mid contamination;
                               designed so BasicRecommender and FakeClaudeClient
                               both land on "purify" but arrive via different
                               thresholds, making the reasoning differ.
    Cluster D — pure conflict: mid pass-frac + high contamination;
                               BasicRecommender drops, mock LLM purifies
                               (mock threshold is contam >= 0.5 for drop;
                               basic threshold is contam > 0.4 for drop).
    """
    return pd.DataFrame(
        {
            "cluster": ["A", "B", "C", "D"],
            "n_cells": [420, 380, 210, 190],
            "score_pan_t_mean": [2.4, 0.1, 1.1, 1.0],
            "pass_pan_t_frac": [0.82, 0.05, 0.35, 0.30],
            "is_contaminant_frac": [0.02, 0.55, 0.22, 0.45],
        }
    )


def _make_profile() -> TargetCellProfile:
    return TargetCellProfile(
        profile_id="demo",
        name="Pan-T Cell",
        description="Demonstration profile for recommender comparison.",
        target_lineage="lymphoid",
        tissue=["blood"],
        expected_abundance=ExpectedAbundance(min_fraction=0.05, max_fraction=0.40),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D", "CD3E"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:0", source="manual")],
            )
        },
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=15),
    )


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def compare(
    table: pd.DataFrame,
    profile: TargetCellProfile,
    llm_client: Any | None = None,
) -> pd.DataFrame:
    """Run both recommenders on *table* and return a per-cluster comparison DataFrame.

    Columns returned:
        cluster     — cluster ID
        heuristic   — BasicRecommender decision
        llm         — ClaudeRecommender decision (mock or real)
        agree       — True when both decisions match
        llm_reasoning — reasoning string from the LLM recommender
        heuristic_reasoning — reasoning string from the heuristic

    Args:
        table: consensus table with the columns expected by both recommenders.
        profile: TargetCellProfile instance.
        llm_client: injectable client (FakeClaudeClient by default).
    """
    if llm_client is None:
        llm_client = FakeClaudeClient()

    heuristic_recs = BasicRecommender(profile).recommend(table)
    llm_recs = ClaudeRecommender(profile=profile, client=llm_client).recommend(table)

    h_map = {r.cluster_id: r for r in heuristic_recs}
    l_map = {r.cluster_id: r for r in llm_recs}

    rows = []
    for cid in table["cluster"].astype(str):
        h = h_map.get(cid)
        ll = l_map.get(cid)
        rows.append(
            {
                "cluster": cid,
                "heuristic": h.recommendation if h else None,
                "llm": ll.recommendation if ll else None,
                "agree": (h.recommendation == ll.recommendation) if (h and ll) else None,
                "llm_reasoning": ll.reasoning if ll else "",
                "heuristic_reasoning": h.reasoning if h else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 60) -> str:
    """Wrap long text for table display."""
    return " ".join(text.split())[:width] + ("…" if len(text) > width else "")


def render_markdown(comparison: pd.DataFrame) -> str:
    """Render comparison DataFrame as a markdown table."""
    header = "| Cluster | Heuristic | LLM | Agree | LLM Reasoning (excerpt) |"
    sep = "|---------|-----------|-----|-------|--------------------------|"
    lines = [header, sep]
    for _, row in comparison.iterrows():
        agree_icon = "yes" if row["agree"] else "**NO**"
        excerpt = _wrap(str(row["llm_reasoning"]))
        lines.append(
            f"| {row['cluster']} | {row['heuristic']} | {row['llm']} | {agree_icon} | {excerpt} |"
        )
    return "\n".join(lines)


def render_summary(comparison: pd.DataFrame) -> str:
    """Return a short agreement summary string."""
    total = len(comparison)
    agreed = comparison["agree"].sum()
    disagreed = total - agreed
    lines = [
        f"Agreement: {agreed}/{total} clusters ({100 * agreed / total:.0f}%)",
        f"Disagreements: {disagreed}",
    ]
    if disagreed:
        dis = comparison[~comparison["agree"]]
        for _, row in dis.iterrows():
            lines.append(
                f"  cluster {row['cluster']}: heuristic={row['heuristic']}  llm={row['llm']}"
            )
            lines.append(
                textwrap.fill(
                    f"    LLM says: {row['llm_reasoning']}",
                    width=80,
                    subsequent_indent="    ",
                )
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    profile = _make_profile()
    table = build_demo_table()

    print("=== Synthetic consensus table ===")
    print(table.to_string(index=False))
    print()

    comparison = compare(table, profile)

    print("=== Per-cluster comparison ===")
    print(comparison[["cluster", "heuristic", "llm", "agree"]].to_string(index=False))
    print()

    print("=== Agreement summary ===")
    print(render_summary(comparison))
    print()

    print("=== Markdown table (copy into docs/recommender-comparison.md) ===")
    print(render_markdown(comparison))


if __name__ == "__main__":
    main()
