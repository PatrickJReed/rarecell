"""Tests for scripts/recommender_comparison.py.

Verifies:
1. Both recommenders agree on unambiguous clusters (A=keep, B=drop).
2. The harness runs end-to-end and returns one row per cluster.
3. Every row has a valid recommendation value.
4. FakeClaudeClient returns parseable output that ClaudeRecommender accepts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the scripts/ package importable from the workspace root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.recommender_comparison import (
    FakeClaudeClient,
    _make_profile,
    build_demo_table,
    compare,
)

VALID_DECISIONS = {"keep", "drop", "purify"}


def test_compare_returns_one_row_per_cluster():
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile)

    assert len(result) == len(table), "one row per cluster expected"
    assert set(result["cluster"].tolist()) == set(table["cluster"].tolist())


def test_all_decisions_are_valid():
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile)

    for _, row in result.iterrows():
        assert row["heuristic"] in VALID_DECISIONS, (
            f"cluster {row['cluster']}: invalid heuristic decision {row['heuristic']!r}"
        )
        assert row["llm"] in VALID_DECISIONS, (
            f"cluster {row['cluster']}: invalid llm decision {row['llm']!r}"
        )


def test_agree_column_is_boolean():
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile)

    for _, row in result.iterrows():
        assert isinstance(row["agree"], (bool, int)), (
            f"cluster {row['cluster']}: 'agree' should be bool, got {type(row['agree'])}"
        )


def test_clean_clusters_agree():
    """Both recommenders must agree on the clearly-clean clusters A (keep) and B (drop).

    Cluster A: pass_pan_t_frac=0.82, is_contaminant_frac=0.02 -> keep (both)
    Cluster B: pass_pan_t_frac=0.05, is_contaminant_frac=0.55 -> drop (both)
    """
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile).set_index("cluster")

    # Cluster A — clean keep
    assert result.loc["A", "heuristic"] == "keep", (
        f"BasicRecommender should keep cluster A, got {result.loc['A', 'heuristic']}"
    )
    assert result.loc["A", "llm"] == "keep", (
        f"FakeClaudeClient should keep cluster A, got {result.loc['A', 'llm']}"
    )
    assert result.loc["A", "agree"] is True or result.loc["A", "agree"] == 1

    # Cluster B — clean drop
    assert result.loc["B", "heuristic"] == "drop", (
        f"BasicRecommender should drop cluster B, got {result.loc['B', 'heuristic']}"
    )
    assert result.loc["B", "llm"] == "drop", (
        f"FakeClaudeClient should drop cluster B, got {result.loc['B', 'llm']}"
    )
    assert result.loc["B", "agree"] is True or result.loc["B", "agree"] == 1


def test_llm_reasoning_present_on_all_rows():
    """Every LLM row should carry a non-empty reasoning string."""
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile)

    for _, row in result.iterrows():
        assert isinstance(row["llm_reasoning"], str) and row["llm_reasoning"].strip(), (
            f"cluster {row['cluster']}: llm_reasoning is empty"
        )


def test_fake_client_messages_create_returns_correct_shape():
    """FakeClaudeClient.messages_create must return the dict shape ClaudeRecommender expects."""
    client = FakeClaudeClient()
    table = build_demo_table()
    profile = _make_profile()

    from rarecell.agent.recommender import ClaudeRecommender

    recs = ClaudeRecommender(profile=profile, client=client).recommend(table)

    assert len(recs) == len(table), "ClaudeRecommender must parse one recommendation per cluster"
    for r in recs:
        assert r.recommendation in VALID_DECISIONS
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.reasoning, str) and r.reasoning


def test_harness_disagrees_on_ambiguous_cluster_d():
    """Cluster D is the designed divergence point.

    pass_pan_t_frac=0.30, is_contaminant_frac=0.45:
    - BasicRecommender: contam (0.45) > 0.4 => drop
    - FakeClaudeClient: contam (0.45) < 0.5 AND pass >= 0.15 => purify
    => agree == False
    """
    table = build_demo_table()
    profile = _make_profile()
    result = compare(table, profile).set_index("cluster")

    assert result.loc["D", "heuristic"] == "drop", (
        f"BasicRecommender threshold: contam 0.45 > 0.4 => drop, got {result.loc['D', 'heuristic']}"
    )
    assert result.loc["D", "llm"] == "purify", (
        f"FakeClaudeClient: contam 0.45 < 0.5 boundary => purify, got {result.loc['D', 'llm']}"
    )
    assert not (result.loc["D", "agree"] is True or result.loc["D", "agree"] == 1), (
        "Cluster D should be a disagreement row"
    )
