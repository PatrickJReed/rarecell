import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Make the fixtures dir importable
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from anthropic_responses import RECOMMENDATIONS_RESPONSE
from rarecell.agent.recommender import ClaudeRecommender
from rarecell.profile.schema import (
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    QCParams,
    TargetCellProfile,
)


def _profile():
    return TargetCellProfile(
        profile_id="t",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def _table():
    return pd.DataFrame(
        {
            "cluster": ["0", "1", "2"],
            "n_cells": [100, 100, 100],
            "score_pan_t_mean": [2.0, 0.1, 1.5],
            "pass_pan_t_frac": [0.9, 0.05, 0.7],
            "is_contaminant_frac": [0.02, 0.5, 0.15],
        }
    )


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
    recs = recommender.recommend(_table())
    assert recs == []
