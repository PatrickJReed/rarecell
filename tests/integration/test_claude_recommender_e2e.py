"""End-to-end: ClaudeRecommender drives IsolateRunner on synthetic fixture."""

from unittest.mock import MagicMock

from rarecell.agent.recommender import ClaudeRecommender
from rarecell.profile.schema import (
    BatchCorrection,
    BICCNRules,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    PurifyParams,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)
from rarecell.state_machine.isolate import IsolateRunner

from tests.fixtures.make_synthetic import make_synthetic


def _profile():
    return TargetCellProfile(
        profile_id="claude-t",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.02, max_fraction=0.10),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D", "CD3E", "CD3G", "TRAC"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[]),
        biccn_rules=BICCNRules(enabled=False),
        qc=QCParams(
            min_genes_per_cell=10, max_pct_mt=100, max_genes_per_cell=10000, min_cells_per_gene=1
        ),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()


def _build_keep_all_response(cluster_ids: list[str]) -> dict:
    """Build an Anthropic response that recommends keep for every cluster."""
    recs = ", ".join(
        f'{{"cluster_id": "{cid}", "recommendation": "keep", '
        f'"confidence": 0.9, "evidence_summary": {{}}, '
        f'"reasoning": "auto-keep", "citations": []}}'
        for cid in cluster_ids
    )
    json_text = f'```json\n{{"recommendations": [{recs}]}}\n```'
    return {"content": [{"type": "text", "text": json_text}]}


def test_claude_recommender_drives_runner(tmp_path):
    profile = _profile()
    adata = make_synthetic(seed=0)

    def fake_messages_create(*, messages, tools=None):
        # Extract cluster IDs from the prompt (the runner sends the consensus
        # table as text). The table's "cluster" column appears at the start
        # of each data line.
        # Crude parse: find the consensus-table section, pull out cluster
        # column values. Fallback: keep all clusters 0..99.
        return _build_keep_all_response([str(i) for i in range(100)])

    mock_client = MagicMock()
    mock_client.messages_create.side_effect = fake_messages_create

    recommender = ClaudeRecommender(profile=profile, client=mock_client)
    runner = IsolateRunner(
        adata=adata,
        profile=profile,
        recommender=recommender,
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    result = runner.run()
    assert result.isolated.n_obs > 0
    assert mock_client.messages_create.called
