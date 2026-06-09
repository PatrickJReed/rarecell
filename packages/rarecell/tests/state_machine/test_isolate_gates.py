"""Tests for S5_GATE2 and S6_GATE3 wiring in IsolateRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rarecell.errors import IsolationAbortedError
from rarecell.profile.schema import (
    AutoPolicy,
    BatchCorrection,
    BICCNRules,
    Citation,
    ExpectedAbundance,
    GateAutoPolicy,
    MarkerPanel,
    PurifyParams,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

from tests.fixtures.make_synthetic import make_synthetic


def _profile_with_purify() -> TargetCellProfile:
    return TargetCellProfile(
        profile_id="syn-t",
        name="syn T",
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
            min_genes_per_cell=10,
            max_pct_mt=100,
            max_genes_per_cell=10000,
            min_cells_per_gene=1,
        ),
        purify=PurifyParams(enabled=True, min_cluster_purity=0.5),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()


def test_gate2_decisions_logged_when_purify_runs(tmp_path: Path) -> None:
    """When the runner enters S5_PURIFY, it must enter S5_GATE2 and log gate=2."""
    profile = _profile_with_purify()
    runner = IsolateRunner(
        adata=make_synthetic(seed=0),
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    runner.run()

    decisions = (tmp_path / "decisions.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(line) for line in decisions]
    gates = [d["gate"] for d in parsed]
    # If gate 1 produced any "purify" decisions, gate 2 must have run too.
    g1_purify = any(d["user_decision"] == "purify" for d in parsed if d["gate"] == 1)
    if g1_purify:
        assert 2 in gates, "Gate 2 must run when any gate-1 decision is 'purify'"


def test_gate3_aborts_when_abundance_out_of_bounds(tmp_path: Path) -> None:
    """Narrow expected_abundance + abort_on_anomaly => IsolationAbortedError."""
    profile = TargetCellProfile(
        profile_id="abort-t",
        name="t",
        description="d",
        target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(
            min_fraction=0.0001,
            max_fraction=0.001,
        ),
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
            min_genes_per_cell=10,
            max_pct_mt=100,
            max_genes_per_cell=10000,
            min_cells_per_gene=1,
        ),
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        auto_policy=AutoPolicy(
            gates=GateAutoPolicy(
                gate3_final="abort_on_anomaly",
                max_abundance_deviation=2.0,
            )
        ),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()

    runner = IsolateRunner(
        adata=make_synthetic(seed=0),
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
        auto_policy="recommendation",
    )
    with pytest.raises(IsolationAbortedError, match="abundance"):
        runner.run()
