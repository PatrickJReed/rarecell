"""Replay determinism — same input + profile + decisions yields byte-identical output."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
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
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

from tests.fixtures.make_synthetic import make_synthetic


def _profile_for_synthetic() -> TargetCellProfile:
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
        purify=PurifyParams(enabled=False),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        human_reviewed=True,
        reviewer="test@x",
    ).freeze()


def _hash_adata(path: Path) -> str:
    """Hash representative arrays of an h5ad (not raw file bytes — HDF5 metadata varies)."""
    a = ad.read_h5ad(path)
    h = hashlib.sha256()
    h.update(repr(tuple(a.shape)).encode())
    X = a.X.toarray() if hasattr(a.X, "toarray") else a.X
    h.update(X.tobytes())
    h.update(",".join(a.obs.columns).encode())
    h.update(",".join(map(str, a.obs.index)).encode())
    if "leiden" in a.obs.columns:
        h.update(",".join(a.obs["leiden"].astype(str)).encode())
    return h.hexdigest()


def test_replay_byte_deterministic(tmp_path: Path) -> None:
    profile = _profile_for_synthetic()
    adata = make_synthetic(seed=0)

    # First run — record decisions from the recommender.
    run1_dir = tmp_path / "run1"
    r1 = IsolateRunner(
        adata=adata.copy(),
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=run1_dir,
        auto_policy="recommendation",
    ).run()

    # Replay using the recorded decisions.
    run2_dir = tmp_path / "run2"
    IsolateRunner(
        adata=adata.copy(),
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=run2_dir,
        auto_policy="from_decisions",
        replay_decisions_path=r1.decisions_path,
    ).run()

    h1 = _hash_adata(run1_dir / "isolated.h5ad")
    h2 = _hash_adata(run2_dir / "isolated.h5ad")
    assert h1 == h2, "Replay output differs from first run; non-determinism in the pipeline"
