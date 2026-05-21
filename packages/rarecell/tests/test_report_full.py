import json
from pathlib import Path

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


def _profile():
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


def test_full_report_written(tmp_path: Path):
    profile = _profile()
    runner = IsolateRunner(
        adata=make_synthetic(seed=0),
        profile=profile,
        recommender=BasicRecommender(profile),
        out_dir=tmp_path,
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
