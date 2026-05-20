import pytest
from rarecell.profile.schema import (
    AutoPolicy,
    BatchCorrection,
    Citation,
    MarkerPanel,
    PurifyParams,
    QCParams,
    TargetCellProfile,
)


def test_minimal_valid_profile():
    from rarecell.profile.schema import ExpectedAbundance
    p = TargetCellProfile(
        profile_id="test", name="T cells, PBMC", description="pan T cells",
        target_lineage="lymphoid", tissue=["blood"],
        expected_abundance=ExpectedAbundance(min_fraction=0.05, max_fraction=0.6),
        positive_markers={
            "pan_t": MarkerPanel(genes=["CD3D", "CD3E"], threshold_z=1.0,
                                  citations=[Citation(source_id="pmid:1", source="europepmc")])
        },
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
        purify=PurifyParams(enabled=True, high_resolution=2.0, min_cluster_purity=0.7),
        batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
        auto_policy=AutoPolicy(),
    )
    assert p.schema_version == "1.0"
    assert p.frozen is False                # default
    assert p.human_reviewed is False        # default
    assert p.content_hash is None           # only set on freeze


@pytest.mark.skip(reason="needs preset from Task 6")
def test_yaml_roundtrip(tmp_path):
    src = TargetCellProfile.from_yaml_path("packages/rarecell/src/rarecell/profile/presets/t_cell_pbmc.yaml")
    out = tmp_path / "out.yaml"
    src.to_yaml_path(out)
    rebuilt = TargetCellProfile.from_yaml_path(out)
    assert rebuilt.model_dump() == src.model_dump()
