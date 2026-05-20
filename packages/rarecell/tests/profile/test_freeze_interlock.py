import pytest
from rarecell.errors import UnreviewedProfileError
from rarecell.profile.schema import (
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    QCParams,
    TargetCellProfile,
)


def _minimal_profile_kwargs():
    return dict(
        profile_id="test", name="T", description="d", target_lineage="lymphoid",
        tissue=["blood"],
        expected_abundance=ExpectedAbundance(min_fraction=0.01, max_fraction=0.5),
        positive_markers={"pan": MarkerPanel(genes=["CD3D"], threshold_z=1.0,
                                              citations=[Citation(source_id="pmid:1",
                                                                  source="europepmc")])},
        negative_markers={},
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_frozen_without_review_raises():
    with pytest.raises(UnreviewedProfileError, match="human_reviewed"):
        TargetCellProfile(**_minimal_profile_kwargs(),
                          frozen=True, human_reviewed=False)


def test_frozen_with_review_succeeds_and_has_hash():
    p = TargetCellProfile(**_minimal_profile_kwargs(),
                          human_reviewed=True, reviewer="r@x")
    assert p.content_hash is None     # not frozen yet, no hash
    frozen = p.freeze()
    assert frozen.frozen is True
    assert frozen.content_hash is not None
    assert frozen.content_hash.startswith("sha256:")


def test_freezing_unreviewed_raises():
    p = TargetCellProfile(**_minimal_profile_kwargs(), human_reviewed=False)
    with pytest.raises(UnreviewedProfileError):
        p.freeze()
