from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from rarecell.core.annotate import annotate_celltypist
from rarecell.profile.schema import (
    CellTypistRef,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)


def _profile_with_two_models():
    return TargetCellProfile(
        profile_id="t", name="t", description="d", target_lineage="lymphoid",
        tissue=["pbmc"],
        expected_abundance=ExpectedAbundance(min_fraction=0.1, max_fraction=0.6),
        positive_markers={"pan_t": MarkerPanel(
            genes=["CD3D"], threshold_z=1.0,
            citations=[Citation(source_id="pmid:1", source="europepmc")])},
        negative_markers={},
        reference_labels=ReferenceLabels(celltypist_models=[
            CellTypistRef(model="Immune_All_Low.pkl",
                          match_patterns=["T cell"], enabled=True),
            CellTypistRef(model="Disabled.pkl",
                          match_patterns=["x"], enabled=False),
        ]),
        qc=QCParams(min_genes_per_cell=200, max_pct_mt=10),
    )


def test_annotate_celltypist_skips_disabled_and_runs_enabled():
    a = ad.AnnData(X=np.zeros((10, 5)), var={"g": list("abcde")})
    a.var_names = list("abcde")
    with patch("rarecell.core.annotate._run_one_celltypist_model") as mock:
        mock.return_value = pd.DataFrame({
            "predicted_labels": ["T cell"] * 10,
            "majority_voting": ["T cell"] * 10,
            "conf_score": [0.9] * 10,
        }, index=a.obs_names)
        annotate_celltypist(a, _profile_with_two_models())
    mock.assert_called_once()
    args, _ = mock.call_args
    assert args[1] == "Immune_All_Low.pkl"
    assert "celltypist_Immune_All_Low_label" in a.obs.columns
    assert "celltypist_Immune_All_Low_label_majority" in a.obs.columns
    assert "celltypist_Immune_All_Low_conf" in a.obs.columns
