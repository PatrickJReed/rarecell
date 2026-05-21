import anndata as ad
import numpy as np
import pandas as pd
from rarecell.core.evidence import (
    render_consensus_table,  # noqa: F401 — public API smoke import
    score_biccn_evidence,  # noqa: F401 — public API smoke import
    score_evidence,
    select_clusters,
)
from rarecell.profile.schema import (
    BICCNRules,
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
        tissue=["brain"],
        expected_abundance=ExpectedAbundance(min_fraction=0.001, max_fraction=0.05),
        positive_markers={
            "pan_t": MarkerPanel(
                genes=["CD3D"],
                threshold_z=1.0,
                citations=[Citation(source_id="pmid:1", source="europepmc")],
            )
        },
        negative_markers={},
        biccn_rules=BICCNRules(
            enabled=True, class_filter=["TCELL"], subclass_filter=["TCELL", "BCELL"]
        ),
        qc=QCParams(min_genes_per_cell=150, max_pct_mt=10),
    )


def _adata_with_clusters():
    n = 200
    a = ad.AnnData(
        X=np.zeros((n, 5)),
        obs={
            "leiden": ["0"] * 100 + ["1"] * 100,
            "score_pan_t": [2.0] * 100 + [0.1] * 100,
            "pass_pan_t": [True] * 100 + [False] * 100,
            "is_contaminant": [False] * 200,
        },
    )
    a.var_names = ["CD3D", "GFAP", "MS4A1", "AQP4", "RBFOX3"]
    return a


def test_score_evidence_returns_one_row_per_cluster():
    a = _adata_with_clusters()
    table = score_evidence(a, _profile(), cluster_key="leiden")
    assert isinstance(table, pd.DataFrame)
    assert set(table["cluster"].astype(str)) == {"0", "1"}
    score_0 = float(table.set_index("cluster").loc["0", "score_pan_t_mean"])
    score_1 = float(table.set_index("cluster").loc["1", "score_pan_t_mean"])
    assert score_0 > score_1


def test_select_clusters_routes_by_recommendation():
    a = _adata_with_clusters()
    table = score_evidence(a, _profile(), cluster_key="leiden")
    table["recommendation"] = ["keep", "drop"]
    keep = select_clusters(table, "keep")
    drop = select_clusters(table, "drop")
    assert set(keep) == {"0"}
    assert set(drop) == {"1"}
