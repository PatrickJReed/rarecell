import anndata as ad
import numpy as np
import pandas as pd
from rarecell.core.evidence import (
    render_consensus_table,  # noqa: F401 — public API smoke import
    score_biccn_evidence,
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


def test_trinarize_calls_present_absent_unknown():
    """_trinarize returns P(theta >= f | k, n) as a float in [0, 1].

    High expressed fraction => near 1.0 ("present").
    Near-zero fraction      => near 0.0 ("absent").
    Mid fraction (~f=0.2)   => intermediate ("unknown" / ambiguous).
    """
    from rarecell.core.evidence import _trinarize

    present = _trinarize(95, 100)  # 95% expressing -> strongly present
    absent = _trinarize(2, 100)  # 2% expressing  -> strongly absent
    unknown = _trinarize(20, 100)  # 20% expressing -> at f threshold, ambiguous

    # All values are valid probabilities
    assert 0.0 <= absent <= 1.0
    assert 0.0 <= unknown <= 1.0
    assert 0.0 <= present <= 1.0

    # Ordering: present > unknown > absent
    assert present > unknown
    assert unknown > absent

    # Absolute semantics: present is near 1, absent is near 0
    assert present > 0.9, f"Expected high probability for 95/100 expressed, got {present}"
    assert absent < 0.01, f"Expected near-zero probability for 2/100 expressed, got {absent}"

    # Edge cases: all-zero is at least as absent as 2/100
    all_absent = _trinarize(0, 50)
    assert all_absent <= absent or all_absent < 0.01

    # Full expression saturates to present
    all_present = _trinarize(50, 50)
    assert all_present >= present or all_present > 0.9


def test_score_biccn_evidence_labels_and_threshold():
    """Cluster 0 strongly expresses T-cell genes -> TCELL; cluster 1 expresses none -> N/A."""
    # var_names include the three TCELL rule genes plus two non-rule genes.
    var_names = ["CD3D", "CD3E", "TRAC", "GFAP", "MS4A1"]
    n_per_cluster = 50

    # Build expression matrix: cluster 0 has high T-cell gene expression, cluster 1 has none.
    X = np.zeros((n_per_cluster * 2, len(var_names)))
    tcell_indices = [var_names.index(g) for g in ["CD3D", "CD3E", "TRAC"]]
    X[:n_per_cluster, tcell_indices] = 5.0  # cluster 0: strong T-cell signal

    a = ad.AnnData(
        X=X,
        obs={"leiden": ["0"] * n_per_cluster + ["1"] * n_per_cluster},
    )
    a.var_names = var_names

    score_biccn_evidence(a, _profile(), cluster_key="leiden")

    # biccn_label should be a Categorical in obs; biccn_evidence should be in uns.
    assert "biccn_label" in a.obs.columns
    assert hasattr(a.obs["biccn_label"], "cat"), "biccn_label should be Categorical"
    assert "biccn_evidence" in a.uns

    # uns structure: must have both keys; rules must have the expected columns.
    ev = a.uns["biccn_evidence"]
    assert set(ev.keys()) == {"trinaries", "rules"}
    rules: pd.DataFrame = ev["rules"]
    assert list(rules.columns) == ["cluster", "biccn_label", "biccn_score", "biccn_details"]

    rules_idx = rules.set_index("cluster")

    # cluster "0": should get TCELL (all three T-cell rule genes expressed at 5.0).
    label_0 = rules_idx.loc["0", "biccn_label"]
    score_0 = rules_idx.loc["0", "biccn_score"]
    assert label_0 == "TCELL", f"cluster 0: expected TCELL but got {label_0!r} (score={score_0})"

    # cluster "1": no rule genes expressed -> score below MIN_BICCN_SCORE -> N/A.
    label_1 = rules_idx.loc["1", "biccn_label"]
    score_1 = rules_idx.loc["1", "biccn_score"]
    assert label_1 == "N/A", f"cluster 1: expected N/A but got {label_1!r} (score={score_1})"
    assert score_1 == 0.0, f"cluster 1: expected score 0.0 but got {score_1}"
