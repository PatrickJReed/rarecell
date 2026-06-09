from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import annotate_s3


def _s3_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [0, 1, 52],
            "class_auto": ["BCELL", "TCELL", "ASTRO"],
            "subtype_auto": [np.nan, np.nan, np.nan],
            "neuropeptide_auto": [np.nan, np.nan, np.nan],
            "top_enriched_genes": ["IGHM, MS4A1, CD79A", "CD3D, TRAC", "AQP4, GFAP, SLC1A2"],
        }
    )


def test_build_s3_map_parses_class_and_markers() -> None:
    m = annotate_s3.build_s3_map(_s3_df())
    assert m["0"]["class"] == "BCELL"
    assert m["0"]["markers"] == ["IGHM", "MS4A1", "CD79A"]
    assert m["52"]["class"] == "ASTRO"
    assert m["52"]["markers"] == ["AQP4", "GFAP", "SLC1A2"]
    assert m["0"]["subtype"] == ""  # NaN -> ""


def test_cluster_annotations_maps_name_via_cluster_id() -> None:
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    a.obs = pd.DataFrame(
        {"cluster_id": ["0", "1", "52"], "cluster_name": ["Bcell_0", "Tcell_1", "Astro_52"]},
        index=[f"c{i}" for i in range(3)],
    )
    out = annotate_s3.cluster_annotations(a, annotate_s3.build_s3_map(_s3_df()))
    assert out["Astro_52"]["class"] == "ASTRO"
    assert out["Astro_52"]["markers"] == ["AQP4", "GFAP", "SLC1A2"]
    assert out["Bcell_0"]["class"] == "BCELL"


def test_cluster_annotations_falls_back_to_cluster_id_when_no_name() -> None:
    a = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
    a.obs = pd.DataFrame({"cluster_id": ["52"]}, index=["c0"])  # no cluster_name
    out = annotate_s3.cluster_annotations(a, annotate_s3.build_s3_map(_s3_df()))
    assert out["52"]["class"] == "ASTRO"


def test_build_s3_map_includes_regions() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [52],
            "class_auto": ["ASTRO"],
            "subtype_auto": [np.nan],
            "neuropeptide_auto": [np.nan],
            "top_enriched_genes": ["AQP4, GFAP"],
            "top_regions": ["Cerebral cortex: 40%, Thalamus: 20%"],
        }
    )
    m = annotate_s3.build_s3_map(df)
    assert m["52"]["regions"] == ["Cerebral cortex", "Thalamus"]


def test_vendored_table_s3_loads() -> None:
    df = annotate_s3.load_table_s3()
    assert len(df) == 461
    assert {"cluster_id", "class_auto", "top_enriched_genes"} <= set(df.columns)
    m = annotate_s3.build_s3_map(df)
    # Every cluster has a curated marker panel.
    assert all(len(v["markers"]) > 0 for v in m.values())
    assert "regions" in next(iter(m.values()))
