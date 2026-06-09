from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.build_cns_reference import annotate_abc


def _membership() -> pd.DataFrame:
    rows = []
    # (supercluster, cluster, subcluster, neurotransmitter|None). None = no NT row
    # (non-neuronal clusters like astrocytes have no neurotransmitter in ABC).
    spec = {
        0: ("MGE interneuron", "MGE_259", "MGE_259_0", "GABA"),
        1: ("MGE interneuron", "MGE_259", "MGE_259_1", "GABA"),
        2: ("MGE interneuron", "MGE_260", "MGE_260_2", "GABA VGLUT3"),
        3: ("Astrocyte", "Astro_12", "Astro_12_3", None),
    }
    for alias, (sc, cl, sub, nt) in spec.items():
        terms = [("supercluster", sc), ("cluster", cl), ("subcluster", sub)]
        if nt is not None:
            terms.append(("neurotransmitter", nt))
        for term_set, name in terms:
            rows.append(
                {
                    "cluster_alias": alias,
                    "cluster_annotation_term_set_name": term_set,
                    "cluster_annotation_term_name": name,
                }
            )
    return pd.DataFrame(rows)


def test_build_annotation_map_pivots_per_subcluster() -> None:
    amap = annotate_abc.build_annotation_map(_membership())
    assert amap["0"]["cluster"] == "MGE_259"
    assert amap["0"]["supercluster"] == "MGE interneuron"
    assert amap["0"]["neurotransmitter"] == "GABA"
    assert amap["3"]["cluster"] == "Astro_12"
    # Non-neuronal cluster has no neurotransmitter -> "" (not the string "nan").
    assert amap["3"]["neurotransmitter"] == ""


def test_annotate_atlas_adds_columns() -> None:
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    a.obs = pd.DataFrame({"subcluster_id": ["0", "1", "3"]}, index=[f"c{i}" for i in range(3)])
    amap = annotate_abc.build_annotation_map(_membership())
    annotate_abc.annotate_atlas(a, amap)
    assert list(a.obs[annotate_abc.CLUSTER_NAME_COL]) == ["MGE_259", "MGE_259", "Astro_12"]
    # Astro_12 is non-neuronal -> empty neurotransmitter (not "nan").
    assert list(a.obs[annotate_abc.NEUROTRANSMITTER_COL]) == ["GABA", "GABA", ""]
    nt = annotate_abc.cluster_neurotransmitters(a)
    assert nt == {"MGE_259": "GABA", "Astro_12": ""}


def test_annotate_atlas_passes_through_unknown_subcluster() -> None:
    a = ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))
    a.obs = pd.DataFrame({"subcluster_id": ["9999"]}, index=["c0"])
    annotate_abc.annotate_atlas(a, annotate_abc.build_annotation_map(_membership()))
    # Unknown subcluster falls back to its own id for the name, empty neurotransmitter.
    assert a.obs[annotate_abc.CLUSTER_NAME_COL].iloc[0] == "9999"
    assert a.obs[annotate_abc.NEUROTRANSMITTER_COL].iloc[0] == ""


def test_load_membership_uses_cache(tmp_path: Path) -> None:
    # Pre-place the cache file so no network download happens.
    (tmp_path / "abc_whb_membership.csv").write_text(_membership().to_csv(index=False))
    df = annotate_abc.load_membership(tmp_path)
    assert set(df["cluster_annotation_term_set_name"]) >= {"supercluster", "cluster"}


def test_annotations_sidecar_round_trip(tmp_path: Path) -> None:
    from rarecell.cns import format as fmt

    assert fmt.load_annotations(tmp_path) == {}  # absent -> empty
    ann = {"MGE_259": {"neurotransmitter": "GABA"}}
    fmt.write_annotations(tmp_path, ann)
    assert fmt.load_annotations(tmp_path) == ann
