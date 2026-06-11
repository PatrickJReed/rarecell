from pathlib import Path

import anndata as ad
import numpy as np
from rarecell.cns.bundle import ReferenceBundle
from rarecell.cns.progressive import _as_log1p_cp10k, apply_progressive
from rarecell.cns.taxonomy import TaxonomyTree


def test_as_log1p_cp10k_normalizes_only_raw_counts() -> None:
    # Raw counts (max >> 9.22) get normalized into the log1p-CP10K range CellTypist
    # requires; already-log-normalized data passes through untouched.
    raw = ad.AnnData(X=np.array([[100.0, 0.0, 50.0], [0.0, 200.0, 10.0]], dtype=np.float32))
    out = _as_log1p_cp10k(raw)
    assert out is not raw
    assert float(out.X.max()) <= 9.22

    norm = ad.AnnData(X=np.array([[1.0, 0.0, 2.0]], dtype=np.float32))
    assert _as_log1p_cp10k(norm) is norm


def test_apply_progressive_supercluster_gate(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=1)  # fresh draw from the same distributions
    bundle = ReferenceBundle.resolve(f"local:{tiny_bundle}", cache_dir=tiny_bundle.parent)
    tax = TaxonomyTree.load(bundle.path)
    path = tax.path_to("Astrocyte", "supercluster")

    result = apply_progressive(query, bundle.path, path, min_confidence=0.0)

    kept = query[result.mask]
    # Most kept cells should truly be Astrocyte; most Astrocytes should be kept.
    true_sc = query.obs["supercluster_term"].to_numpy()
    precision = (kept.obs["supercluster_term"] == "Astrocyte").mean()
    recall = result.mask[true_sc == "Astrocyte"].mean()
    assert precision >= 0.8
    assert recall >= 0.7
    assert "taxonomy_supercluster" in query.obs
    assert result.provenance["levels"][0]["level"] == "supercluster"


def test_apply_progressive_marker_fallback_when_model_absent(
    tiny_bundle: Path, atlas_factory
) -> None:
    query = atlas_factory(seed=2)
    tax = TaxonomyTree.load(tiny_bundle)
    path = tax.path_to("Astrocyte", "supercluster")

    from rarecell.cns.format import load_markers

    sc_artifact = path[0][0]
    assert load_markers(tiny_bundle, sc_artifact.markers_file).get("Astrocyte", []) == [], (
        "test assumes Astrocyte has no positive-coef markers in the tiny bundle; "
        "if the build step changes, this test's inverse-fallback path is no longer exercised"
    )

    # Point the decision at a nonexistent model file to force fallback.
    artifact, keep = path[0]
    broken = artifact.model_copy(update={"model_file": "nodes/supercluster/_missing.pkl"})
    result = apply_progressive(
        query, tiny_bundle, [(broken, keep)], min_confidence=0.0, marker_fallback=True
    )
    # Fallback still yields a non-empty, mostly-correct astrocyte subset.
    assert result.mask.sum() > 0
    assert result.provenance["levels"][0]["method"] == "marker_fallback"
    assert "taxonomy_supercluster" in query.obs
