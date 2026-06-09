from __future__ import annotations

from pathlib import Path

from rarecell.cns.gate import apply_cns_class_gate
from rarecell.profile.schema import CNSTaxonomyConfig


def test_cns_gate_narrows_to_target(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=3)
    cfg = CNSTaxonomyConfig(
        enabled=True,
        target_node="Astrocyte",
        target_level="supercluster",
        reference_release=f"local:{tiny_bundle}",
        min_confidence=0.0,
    )
    narrowed, provenance = apply_cns_class_gate(query, cfg, cache_dir=tiny_bundle.parent)
    assert narrowed.n_obs < query.n_obs
    assert (narrowed.obs["supercluster_term"] == "Astrocyte").mean() >= 0.8
    assert provenance["target_node"] == "Astrocyte"


def test_cns_gate_disabled_is_noop(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=4)
    cfg = CNSTaxonomyConfig(enabled=False)
    narrowed, provenance = apply_cns_class_gate(query, cfg, cache_dir=tiny_bundle.parent)
    assert narrowed.n_obs == query.n_obs
    assert provenance == {"enabled": False}


def test_cns_gate_skips_on_unresolvable_bundle(tmp_path: Path, atlas_factory) -> None:
    query = atlas_factory(seed=5)
    cfg = CNSTaxonomyConfig(
        enabled=True,
        target_node="Astrocyte",
        target_level="supercluster",
        reference_release=f"local:{tmp_path / 'no-such-bundle'}",  # cannot resolve
        on_missing="skip",
    )
    narrowed, provenance = apply_cns_class_gate(query, cfg, cache_dir=tmp_path)
    # Graceful degradation: unchanged data, skipped provenance.
    assert narrowed.n_obs == query.n_obs
    assert provenance["enabled"] is True
    assert provenance["skipped"] is True
    assert "error" in provenance
