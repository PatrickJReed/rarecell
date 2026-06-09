from pathlib import Path

from rarecell.profile.schema import CNSTaxonomyConfig
from rarecell.state_machine.isolate import characterize_isolated


def test_characterize_isolated_stores_summary(tiny_bundle: Path, atlas_factory) -> None:
    query = atlas_factory(seed=8)
    isolated = query[query.obs["supercluster_term"] == "Astrocyte"].copy()
    cfg = CNSTaxonomyConfig(
        enabled=True,
        target_node="Astrocyte",
        target_level="supercluster",
        mode="program",
        characterize_level="cluster",
        reference_release=f"local:{tiny_bundle}",
    )
    characterize_isolated(isolated, cfg, cache_dir=tiny_bundle.parent)
    assert "cns_characterization" in isolated.uns
    assert isolated.uns["cns_characterization"]["summary"]
    assert "reference_cluster" in isolated.obs.columns


def test_characterize_isolated_noop_when_disabled(tiny_bundle: Path, atlas_factory) -> None:
    isolated = atlas_factory(seed=9)
    characterize_isolated(isolated, CNSTaxonomyConfig(enabled=False), cache_dir=tiny_bundle.parent)
    assert "cns_characterization" not in isolated.uns
