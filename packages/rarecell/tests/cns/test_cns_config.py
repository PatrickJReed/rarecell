from rarecell.profile.schema import CNSTaxonomyConfig, TargetCellProfile


def test_default_is_disabled() -> None:
    cfg = CNSTaxonomyConfig()
    assert cfg.enabled is False
    assert cfg.target_level == "supercluster"


def test_profile_has_cns_taxonomy_default(minimal_profile_kwargs: dict) -> None:
    p = TargetCellProfile(**minimal_profile_kwargs)
    assert p.cns_taxonomy.enabled is False


def test_cns_config_resolution_fields_default() -> None:
    cfg = CNSTaxonomyConfig()
    assert cfg.mode == "node"
    assert cfg.characterize_level == "cluster"
    assert cfg.rationale is None
    assert cfg.citations == []
