from __future__ import annotations

from pathlib import Path

from rarecell.cns.characterize import characterize


def test_characterize_labels_and_summarizes(tiny_bundle: Path, atlas_factory) -> None:
    # Use only Astrocyte cells as the "isolated" population.
    query = atlas_factory(seed=7)
    isolated = query[query.obs["supercluster_term"] == "Astrocyte"].copy()
    result = characterize(isolated, tiny_bundle, level="cluster", parent_node="Astrocyte")

    assert len(result.per_cell_labels) == isolated.n_obs
    assert list(result.per_cell_labels.index) == list(isolated.obs_names)
    assert result.summary  # non-empty
    fractions = sum(row["fraction"] for row in result.summary)
    assert 0.99 <= fractions <= 1.01
    # rows carry annotation fields
    assert "class" in result.summary[0] and "fraction" in result.summary[0]
