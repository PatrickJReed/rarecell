from __future__ import annotations

import pytest

from scripts.build_cns_reference import discover

WHB_COLLECTION = "283d65eb-dd53-496d-adb7-7570c7caa443"


@pytest.mark.integration
def test_discover_lists_real_whb_h5ads() -> None:
    datasets = discover.list_h5ad_datasets(WHB_COLLECTION)
    assert len(datasets) >= 20
    titles = [d.title for d in datasets]
    assert any("Astrocyte" in t for t in titles)
    assert all(d.h5ad_url.endswith(".h5ad") for d in datasets)
