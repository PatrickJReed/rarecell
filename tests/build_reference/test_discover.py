from __future__ import annotations

import httpx
import pytest
import respx

from scripts.build_cns_reference import discover

COLLECTION_ID = "283d65eb-dd53-496d-adb7-7570c7caa443"


@respx.mock
def test_list_h5ad_datasets_parses_assets() -> None:
    payload = {
        "datasets": [
            {
                "title": "Supercluster: Astrocyte",
                "cell_count": 155025,
                "assets": [
                    {"filetype": "H5AD", "url": "https://x/astro.h5ad", "filesize": 100},
                    {"filetype": "RDS", "url": "https://x/astro.rds", "filesize": 100},
                ],
            },
            {
                "title": "No H5AD dataset",
                "cell_count": 10,
                "assets": [{"filetype": "RDS", "url": "https://x/only.rds", "filesize": 1}],
            },
        ]
    }
    respx.get(f"{discover.CURATION_BASE}/collections/{COLLECTION_ID}").mock(
        return_value=httpx.Response(200, json=payload)
    )
    datasets = discover.list_h5ad_datasets(COLLECTION_ID)
    assert len(datasets) == 1
    assert datasets[0].title == "Supercluster: Astrocyte"
    assert datasets[0].h5ad_url == "https://x/astro.h5ad"
    assert datasets[0].cell_count == 155025


@respx.mock
def test_download_streams_to_file(tmp_path: object) -> None:
    respx.get("https://x/astro.h5ad").mock(return_value=httpx.Response(200, content=b"H5ADBYTES"))
    from pathlib import Path

    dest = Path(str(tmp_path)) / "astro.h5ad"
    discover.download("https://x/astro.h5ad", dest)
    assert dest.read_bytes() == b"H5ADBYTES"
    # Atomic: no leftover temp file after a successful download.
    assert not dest.with_name(dest.name + ".part").exists()


@respx.mock
def test_download_is_atomic_on_failure(tmp_path: object) -> None:
    from pathlib import Path

    # A failed download must leave NO file at dest (a truncated cache entry
    # would be treated as complete on the next run and crash the HDF5 reader).
    respx.get("https://x/broken.h5ad").mock(return_value=httpx.Response(500))
    dest = Path(str(tmp_path)) / "broken.h5ad"
    with pytest.raises(httpx.HTTPStatusError):
        discover.download("https://x/broken.h5ad", dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
