"""CELLxGENE Discover Curation API client (build-only; httpx is lazy-imported)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rarecell.errors import ReferenceBuildError

CURATION_BASE = "https://api.cellxgene.cziscience.com/curation/v1"


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ReferenceBuildError(
            "httpx is required for the build pipeline. Install with: "
            "uv sync --group build-reference"
        ) from e
    return httpx


@dataclass
class DiscoverDataset:
    title: str
    cell_count: int
    h5ad_url: str


def list_h5ad_datasets(collection_id: str) -> list[DiscoverDataset]:
    httpx = _httpx()
    resp = httpx.get(f"{CURATION_BASE}/collections/{collection_id}", timeout=60)
    resp.raise_for_status()
    out: list[DiscoverDataset] = []
    for ds in resp.json().get("datasets", []):
        h5ad = next((a["url"] for a in ds.get("assets", []) if a.get("filetype") == "H5AD"), None)
        if h5ad is None:
            continue
        out.append(
            DiscoverDataset(
                title=ds.get("title", ""),
                cell_count=int(ds.get("cell_count", 0)),
                h5ad_url=h5ad,
            )
        )
    return out


def download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    httpx = _httpx()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=None, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for block in resp.iter_bytes(chunk):
                fh.write(block)
