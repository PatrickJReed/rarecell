"""Resolve a CNS reference bundle from a local path or a GitHub release tag.

Runtime download uses the standard library (urllib/tarfile) so the shipped
package needs no HTTP dependency.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rarecell.cns.format import BundleManifest, load_manifest
from rarecell.errors import ReferenceBuildError
from rarecell.logging import get_logger

log = get_logger("rarecell.cns.bundle")

# GitHub release asset URL template. A release tagged `cns-<release>` carries
# an asset `cns-reference-<release>.tar.gz` extracting to `cns-reference-<release>/`.
RELEASE_URL = (
    "https://github.com/PatrickJReed/rarecell/releases/download/"
    "cns-{release}/cns-reference-{release}.tar.gz"
)


@dataclass
class ReferenceBundle:
    path: Path

    @property
    def manifest(self) -> BundleManifest:
        return load_manifest(self.path)

    @classmethod
    def resolve(cls, reference_release: str, *, cache_dir: Path) -> ReferenceBundle:
        """Resolve to a bundle directory.

        ``local:<path>`` uses ``<path>`` directly. Otherwise treat
        ``reference_release`` as a release tag and download+cache the asset.
        """
        if reference_release.startswith("local:"):
            path = Path(reference_release[len("local:") :])
            if not (path / "manifest.json").exists():
                raise ReferenceBuildError(f"No bundle (manifest.json) at {path}")
            return cls(path)

        cache_dir = Path(cache_dir)
        dest = cache_dir / f"cns-reference-{reference_release}"
        if not (dest / "manifest.json").exists():
            cls._download_and_extract(reference_release, cache_dir)
        if not (dest / "manifest.json").exists():
            raise ReferenceBuildError(
                f"Bundle for release {reference_release!r} not found after download"
            )
        return cls(dest)

    @staticmethod
    def _download_and_extract(release: str, cache_dir: Path) -> None:  # pragma: no cover - network
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = RELEASE_URL.format(release=release)
        tar_path = cache_dir / f"cns-reference-{release}.tar.gz"
        log.info("bundle.download", release=release, url=url)
        urllib.request.urlretrieve(url, tar_path)  # trusted https github asset
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(cache_dir, filter="data")
