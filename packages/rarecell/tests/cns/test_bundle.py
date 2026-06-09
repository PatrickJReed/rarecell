from __future__ import annotations

from pathlib import Path

import pytest
from rarecell.cns.bundle import ReferenceBundle
from rarecell.errors import ReferenceBuildError


def test_resolve_local_path(tiny_bundle: Path, tmp_path: Path) -> None:
    rb = ReferenceBundle.resolve(f"local:{tiny_bundle}", cache_dir=tmp_path)
    assert rb.path == tiny_bundle
    # manifest is loadable through the bundle
    assert rb.manifest.biccn_release == "WHB-test"


def test_resolve_missing_local_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ReferenceBuildError):
        ReferenceBundle.resolve(f"local:{tmp_path / 'nope'}", cache_dir=tmp_path)
