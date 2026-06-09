from __future__ import annotations

from pathlib import Path

import pytest
from rarecell.cns import format as fmt
from rarecell.errors import ReferenceBuildError


def test_load_model_returns_usable_model(tiny_bundle: Path) -> None:
    manifest = fmt.load_manifest(tiny_bundle)
    sc_dec = next(d for d in manifest.decisions if d.level == "supercluster")
    model = fmt.load_model(tiny_bundle, sc_dec)
    # A celltypist Model exposes the trained classes.
    assert set(sc_dec.classes).issubset({str(c) for c in model.classifier.classes_})


def test_load_model_detects_sha_mismatch(tiny_bundle: Path) -> None:
    manifest = fmt.load_manifest(tiny_bundle)
    sc_dec = manifest.decisions[0]
    tampered = sc_dec.model_copy(update={"model_sha256": "0" * 64})
    with pytest.raises(ReferenceBuildError):
        fmt.load_model(tiny_bundle, tampered)
