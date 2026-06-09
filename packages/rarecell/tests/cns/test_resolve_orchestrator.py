"""Orchestrator integration test: resolve_cns_target -> CNSTaxonomyConfig.

The tiny_bundle fixture (module-scoped) is provided by the cns conftest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from rarecell.agent.resolve import resolve_cns_target
from rarecell.profile.schema import CNSTaxonomyConfig


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def messages_create(self, *, messages: list, tools: object = None) -> dict:  # type: ignore[type-arg]
        return {
            "content": [{"type": "text", "text": "```json\n" + json.dumps(self._payload) + "\n```"}]
        }


def test_resolve_cns_target_disables_on_low_confidence(tiny_bundle: Path) -> None:
    client = _FakeClient(
        {
            "mode": "program",
            "gate_node": "Astrocyte",
            "gate_level": "supercluster",
            "characterize_level": "cluster",
            "rationale": "unsure",
            "citations": [],
            "confidence": 0.1,
        }
    )

    class _P:
        name = "x"
        target_lineage = "astrocyte"
        description = "d"
        tissue: ClassVar[list[str]] = ["brain"]
        positive_markers: ClassVar[dict] = {}

    cfg = resolve_cns_target(
        _P(),
        bundle_dir=tiny_bundle,
        reference_release=f"local:{tiny_bundle}",
        client=client,
        min_resolution_confidence=0.5,
    )
    assert cfg.enabled is False
    assert cfg.rationale == "unsure"


def test_resolve_cns_target_populates_config(tiny_bundle: Path) -> None:
    client = _FakeClient(
        {
            "mode": "program",
            "gate_node": "Astrocyte",
            "gate_level": "supercluster",
            "characterize_level": "cluster",
            "rationale": "program",
            "citations": [],
            "confidence": 0.7,
        }
    )

    class _P:
        name = "x"
        target_lineage = "astrocyte"
        description = "d"
        tissue: ClassVar[list[str]] = ["brain"]
        positive_markers: ClassVar[dict] = {"astro": type("M", (), {"genes": ["AQP4", "GFAP"]})()}

    cfg = resolve_cns_target(
        _P(), bundle_dir=tiny_bundle, reference_release=f"local:{tiny_bundle}", client=client
    )
    assert isinstance(cfg, CNSTaxonomyConfig)
    assert cfg.enabled and cfg.mode == "program"
    assert cfg.target_node == "Astrocyte" and cfg.target_level == "supercluster"
    assert cfg.reference_release == f"local:{tiny_bundle}"
    assert cfg.rationale == "program"
    assert cfg.citations == []
    assert cfg.characterize_level == "cluster"
