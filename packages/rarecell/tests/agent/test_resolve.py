from __future__ import annotations

import json
from typing import ClassVar

import pytest
from rarecell.agent.resolve import TargetResolution, resolve_target
from rarecell.cns.retrieve import NodeDescriptor, NodeMatch


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def messages_create(self, *, messages, tools=None):  # type: ignore[no-untyped-def]
        return {
            "content": [{"type": "text", "text": "```json\n" + json.dumps(self._payload) + "\n```"}]
        }


def _candidates() -> list[NodeMatch]:
    return [
        NodeMatch(
            NodeDescriptor("Astrocyte", "supercluster", None, "Astrocyte", ["AQP4", "GFAP"]),
            0.6,
            {"marker_overlap": 0.2, "class_match": 1.0, "region_match": 0.0},
        ),
        NodeMatch(
            NodeDescriptor("Astro_52", "cluster", "Astrocyte", "ASTRO", ["AQP4"]),
            0.4,
            {"marker_overlap": 0.1, "class_match": 1.0, "region_match": 0.0},
        ),
    ]


def test_resolve_returns_validated_program_decision() -> None:
    client = _FakeClient(
        {
            "mode": "program",
            "gate_node": "Astrocyte",
            "gate_level": "supercluster",
            "characterize_level": "cluster",
            "rationale": "SNAP is an astrocyte gene program, not a discrete cluster.",
            "citations": ["pmid:38448582"],
            "confidence": 0.8,
        }
    )

    class _P:  # minimal profile stand-in (resolve only reads a few attrs)
        name = "SNAP astrocytes"
        target_lineage = "astrocyte"
        description = "SNAP-expressing astrocytes from DLPFC"
        positive_markers: ClassVar[dict] = {}

    res = resolve_target(_P(), candidates=_candidates(), client=client)
    assert isinstance(res, TargetResolution)
    assert res.mode == "program"
    assert res.gate_node == "Astrocyte"
    assert res.characterize_level == "cluster"


class _NoTextClient:
    def messages_create(self, *, messages, tools=None):  # type: ignore[no-untyped-def]
        return {"content": []}


class _NoJsonClient:
    def messages_create(self, *, messages, tools=None):  # type: ignore[no-untyped-def]
        return {"content": [{"type": "text", "text": "no json here"}]}


def test_resolve_raises_on_no_text() -> None:
    with pytest.raises(ValueError):
        resolve_target(object(), candidates=_candidates(), client=_NoTextClient())


def test_resolve_raises_on_no_json() -> None:
    with pytest.raises(ValueError):
        resolve_target(object(), candidates=_candidates(), client=_NoJsonClient())


def test_resolve_returns_validated_node_decision() -> None:
    client = _FakeClient(
        {
            "mode": "node",
            "gate_node": "MGE_259",
            "gate_level": "cluster",
            "characterize_level": "subcluster",
            "rationale": "discrete subtype",
            "citations": [],
            "confidence": 0.9,
        }
    )

    class _P:
        name = "chandelier"
        target_lineage = "neuron"
        description = "PVALB chandelier interneurons"
        positive_markers: ClassVar[dict] = {}

    res = resolve_target(_P(), candidates=_candidates(), client=client)
    assert res.mode == "node" and res.gate_level == "cluster"
    assert res.characterize_level == "subcluster"
