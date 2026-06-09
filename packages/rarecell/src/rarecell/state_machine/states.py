"""IsolateState enum + valid transitions."""

from __future__ import annotations

from enum import Enum, auto


class IsolateState(Enum):
    S0_LOAD = auto()
    S1_INGEST = auto()
    S2_QC = auto()
    S2B_CLASS_GATE = auto()
    S3_CLUSTER = auto()
    S4_GATE1 = auto()
    S5_PURIFY = auto()
    S5_GATE2 = auto()
    S6_FINAL = auto()
    S6_GATE3 = auto()
    S7_REPORT = auto()
    S_ABORTED = auto()


_TRANSITIONS = {
    IsolateState.S0_LOAD: {IsolateState.S1_INGEST},
    IsolateState.S1_INGEST: {IsolateState.S2_QC},
    IsolateState.S2_QC: {IsolateState.S2B_CLASS_GATE},
    IsolateState.S2B_CLASS_GATE: {IsolateState.S3_CLUSTER},
    IsolateState.S3_CLUSTER: {IsolateState.S4_GATE1},
    IsolateState.S4_GATE1: {IsolateState.S5_PURIFY, IsolateState.S6_FINAL},
    IsolateState.S5_PURIFY: {IsolateState.S5_GATE2},
    IsolateState.S5_GATE2: {IsolateState.S6_FINAL},
    IsolateState.S6_FINAL: {IsolateState.S6_GATE3},
    IsolateState.S6_GATE3: {IsolateState.S7_REPORT},
    IsolateState.S7_REPORT: set(),
    IsolateState.S_ABORTED: set(),
}


def valid_transitions(state: IsolateState) -> set[IsolateState]:
    """Return the set of states reachable in one step."""
    base = set(_TRANSITIONS.get(state, set()))
    if state is not IsolateState.S_ABORTED:
        base.add(IsolateState.S_ABORTED)
    return base
