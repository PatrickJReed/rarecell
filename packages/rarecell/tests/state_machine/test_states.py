from itertools import pairwise

from rarecell.state_machine.states import IsolateState, valid_transitions


def test_normal_progression():
    seq = [
        IsolateState.S0_LOAD,
        IsolateState.S1_INGEST,
        IsolateState.S2_QC,
        IsolateState.S2B_CLASS_GATE,
        IsolateState.S3_CLUSTER,
        IsolateState.S4_GATE1,
        IsolateState.S5_PURIFY,
        IsolateState.S5_GATE2,
        IsolateState.S6_FINAL,
        IsolateState.S6_GATE3,
        IsolateState.S7_REPORT,
    ]
    for a, b in pairwise(seq):
        assert b in valid_transitions(a)


def test_skip_purify_path():
    assert IsolateState.S6_FINAL in valid_transitions(IsolateState.S4_GATE1)


def test_abort_from_anywhere():
    for s in IsolateState:
        if s is not IsolateState.S_ABORTED:
            assert IsolateState.S_ABORTED in valid_transitions(s)
