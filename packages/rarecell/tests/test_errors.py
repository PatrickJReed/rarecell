import pytest
from rarecell.errors import (
    CacheCorruptedError,
    IncompatibleSchemaError,
    InvalidProfileError,
    IsolationAbortedError,
    LLMBudgetExceededError,
    MCPUnreachableError,
    MissingRawCountsError,
    RareCellError,
    UnreviewedProfileError,
)


@pytest.mark.parametrize(
    "cls",
    [
        MissingRawCountsError,
        InvalidProfileError,
        UnreviewedProfileError,
        IncompatibleSchemaError,
        MCPUnreachableError,
        LLMBudgetExceededError,
        CacheCorruptedError,
        IsolationAbortedError,
    ],
)
def test_subclasses_of_base(cls):
    err = cls("test message")
    assert isinstance(err, RareCellError)
    assert str(err) == "test message"
