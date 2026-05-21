import pytest
from rarecell_mcp_knowledge.errors import (
    BackendUnreachableError,
    CacheCorruptedError,
    InvalidQueryError,
    KnowledgeError,
    RateLimitedError,
)


@pytest.mark.parametrize(
    "cls",
    [
        BackendUnreachableError,
        CacheCorruptedError,
        InvalidQueryError,
        RateLimitedError,
    ],
)
def test_subclass_of_base(cls):
    err = cls("msg")
    assert isinstance(err, KnowledgeError)
    assert str(err) == "msg"
