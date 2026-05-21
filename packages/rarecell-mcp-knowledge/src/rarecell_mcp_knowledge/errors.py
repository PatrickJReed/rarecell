"""Exception hierarchy for rarecell-mcp-knowledge."""


class KnowledgeError(Exception):
    """Base for all rarecell-mcp-knowledge exceptions."""


class BackendUnreachableError(KnowledgeError): ...


class CacheCorruptedError(KnowledgeError): ...


class InvalidQueryError(KnowledgeError): ...


class RateLimitedError(KnowledgeError): ...
