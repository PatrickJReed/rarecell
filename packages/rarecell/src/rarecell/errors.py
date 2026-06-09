"""Exception hierarchy for rarecell."""


class RareCellError(Exception):
    """Base class for all rarecell-specific exceptions."""


# --- User-input errors (recoverable) ---
class MissingRawCountsError(RareCellError): ...


class InvalidProfileError(RareCellError): ...


class UnreviewedProfileError(RareCellError):
    """Raised when a profile is set frozen=true without human_reviewed=true."""


class IncompatibleSchemaError(RareCellError): ...


# --- Runtime errors (have fallbacks) ---
class ReferenceBuildError(RareCellError):
    """Raised when building or reading a CNS reference bundle fails."""


class MCPUnreachableError(RareCellError): ...


class LLMBudgetExceededError(RareCellError): ...


class CacheCorruptedError(RareCellError): ...


# --- Catastrophic (partial-run saved) ---
class IsolationAbortedError(RareCellError): ...
