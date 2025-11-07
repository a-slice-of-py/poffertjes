"""Exception hierarchy for poffertjes."""


class PoffertjesError(Exception):
    """Base exception for poffertjes."""
    pass


class DataframeError(PoffertjesError):
    """Errors related to dataframe operations."""
    pass


class VariableError(PoffertjesError):
    """Errors related to variables."""
    pass


class ExpressionError(PoffertjesError):
    """Errors related to expressions."""
    pass


class ProbabilityError(PoffertjesError):
    """Errors related to probability calculations."""
    pass
