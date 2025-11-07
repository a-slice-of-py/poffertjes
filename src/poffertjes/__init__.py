"""Poffertjes: Friendly interface to run probabilistic queries on dataframes."""

from poffertjes.p_interface import p
from poffertjes.exceptions import (
    PoffertjesError,
    DataframeError,
    VariableError,
    ExpressionError,
    ProbabilityError,
)

__version__ = "0.1.0"

# Main exports
__all__ = [
    "p",
    "PoffertjesError",
    "DataframeError", 
    "VariableError",
    "ExpressionError",
    "ProbabilityError",
]
