"""Poffertjes: Friendly interface to run probabilistic queries on dataframes.

Poffertjes provides a pythonic and intuitive interface for calculating probabilities
on dataframes using mathematical notation like p(x) or p(x).given(y). It supports
both Pandas and Polars dataframes through the Narwhals abstraction layer.

Key Features:
    - Mathematical notation: p(x), p(x == 5), p(x).given(y)
    - Dataframe agnostic: Works with Pandas and Polars
    - Efficient computation: Uses lazy evaluation and optimized operations
    - Type safe: Full type hints and IDE support
    - Comprehensive: Marginal, conditional, and joint probabilities

Basic Usage:
    >>> import pandas as pd
    >>> from poffertjes import p
    >>> from poffertjes.variable import VariableBuilder
    >>> 
    >>> # Create some data
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 1, 2, 1],
    ...     'y': ['A', 'B', 'A', 'B', 'A']
    ... })
    >>> 
    >>> # Extract variables
    >>> vb = VariableBuilder.from_data(df)
    >>> x, y = vb.get_variables('x', 'y')
    >>> 
    >>> # Calculate probabilities
    >>> p(x)  # Marginal distribution of x
    >>> p(x == 1)  # P(X = 1)
    >>> p(x).given(y == 'A')  # P(X | Y = 'A')
    >>> p(x, y)  # Joint distribution of x and y

For more examples and detailed documentation, see the individual module docstrings.
"""

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
