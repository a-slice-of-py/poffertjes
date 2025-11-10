"""Exception hierarchy for poffertjes.

This module defines the exception hierarchy used throughout the poffertjes library.
All exceptions inherit from PoffertjesError, which allows users to catch all
library-specific exceptions with a single except clause.

Examples:
    >>> try:
    ...     p(x, y)  # Variables from different dataframes
    ... except PoffertjesError as e:
    ...     print(f"Poffertjes error: {e}")
    
    >>> try:
    ...     vb.get_variables('nonexistent_column')
    ... except VariableError as e:
    ...     print(f"Variable error: {e}")
"""


class PoffertjesError(Exception):
    """Base exception for poffertjes.
    
    All poffertjes-specific exceptions inherit from this class, allowing
    users to catch all library errors with a single except clause.
    
    This exception should not be raised directly; instead, use one of
    the more specific subclasses.
    
    Examples:
        >>> try:
        ...     # Some poffertjes operation
        ...     result = p(x).given(y == 0)  # Might raise ProbabilityError
        ... except PoffertjesError as e:
        ...     print(f"A poffertjes error occurred: {e}")
    """
    pass


class DataframeError(PoffertjesError):
    """Errors related to dataframe operations.
    
    This exception is raised when there are issues with dataframe handling,
    such as empty dataframes, incompatible dataframes, or dataframe
    conversion problems.
    
    Examples:
        >>> # Empty dataframe
        >>> vb = VariableBuilder.from_data(empty_df)  # Raises DataframeError
        
        >>> # Mixed dataframes
        >>> p(x_from_df1, y_from_df2)  # Raises DataframeError
    """
    pass


class VariableError(PoffertjesError):
    """Errors related to variables.
    
    This exception is raised when there are issues with Variable objects,
    such as referencing non-existent columns, invalid variable operations,
    or improper variable usage.
    
    Examples:
        >>> # Non-existent column
        >>> vb.get_variables('missing_column')  # Raises VariableError
        
        >>> # Invalid conditioning
        >>> p(x == 1).given(y)  # Raises VariableError (scalar can't condition on variable)
    """
    pass


class ExpressionError(PoffertjesError):
    """Errors related to expressions.
    
    This exception is raised when there are issues with Expression objects,
    such as invalid operators, malformed expressions, or unsupported
    expression combinations.
    
    Examples:
        >>> # Invalid operator
        >>> Expression(x, "invalid_op", 5)  # Raises ExpressionError
        
        >>> # Empty list for isin
        >>> x.isin([])  # Raises ExpressionError
    """
    pass


class ProbabilityError(PoffertjesError):
    """Errors related to probability calculations.
    
    This exception is raised when there are issues with probability
    computations, such as zero probability conditioning events,
    invalid probability values, or computational errors.
    
    Examples:
        >>> # Zero probability conditioning
        >>> p(x).given(y == impossible_value)  # Raises ProbabilityError
        
        >>> # Joint probability with insufficient variables
        >>> calculator.calculate_joint([single_variable])  # Raises ProbabilityError
    """
    pass
