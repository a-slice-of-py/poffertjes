"""P singleton interface for probability queries."""

from __future__ import annotations

from typing import Union, List

from poffertjes.variable import Variable
from poffertjes.expression import Expression, CompositeExpression
from poffertjes.exceptions import DataframeError, VariableError

# Import for type hints - will be resolved at runtime by query_builder
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from poffertjes.result import QueryResult


class P:
    """Singleton probability query interface.
    
    The P class provides the main user-facing API for probability queries.
    It follows the singleton pattern to ensure only one instance exists,
    allowing users to import and use `p` consistently throughout their code.
    
    Examples:
        >>> from poffertjes import p
        >>> from poffertjes.variable import VariableBuilder
        >>> 
        >>> vb = VariableBuilder.from_data(df)
        >>> x, y = vb.get_variables('x', 'y')
        >>> 
        >>> # Marginal probability
        >>> p(x)  # Returns distribution
        >>> p(x == 5)  # Returns scalar
        >>> 
        >>> # Conditional probability
        >>> p(x).given(y == 2)
        >>> p(x == 1).given(y == 2)
    """

    _instance = None

    def __new__(cls):
        """Create or return the singleton instance.
        
        This ensures that only one instance of P exists throughout the
        application lifetime. Multiple calls to P() or imports of `p`
        will all reference the same object.
        
        Returns:
            The singleton P instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(
        self,
        *args: Union["Variable", "Expression", "CompositeExpression"]
    ) -> "QueryResult":
        """Execute a probability query.
        
        This method is called when using the p instance like a function:
        p(x), p(x == 5), p(x, y), etc.
        
        Args:
            *args: Variables or expressions to query. Can be:
                - Single Variable: p(x) -> marginal distribution
                - Single Expression: p(x == 5) -> scalar probability
                - Multiple Variables: p(x, y) -> joint distribution
                - Multiple Expressions: p(x == 1, y == 2) -> scalar probability
                - Mix of Variables and Expressions
        
        Returns:
            QueryResult that can be a ScalarResult or DistributionResult,
            both of which support .given() for conditional probabilities.
        
        Raises:
            ValueError: If variables from different dataframes are mixed
            ValueError: If no arguments are provided
        
        Examples:
            >>> p(x)  # Marginal distribution of x
            >>> p(x == 5)  # P(X = 5)
            >>> p(x, y)  # Joint distribution of x and y
            >>> p(x == 1, y == 2)  # P(X=1, Y=2)
            >>> p(x).given(y == 2)  # P(X | Y=2)
        """
        if not args:
            raise VariableError("p() requires at least one argument")
        
        # Validate all variables come from same dataframe
        variables = self._extract_variables(args)
        if not variables:
            raise VariableError("No variables found in arguments")
        
        self._validate_same_dataframe(variables)
        
        # Import here to avoid circular imports
        # This will be implemented in task 5
        from poffertjes.query_builder import QueryBuilder
        
        # Build and execute query
        query = QueryBuilder(args)
        return query.execute()

    def _extract_variables(self, args) -> List["Variable"]:
        """Extract all variables from arguments.
        
        This method recursively extracts Variable objects from the arguments,
        which may be Variables, Expressions, or CompositeExpressions.
        
        Args:
            args: Tuple of arguments passed to __call__
        
        Returns:
            List of Variable objects found in the arguments
        """
        from poffertjes.variable import Variable
        from poffertjes.expression import Expression, CompositeExpression
        
        variables = []
        for arg in args:
            if isinstance(arg, Variable):
                variables.append(arg)
            elif isinstance(arg, Expression):
                variables.append(arg.variable)
            elif isinstance(arg, CompositeExpression):
                # Recursively extract from composite expressions
                for expr in arg.expressions:
                    if isinstance(expr, Expression):
                        variables.append(expr.variable)
                    elif isinstance(expr, CompositeExpression):
                        # Handle nested composite expressions
                        variables.extend(self._extract_variables((expr,)))
        return variables

    def _validate_same_dataframe(self, variables: List["Variable"]) -> None:
        """Ensure all variables come from the same dataframe.
        
        This is critical because probability calculations require all variables
        to be from the same dataset. If we allowed mixing variables from different
        dataframes, we'd be computing meaningless probabilities (e.g., P(X from df1, Y from df2)).
        
        We use id() to track dataframe identity because:
        1. It's fast (O(1) comparison)
        2. It correctly identifies the same dataframe instance
        3. It prevents accidental mixing even if dataframes have same content
        
        Args:
            variables: List of Variable objects to validate
        
        Raises:
            ValueError: If variables from different dataframes are detected
        
        Example of what we prevent:
            >>> df1 = pd.DataFrame({'x': [1, 2, 3]})
            >>> df2 = pd.DataFrame({'y': [4, 5, 6]})
            >>> vb1 = VariableBuilder.from_data(df1)
            >>> vb2 = VariableBuilder.from_data(df2)
            >>> x = vb1.get_variables('x')
            >>> y = vb2.get_variables('y')
            >>> p(x, y)  # This should raise an error!
        """
        if not variables:
            return
        
        first_id = variables[0].dataframe_id
        for var in variables[1:]:
            if var.dataframe_id != first_id:
                raise DataframeError(
                    f"Variables from different dataframes cannot be mixed: "
                    f"'{variables[0].name}' and '{var.name}' come from different dataframes"
                )


# Create singleton instance for export
p = P()
