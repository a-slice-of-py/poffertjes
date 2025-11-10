"""QueryBuilder for constructing and executing probability queries."""

from __future__ import annotations

from typing import Any, List, Tuple, Union, TYPE_CHECKING

from poffertjes.exceptions import VariableError, ExpressionError

if TYPE_CHECKING:
    from poffertjes.variable import Variable
    from poffertjes.expression import Expression, CompositeExpression


class QueryBuilder:
    """Builds and executes probability queries.

    The QueryBuilder is responsible for parsing arguments passed to p(...) calls,
    determining whether the query should return a scalar probability or a
    probability distribution, and coordinating the execution of the query.

    Examples:
        >>> # Distribution query - returns probability distribution
        >>> p(x)  # QueryBuilder parses [Variable(x)]

        >>> # Scalar query - returns scalar probability
        >>> p(x == 5)  # QueryBuilder parses [Expression(x == 5)]

        >>> # Joint distribution query
        >>> p(x, y)  # QueryBuilder parses [Variable(x), Variable(y)]

        >>> # Joint scalar query
        >>> p(x == 1, y == 2)  # QueryBuilder parses [Expression(x == 1), Expression(y == 2)]
    """

    def __init__(
        self, args: Tuple[Union["Variable", "Expression", "CompositeExpression"], ...]
    ) -> None:
        """Initialize QueryBuilder with arguments from p(...) call.

        Args:
            args: Tuple of Variables, Expressions, or CompositeExpressions passed to p(...)
        """
        self.args = args
        self.variables: List["Variable"] = []
        self.expressions: List[Union["Expression", "CompositeExpression"]] = []
        self._parse_args()

    def _parse_args(self) -> None:
        """Parse arguments into variables and expressions.

        This method categorizes the arguments into:
        - variables: Variable objects (for distribution queries like p(x))
        - expressions: Expression/CompositeExpression objects (for scalar queries like p(x == 5))

        It also extracts variables from expressions to build the complete list of
        variables involved in the query for dataframe validation.

        Examples:
            >>> # p(x) -> variables=[x], expressions=[]
            >>> # p(x == 5) -> variables=[x], expressions=[x == 5]
            >>> # p(x, y) -> variables=[x, y], expressions=[]
            >>> # p(x == 1, y == 2) -> variables=[x, y], expressions=[x == 1, y == 2]
        """
        from poffertjes.variable import Variable
        from poffertjes.expression import Expression, CompositeExpression

        for arg in self.args:
            if isinstance(arg, Variable):
                # Direct variable reference - this indicates a distribution query
                self._add_variable_if_not_present(arg)
            elif isinstance(arg, Expression):
                # Expression on a variable - this indicates a scalar query
                self.expressions.append(arg)
                # Also track the variable for dataframe validation
                self._add_variable_if_not_present(arg.variable)
            elif isinstance(arg, CompositeExpression):
                # Composite expression - this indicates a scalar query
                self.expressions.append(arg)
                # Extract variables from all sub-expressions
                self._extract_variables_from_composite(arg)
            else:
                raise ExpressionError(
                    f"Invalid argument type: {type(arg)}. Expected Variable, Expression, or CompositeExpression."
                )

    def _add_variable_if_not_present(self, variable: "Variable") -> None:
        """Add variable to variables list if not already present.

        This method checks for variable presence by comparing the variable name
        and dataframe ID to avoid duplicates.

        Args:
            variable: Variable to add to the list
        """
        # Check if variable is already in the list by comparing name and dataframe ID
        for existing_var in self.variables:
            if (
                existing_var.name == variable.name
                and existing_var.dataframe_id == variable.dataframe_id
            ):
                return  # Variable already present

        # Variable not found, add it
        self.variables.append(variable)

    def _extract_variables_from_composite(
        self, composite_expr: "CompositeExpression"
    ) -> None:
        """Extract variables from a composite expression recursively.

        This method traverses composite expressions to find all variables involved,
        ensuring they are added to the variables list for dataframe validation.

        Args:
            composite_expr: CompositeExpression to extract variables from
        """
        from poffertjes.expression import Expression, CompositeExpression

        for expr in composite_expr.expressions:
            if isinstance(expr, Expression):
                self._add_variable_if_not_present(expr.variable)
            elif isinstance(expr, CompositeExpression):
                # Recursively extract from nested composite expressions
                self._extract_variables_from_composite(expr)

    def execute(self) -> Any:
        """Execute the query and return appropriate result.

        This method determines the query type and delegates to the appropriate
        calculation method:
        - If expressions are present: scalar query -> ScalarResult
        - If only variables: distribution query -> DistributionResult

        Returns:
            QueryResult (ScalarResult or DistributionResult)

        Raises:
            ValueError: If no arguments were provided or if dataframes don't match
        """
        if not self.variables:
            raise VariableError("No variables found in query arguments")

        # Get the dataframe from the first variable for execution
        # All variables should be from the same dataframe (validated by P class)
        dataframe = self.variables[0]._nw_frame

        # Import here to avoid circular imports
        from poffertjes.calculator import ProbabilityCalculator
        from poffertjes.result import ScalarResult, DistributionResult

        # Determine query type based on presence of expressions
        has_expressions = len(self.expressions) > 0

        if has_expressions:
            # Scalar query: p(x == 5), p(x > 10), p(x == 1, y == 2), etc.
            calculator = ProbabilityCalculator(dataframe)
            prob = calculator.calculate_scalar(self.expressions)
            return ScalarResult(prob, self.expressions, dataframe)
        else:
            # Distribution query: p(x), p(x, y), etc.
            calculator = ProbabilityCalculator(dataframe)
            dist = calculator.calculate_distribution(self.variables)
            return DistributionResult(dist, self.variables, dataframe)

    @property
    def is_scalar_query(self) -> bool:
        """Check if this is a scalar query (has expressions).

        Scalar queries return single probability values and occur when the query
        contains expressions like p(x == 5) or p(x > 10, y == 'A').

        Returns:
            True if the query has expressions (scalar query), False otherwise (distribution query)

        Examples:
            >>> # These are scalar queries
            >>> p(x == 5)  # QueryBuilder.is_scalar_query -> True
            >>> p(x > 10, y == 'A')  # QueryBuilder.is_scalar_query -> True

            >>> # These are distribution queries
            >>> p(x)  # QueryBuilder.is_scalar_query -> False
            >>> p(x, y)  # QueryBuilder.is_scalar_query -> False
        """
        return len(self.expressions) > 0

    @property
    def is_distribution_query(self) -> bool:
        """Check if this is a distribution query (no expressions).

        Distribution queries return probability distributions and occur when the query
        contains only variables like p(x) or p(x, y).

        Returns:
            True if the query has no expressions (distribution query), False otherwise (scalar query)

        Examples:
            >>> # These are distribution queries
            >>> p(x)  # QueryBuilder.is_distribution_query -> True
            >>> p(x, y)  # QueryBuilder.is_distribution_query -> True

            >>> # These are scalar queries
            >>> p(x == 5)  # QueryBuilder.is_distribution_query -> False
            >>> p(x > 10, y == 'A')  # QueryBuilder.is_distribution_query -> False
        """
        return len(self.expressions) == 0

    def __repr__(self) -> str:
        """Return string representation of the QueryBuilder."""
        query_type = "scalar" if self.is_scalar_query else "distribution"
        return f"QueryBuilder({query_type}, variables={len(self.variables)}, expressions={len(self.expressions)})"
