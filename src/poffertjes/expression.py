"""Expression system for building probabilistic queries."""

from enum import Enum
from typing import Any, Union, List, TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from poffertjes.variable import Variable


class ExpressionOp(Enum):
    """Supported expression operators for probabilistic queries."""

    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    BETWEEN = "between"
    IN = "in"


class Expression:
    """Represents an expression on a variable (e.g., x == 5, x > 10).

    Expressions can be combined using & (AND) and | (OR) operators to create
    composite expressions. They can be converted to Narwhals expressions for
    efficient dataframe filtering.
    """

    def __init__(
        self,
        variable: "Variable",
        operator: Union[str, ExpressionOp],
        value: Any,
        upper_bound: Any = None,
    ) -> None:
        """Initialize an Expression.

        Args:
            variable: The Variable this expression operates on
            operator: The operator string (e.g., "==", "<", "in") or ExpressionOp enum
            value: The value to compare against
            upper_bound: Optional upper bound for BETWEEN operations
        """
        self.variable = variable
        # Convert string operator to ExpressionOp enum
        if isinstance(operator, str):
            self.operator = ExpressionOp(operator)
        else:
            self.operator = operator
        self.value = value
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        """Return string representation of the expression."""
        if self.operator == ExpressionOp.IN:
            return f"Expression({self.variable.name} in {self.value})"
        elif self.upper_bound is not None:
            return (
                f"Expression({self.value} < {self.variable.name} < {self.upper_bound})"
            )
        else:
            return f"Expression({self.variable.name} {self.operator.value} {self.value})"

    def __and__(self, other: "Expression") -> "CompositeExpression":
        """Combine expressions with AND logic.

        Args:
            other: Another Expression to combine with

        Returns:
            CompositeExpression representing (self AND other)

        Examples:
            >>> (x > 5) & (x < 10)  # Creates CompositeExpression with AND logic
        """
        return CompositeExpression([self, other], "AND")

    def __or__(self, other: "Expression") -> "CompositeExpression":
        """Combine expressions with OR logic.

        Args:
            other: Another Expression to combine with

        Returns:
            CompositeExpression representing (self OR other)

        Examples:
            >>> (x == 1) | (x == 2)  # Creates CompositeExpression with OR logic
        """
        return CompositeExpression([self, other], "OR")

    def to_narwhals_expr(self) -> Any:
        """Convert expression to Narwhals expression for dataframe filtering.

        This method translates the expression into a Narwhals column expression
        that can be used with df.filter() operations.

        Returns:
            Narwhals expression object

        Raises:
            ValueError: If the operator is not supported

        Examples:
            >>> expr = x == 5
            >>> nw_expr = expr.to_narwhals_expr()  # Returns nw.col('x') == 5
        """
        col = nw.col(self.variable.name)

        if self.operator == ExpressionOp.EQ:
            return col == self.value
        elif self.operator == ExpressionOp.NE:
            return col != self.value
        elif self.operator == ExpressionOp.LT:
            return col < self.value
        elif self.operator == ExpressionOp.LE:
            return col <= self.value
        elif self.operator == ExpressionOp.GT:
            return col > self.value
        elif self.operator == ExpressionOp.GE:
            return col >= self.value
        elif self.operator == ExpressionOp.BETWEEN:
            # Use is_between for efficient ternary conditions
            return col.is_between(self.value, self.upper_bound, closed="none")
        elif self.operator == ExpressionOp.IN:
            # Use is_in for categorical membership
            return col.is_in(self.value)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")


class CompositeExpression:
    """Represents multiple expressions combined with AND/OR logic.

    CompositeExpressions can be further combined with other expressions or
    composite expressions to build complex logical conditions.
    """

    def __init__(
        self,
        expressions: List[Union[Expression, "CompositeExpression"]],
        logic: str,
    ) -> None:
        """Initialize a CompositeExpression.

        Args:
            expressions: List of Expression or CompositeExpression objects
            logic: Either "AND" or "OR"

        Raises:
            ValueError: If logic is not "AND" or "OR"
        """
        if logic not in ("AND", "OR"):
            raise ValueError(f"Logic must be 'AND' or 'OR', got: {logic}")

        self.expressions = expressions
        self.logic = logic

    def __repr__(self) -> str:
        """Return string representation of the composite expression."""
        expr_strs = [repr(e) for e in self.expressions]
        logic_str = " & " if self.logic == "AND" else " | "
        return f"({logic_str.join(expr_strs)})"

    def __and__(self, other: Union[Expression, "CompositeExpression"]) -> "CompositeExpression":
        """Combine with another expression using AND logic.

        Args:
            other: Expression or CompositeExpression to combine with

        Returns:
            New CompositeExpression with AND logic

        Examples:
            >>> ((x > 5) & (x < 10)) & (y == 3)
        """
        return CompositeExpression([self, other], "AND")

    def __or__(self, other: Union[Expression, "CompositeExpression"]) -> "CompositeExpression":
        """Combine with another expression using OR logic.

        Args:
            other: Expression or CompositeExpression to combine with

        Returns:
            New CompositeExpression with OR logic

        Examples:
            >>> ((x == 1) | (x == 2)) | (x == 3)
        """
        return CompositeExpression([self, other], "OR")

    def to_narwhals_expr(self) -> Any:
        """Convert composite expression to Narwhals expression.

        This recursively converts all sub-expressions and combines them
        using the appropriate logical operator.

        Returns:
            Narwhals expression object

        Examples:
            >>> comp_expr = (x > 5) & (x < 10)
            >>> nw_expr = comp_expr.to_narwhals_expr()
        """
        # Convert all sub-expressions to Narwhals expressions
        nw_exprs = [e.to_narwhals_expr() for e in self.expressions]

        # Combine using the appropriate logic
        if self.logic == "AND":
            result = nw_exprs[0]
            for expr in nw_exprs[1:]:
                result = result & expr
            return result
        else:  # OR
            result = nw_exprs[0]
            for expr in nw_exprs[1:]:
                result = result | expr
            return result
