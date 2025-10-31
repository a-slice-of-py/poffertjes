"""Expression system for building probabilistic queries."""

from typing import Any


class Expression:
    """Represents an expression on a variable (e.g., x == 5, x > 10).

    This is a minimal implementation to support Variable operator overloading.
    Full implementation will be added in task 3.
    """

    def __init__(
        self, variable: "Variable", operator: str, value: Any, upper_bound: Any = None
    ) -> None:
        """Initialize an Expression.

        Args:
            variable: The Variable this expression operates on
            operator: The operator string (e.g., "==", "<", "in")
            value: The value to compare against
            upper_bound: Optional upper bound for BETWEEN operations
        """
        self.variable = variable
        self.operator = operator
        self.value = value
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        """Return string representation of the expression."""
        if self.operator == "in":
            return f"Expression({self.variable.name} in {self.value})"
        elif self.upper_bound is not None:
            return (
                f"Expression({self.value} < {self.variable.name} < {self.upper_bound})"
            )
        else:
            return f"Expression({self.variable.name} {self.operator} {self.value})"
