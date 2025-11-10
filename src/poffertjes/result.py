"""Result objects for probability queries (Distribution, ScalarResult, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict, Iterator, Tuple, Optional, TYPE_CHECKING
from narwhals.typing import FrameT

from poffertjes.exceptions import VariableError

if TYPE_CHECKING:
    from poffertjes.expression import Expression, CompositeExpression
    from poffertjes.variable import Variable
    from plotly.graph_objects import Figure


class QueryResult(ABC):
    """Base class for query results."""

    @abstractmethod
    def given(self, *conditions) -> "QueryResult":
        """Apply conditional probability.

        Args:
            *conditions: Expressions or variables to condition on.

        Returns:
            New QueryResult with conditional probability applied.
        """
        pass


class ScalarResult(QueryResult):
    """Represents a scalar probability value."""

    def __init__(
        self,
        value: float,
        expressions: Optional[List[Union["Expression", "CompositeExpression"]]] = None,
        dataframe: Optional[FrameT] = None,
    ) -> None:
        """Initialize a ScalarResult.

        Args:
            value: The probability value (between 0.0 and 1.0)
            expressions: The expressions that were evaluated to get this result
            dataframe: The Narwhals dataframe this result came from
        """
        self.value = value
        self._expressions = expressions or []
        self._dataframe = dataframe

    def _display_(self) -> float:
        return self.value

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __repr__(self) -> str:
        """String representation of the scalar result."""
        return f"{self.value:.6f}"

    def given(self, *args: Union["Expression", "Variable"]) -> "ScalarResult":
        """Calculate conditional probability P(original expressions | conditions).

        Example:
            p(x == 1).given(y == 2)  # P(X=1 | Y=2)

        Args:
            *args: Expressions or variables to condition on.

        Returns:
            New ScalarResult with conditional probability.
        """
        from poffertjes.calculator import ProbabilityCalculator

        # Parse conditioning arguments
        conditions = self._parse_conditioning_args(args)

        # Calculate conditional probability
        calculator = ProbabilityCalculator(self._dataframe)
        prob = calculator.calculate_scalar(
            expressions=self._expressions, conditions=conditions
        )

        return ScalarResult(prob, self._expressions, self._dataframe)

    def _parse_conditioning_args(self, args) -> List["Expression"]:
        """Parse arguments into list of expressions for conditioning."""
        from poffertjes.expression import Expression
        from poffertjes.variable import Variable

        conditions = []
        for arg in args:
            if isinstance(arg, Expression):
                conditions.append(arg)
            elif isinstance(arg, Variable):
                # Variable without expression means condition on all values
                # This is handled differently in distribution case
                raise VariableError(
                    "Scalar result cannot be conditioned on variable without expression. "
                    "Use an expression like y == value instead."
                )
        return conditions


class DistributionResult(QueryResult):
    """Represents a probability distribution."""

    def __init__(
        self,
        distribution: FrameT,  # Narwhals dataframe
        variables: List["Variable"],
        dataframe: FrameT,  # Narwhals dataframe
        conditions: Optional[List[Union["Expression", "Variable"]]] = None,
    ) -> None:
        """Initialize a DistributionResult.

        Args:
            distribution: The probability distribution as a Narwhals dataframe
            variables: The variables this distribution is over
            dataframe: The source Narwhals dataframe
            conditions: Optional conditioning expressions/variables
        """
        self.distribution = distribution
        self.variables = variables
        self.dataframe = dataframe
        self._conditions = conditions or []

    def _display_(self) -> Figure:
        import plotly.express as px
        return px.bar(self.distribution, x="x", y="probability")

    def given(self, *args: Union["Expression", "Variable"]) -> "DistributionResult":
        """Calculate conditional distribution P(variables | conditions).

        Examples:
            p(x).given(y == 2)        # P(X | Y=2) - distribution
            p(x).given(y)              # P(X | Y) - distribution for each Y value
            p(x, y).given(z == 3)      # P(X,Y | Z=3) - joint conditional
            p(x).given(y == 1, z == 2) # P(X | Y=1, Z=2) - multiple conditions

        Args:
            *args: Expressions or variables to condition on.

        Returns:
            New DistributionResult with conditional distribution.
        """
        from poffertjes.calculator import ProbabilityCalculator

        # Parse conditioning arguments
        conditions = self._parse_conditioning_args(args)

        # Calculate conditional distribution
        calculator = ProbabilityCalculator(self.dataframe)
        dist = calculator.calculate_distribution(
            variables=self.variables, conditions=conditions
        )

        return DistributionResult(dist, self.variables, self.dataframe, conditions)

    def _parse_conditioning_args(self, args) -> List[Union["Expression", "Variable"]]:
        """Parse arguments into list of expressions/variables for conditioning."""
        from poffertjes.expression import Expression
        from poffertjes.variable import Variable

        result = []
        for arg in args:
            if isinstance(arg, (Expression, Variable)):
                result.append(arg)
            else:
                raise VariableError(f"Invalid conditioning argument: {arg}")
        return result

    def to_dict(self) -> Dict[Any, float]:
        """Convert distribution to dictionary.

        Returns:
            Dictionary mapping values to probabilities. For single variables,
            keys are the values. For multiple variables, keys are tuples of values.
        """
        # Convert to (value, probability) pairs
        result: Dict[Any, float] = {}
        for row in self.distribution.iter_rows(named=True):
            if len(self.variables) == 1:
                key = row[self.variables[0].name]
            else:
                key = tuple(row[var.name] for var in self.variables)
            result[key] = row.get("probability", 0.0)
        return result

    def to_dataframe(self) -> Any:
        """Convert distribution to native dataframe format.

        Returns:
            Native dataframe (Pandas DataFrame or Polars DataFrame) containing
            the distribution data with columns for variables and probabilities.
        """
        return self.distribution.to_native()


class Distribution:
    """Represents a probability distribution."""

    def __init__(self, data: FrameT, variables: List[str]) -> None:
        """Initialize a Distribution.

        Args:
            data: Narwhals dataframe with columns for each variable, 'count', and 'probability'
            variables: List of variable names this distribution is over
        """
        self.data = data  # Narwhals dataframe
        self.variables = variables

    def __iter__(self) -> Iterator[Tuple[Any, float]]:
        """Iterate over (value, probability) pairs."""
        for row in self.data.iter_rows(named=True):
            if len(self.variables) == 1:
                value = row[self.variables[0]]
            else:
                value = tuple(row[v] for v in self.variables)
            yield value, row["probability"]

    def __repr__(self) -> str:
        """Display distribution in readable format."""
        lines = [f"Distribution over {', '.join(self.variables)}:"]
        lines.append("-" * 50)

        # Show first 10 rows
        rows = list(self.data.iter_rows(named=True))
        for i, row in enumerate(rows[:10]):
            if len(self.variables) == 1:
                value = row[self.variables[0]]
            else:
                value = tuple(row[v] for v in self.variables)
            prob = row["probability"]
            lines.append(f"  {value}: {prob:.6f}")

        if len(rows) > 10:
            lines.append(f"  ... ({len(rows) - 10} more values)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[Any, float]:
        """Convert distribution to dictionary format.

        Returns:
            Dictionary mapping values to probabilities. For single variables,
            keys are the values. For multiple variables, keys are tuples of values.

        Examples:
            >>> dist = p(x)  # Distribution over single variable
            >>> dist.to_dict()  # {1: 0.4, 2: 0.6}

            >>> joint_dist = p(x, y)  # Joint distribution
            >>> joint_dist.to_dict()  # {(1, 'A'): 0.2, (1, 'B'): 0.2, (2, 'A'): 0.3, (2, 'B'): 0.3}
        """
        return {value: prob for value, prob in self}

    def to_dataframe(self) -> Any:
        """Convert to native dataframe format (Pandas/Polars).

        Returns:
            Native dataframe (Pandas DataFrame or Polars DataFrame) containing
            the distribution with columns for each variable, count, and probability.

        Examples:
            >>> dist = p(x)
            >>> df = dist.to_dataframe()
            >>> print(df)
            #    x  count  probability
            # 0  1      2         0.4
            # 1  2      3         0.6
        """
        return self.data.to_native()

    def __eq__(self, other: "Distribution") -> bool:
        """Compare distributions for equality."""
        if not isinstance(other, Distribution):
            return False

        # Compare variables
        if self.variables != other.variables:
            return False

        # Compare data by converting to dictionaries and comparing
        # This handles floating point precision issues
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        if set(self_dict.keys()) != set(other_dict.keys()):
            return False

        # Compare probabilities with tolerance for floating point
        for key in self_dict:
            if abs(self_dict[key] - other_dict[key]) > 1e-10:
                return False

        return True
