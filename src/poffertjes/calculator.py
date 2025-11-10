"""ProbabilityCalculator for computing probabilities using Narwhals operations."""

from typing import Any, List, Optional, Union, TYPE_CHECKING
import narwhals as nw
from narwhals.typing import FrameT

from poffertjes.exceptions import ProbabilityError, DataframeError

if TYPE_CHECKING:
    from poffertjes.variable import Variable
    from poffertjes.expression import Expression, CompositeExpression


class ProbabilityCalculator:
    """Calculates probabilities using Narwhals operations.

    This class implements frequency-based probability calculation using empirical
    counts from the data. It uses Narwhals for dataframe-agnostic operations
    that work with both Pandas and Polars.

    Requirements addressed:
    - 6.1: Use frequency counting as the estimation method
    - 6.2: Use Narwhals operations for dataframe-agnostic counting
    """

    def __init__(self, dataframe: FrameT) -> None:
        """Initialize the calculator with a dataframe.

        Args:
            dataframe: A Narwhals-compatible dataframe (Pandas or Polars)

        Requirements addressed:
        - 6.1: Calculate probabilities using frequency counting
        - 6.2: Use Narwhals operations for counting
        """
        self.df = dataframe
        # Calculate total count for probability normalization
        # This satisfies requirement 6.3: count rows and divide by total
        self.total_count = len(dataframe)

    def calculate_distribution(
        self,
        variables: List["Variable"],
        conditions: Optional[List["Expression"]] = None,
    ) -> FrameT:
        """Calculate probability distribution using group_by + agg.

        This method calculates marginal probability distributions P(X) or P(X,Y)
        using efficient Narwhals group_by operations. For conditional probabilities,
        it first filters the dataframe based on the conditions.

        Args:
            variables: List of Variable objects to calculate distribution for
            conditions: Optional list of Expression objects for conditioning

        Returns:
            Narwhals dataframe with columns for each variable, 'count', and 'probability'

        Raises:
            ValueError: If conditioning event has zero probability (no matching rows)

        Examples:
            For P(X): df.group_by('X').agg(nw.len())
            For P(X,Y): df.group_by(['X', 'Y']).agg(nw.len())
            For P(X|Y=y): df.filter(Y==y).group_by('X').agg(nw.len())

        Requirements addressed:
        - 4.1: Return probability distribution of variables
        - 4.3: Include all observed values and their probabilities
        - 4.4: Probabilities sum to 1.0 (within floating point precision)
        - 5.1: Return conditional probability distribution of x given y
        - 5.3: Return distribution of x conditioned on y equals value
        - 5.4: Return P(X|Y,Z) for multiple conditioning variables
        - 5.6: Return P(X=x_i|Y=y_j) for all combinations
        - 5.7: Raise clear error when conditioning event has zero occurrences
        - 5.8: Conditional probabilities sum to 1.0
        - 6.3: Count rows where conditions hold and divide by total
        - 6.4: Handle joint probabilities P(X,Y)
        - 6.5: Count rows where both hold, divided by rows where Y=y holds
        - 7.2: Use group_by operations followed by aggregations
        - 7.3: Use df.group_by(['X', 'Y']).agg(nw.len()) pattern
        - 7.5: Use efficient filter + group_by operations
        """
        df = self.df

        # Apply conditions if present (for conditional probabilities)
        if conditions:
            # Apply all conditioning expressions using efficient filter operations
            # This satisfies requirement 7.5: use efficient filter + group_by operations
            for condition in conditions:
                df = df.filter(condition.to_narwhals_expr())

            # Check if conditioning event has any occurrences
            # This satisfies requirement 5.7: raise clear error when conditioning event has zero occurrences
            conditional_count = len(df)
            if conditional_count == 0:
                raise ProbabilityError(
                    "Conditioning event has zero probability - no rows match the given conditions"
                )

            # For conditional probabilities, normalize by the conditional count
            # This satisfies requirement 6.5: count rows where both hold, divided by rows where Y=y holds
            total = conditional_count
        else:
            # For marginal probabilities, normalize by the total count
            total = self.total_count

        # Handle empty dataframe case
        if total == 0:
            # Return empty result with correct structure using pandas
            import pandas as pd

            var_names = [var.name for var in variables]
            empty_dict = {name: [] for name in var_names}
            empty_dict.update({"count": [], "probability": []})
            empty_result = nw.from_native(pd.DataFrame(empty_dict))
            return empty_result

        # Extract variable names for group_by operation
        var_names = [var.name for var in variables]

        # Use efficient group_by + agg pattern as specified in requirements
        # This satisfies requirement 7.2: use group_by operations followed by aggregations
        # This satisfies requirement 7.3: use df.group_by(['X', 'Y']).agg(nw.len()) pattern
        result = (
            df.group_by(var_names)
            .agg(count=nw.len())  # Count occurrences of each combination
            .with_columns(
                probability=nw.col("count") / total  # Calculate probabilities
            )
            .sort(var_names)  # Sort for consistent output
        )

        return result

    def calculate_scalar(
        self,
        expressions: List[Union["Expression", "CompositeExpression"]],
        conditions: Optional[List["Expression"]] = None,
    ) -> float:
        """Calculate scalar probability using filter operations.

        This method calculates scalar probabilities P(expressions) or conditional
        probabilities P(expressions|conditions) using efficient Narwhals filter
        operations. It counts rows that satisfy the expressions and divides by
        the appropriate denominator.

        Args:
            expressions: List of Expression or CompositeExpression objects to evaluate
            conditions: Optional list of Expression objects for conditioning

        Returns:
            Scalar probability as a float between 0.0 and 1.0

        Raises:
            ValueError: If conditioning event has zero probability (no matching rows)

        Examples:
            For P(X=x): count(X=x) / total
            For P(X=x|Y=y): count(X=x AND Y=y) / count(Y=y)
            For P(X=x AND Y=y): count(X=x AND Y=y) / total

        Requirements addressed:
        - 4.2: Return scalar probability for conditions
        - 4.4: Handle zero probability conditions appropriately
        - 5.2: Return scalar conditional probability P(X=value1|Y=value2)
        - 5.5: Return P(X|Y=value1 AND Z=value2) for multiple conditions
        - 5.7: Raise clear error when conditioning event has zero occurrences
        - 6.3: Count rows where conditions hold and divide by total
        - 6.5: Count rows where both hold, divided by rows where Y=y holds
        - 7.5: Use efficient filter + group_by operations
        - 7.9: Use Narwhals column expression methods
        - 7.10: Use Narwhals filter expressions rather than manual row selection
        - 9.1: Support comparison operators (==, !=, <, >, <=, >=)
        - 9.2: Support multiple conditions combined with AND
        - 9.3: Support ternary conditions like a < x < b
        """
        df = self.df

        # Apply conditions first (for conditional probabilities)
        if conditions:
            # Apply all conditioning expressions using efficient filter operations
            # This satisfies requirement 7.5: use efficient filter + group_by operations
            for condition in conditions:
                df = df.filter(condition.to_narwhals_expr())

            # Check if conditioning event has any occurrences
            # This satisfies requirement 5.7: raise clear error when conditioning event has zero occurrences
            denominator = len(df)
            if denominator == 0:
                raise ProbabilityError(
                    "Conditioning event has zero probability - no rows match the given conditions"
                )
        else:
            # For marginal probabilities, use total count as denominator
            denominator = self.total_count

        # Apply expressions (the events we want to calculate probability for)
        # This satisfies requirement 7.10: use Narwhals filter expressions rather than manual row selection
        for expression in expressions:
            df = df.filter(expression.to_narwhals_expr())

        # Count rows that satisfy all expressions
        numerator = len(df)

        # Calculate probability
        # Handle edge case where denominator is 0 (empty dataframe)
        if denominator == 0:
            return 0.0

        # This satisfies requirement 6.5: count rows where both hold, divided by rows where Y=y holds
        return numerator / denominator

    def calculate_joint(
        self,
        variables: List["Variable"],
        conditions: Optional[List["Expression"]] = None,
    ) -> FrameT:
        """Calculate joint probability distribution using multi-column group_by.

        This method calculates joint probability distributions P(X,Y) or P(X,Y,Z)
        using efficient Narwhals multi-column group_by operations. It's essentially
        the same as calculate_distribution but explicitly designed for multiple variables
        to make the intent clear when calculating joint probabilities.

        Args:
            variables: List of Variable objects to calculate joint distribution for
            conditions: Optional list of Expression objects for conditioning

        Returns:
            Narwhals dataframe with columns for each variable, 'count', and 'probability'

        Raises:
            ValueError: If conditioning event has zero probability (no matching rows)

        Examples:
            For P(X,Y): df.group_by(['X', 'Y']).agg(nw.len())
            For P(X,Y,Z): df.group_by(['X', 'Y', 'Z']).agg(nw.len())
            For P(X,Y|Z=z): df.filter(Z==z).group_by(['X', 'Y']).agg(nw.len())

        Requirements addressed:
        - 6.4: Handle joint probabilities P(X,Y)
        - 7.4: Use multi-column group_by for joint distributions
        - 11.1: Return joint probability distribution for multiple variables
        - 11.2: Return P(X=value1 AND Y=value2) when conditions are provided
        - 11.3: Include all observed combinations of values
        - 11.4: Support conditional joint distributions P(X,Y|Z)
        - 11.5: Joint probabilities sum to 1.0 across all combinations
        """
        # Joint probability calculation is the same as distribution calculation
        # but we make it explicit that this is for multiple variables
        if len(variables) < 2:
            raise ProbabilityError(
                "Joint probability calculation requires at least 2 variables. "
                "Use calculate_distribution for single variable distributions."
            )

        # Use the same efficient implementation as calculate_distribution
        # This satisfies requirement 7.4: use multi-column group_by for joint distributions
        return self.calculate_distribution(variables, conditions)
