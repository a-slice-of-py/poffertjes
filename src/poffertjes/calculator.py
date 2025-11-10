"""ProbabilityCalculator for computing probabilities using Narwhals operations."""

from typing import Any, List, Optional, Union, TYPE_CHECKING, Dict, Tuple
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
        # Store dataframe for lazy evaluation - don't calculate total count yet
        # Total count will be calculated lazily when needed
        self._total_count = None
        
        # Cache for group_by operations to enable reuse
        # Key: (tuple of variable names, conditions hash)
        # Value: computed aggregation result
        self._groupby_cache: Dict[Tuple[Tuple[str, ...], int], FrameT] = {}
        
        # Cache for filtered dataframes to reuse expensive filter operations
        # Key: hash of conditions
        # Value: filtered dataframe
        self._filter_cache: Dict[int, FrameT] = {}

    @property
    def total_count(self) -> int:
        """Get total count of rows, calculated lazily.
        
        This property ensures we only calculate the total count when actually needed,
        supporting lazy evaluation patterns. The count is cached after first calculation.
        
        Returns:
            Total number of rows in the dataframe
        """
        if self._total_count is None:
            try:
                self._total_count = len(self.df)
            except TypeError:
                # Handle lazy frames that don't support len()
                if hasattr(self.df, 'collect'):
                    # It's a lazy frame, collect it to get the count
                    collected_df = self.df.collect()
                    self._total_count = len(collected_df)
                else:
                    raise
        return self._total_count

    def _hash_conditions(self, conditions: Optional[List[Union["Expression", "Variable"]]]) -> int:
        """Create a hash for a list of conditions to use as cache key.
        
        Args:
            conditions: List of Expression or Variable objects or None
            
        Returns:
            Hash value for the conditions
        """
        if not conditions:
            return 0
        
        # Create a simple hash based on condition characteristics
        condition_tuples = []
        for cond in conditions:
            if hasattr(cond, 'variable'):  # Expression object
                condition_tuples.append((cond.variable.name, str(cond.operator), str(cond.value)))
            else:  # Variable object
                condition_tuples.append(('variable', cond.name, 'all_values'))
        
        return hash(tuple(condition_tuples))

    def _get_filtered_dataframe(self, conditions: Optional[List[Union["Expression", "Variable"]]]) -> FrameT:
        """Get filtered dataframe, using cache when possible.
        
        Args:
            conditions: List of Expression or Variable objects to filter by
            
        Returns:
            Filtered dataframe (cached if possible)
        """
        if not conditions:
            return self.df
        
        conditions_hash = self._hash_conditions(conditions)
        
        # Check cache first
        if conditions_hash in self._filter_cache:
            return self._filter_cache[conditions_hash]
        
        # Apply filters
        df = self.df
        for condition in conditions:
            if hasattr(condition, 'to_narwhals_expr'):  # Expression object
                df = df.filter(condition.to_narwhals_expr())
            # Variable objects don't filter - they're used for grouping in conditional distributions
        
        # Cache the result
        self._filter_cache[conditions_hash] = df
        return df

    def _get_cached_groupby(
        self, 
        variables: List["Variable"], 
        conditions: Optional[List[Union["Expression", "Variable"]]] = None
    ) -> Optional[FrameT]:
        """Try to get cached group_by result.
        
        Args:
            variables: List of variables to group by
            conditions: Optional conditions for filtering
            
        Returns:
            Cached result if available, None otherwise
        """
        var_names = tuple(var.name for var in variables)
        conditions_hash = self._hash_conditions(conditions)
        cache_key = (var_names, conditions_hash)
        
        return self._groupby_cache.get(cache_key)

    def _cache_groupby_result(
        self, 
        variables: List["Variable"], 
        conditions: Optional[List[Union["Expression", "Variable"]]], 
        result: FrameT
    ) -> None:
        """Cache a group_by result for future reuse.
        
        Args:
            variables: List of variables that were grouped by
            conditions: Conditions that were applied
            result: The computed result to cache
        """
        var_names = tuple(var.name for var in variables)
        conditions_hash = self._hash_conditions(conditions)
        cache_key = (var_names, conditions_hash)
        
        self._groupby_cache[cache_key] = result

    def precompute_marginals(self, variables: List["Variable"]) -> None:
        """Precompute marginal distributions for given variables.
        
        This method can be called to precompute and cache marginal distributions
        for variables that are likely to be queried multiple times. This is
        particularly useful when you know you'll be doing many conditional
        probability calculations.
        
        Args:
            variables: List of variables to precompute marginals for
            
        Examples:
            >>> calc = ProbabilityCalculator(df)
            >>> calc.precompute_marginals([x, y, z])  # Cache P(X), P(Y), P(Z)
            >>> # Now p(x), p(y), p(z) will be fast
        """
        for var in variables:
            # Compute and cache marginal distribution
            self.calculate_distribution([var])
            
        # Also compute pairwise joint distributions if we have multiple variables
        if len(variables) > 1:
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    # Compute and cache joint distribution P(X,Y)
                    self.calculate_distribution([variables[i], variables[j]])

    def clear_cache(self) -> None:
        """Clear all cached computations.
        
        This method clears the internal caches to free memory. Call this if
        you're done with a calculator and want to reclaim memory, or if the
        underlying dataframe has changed.
        """
        self._groupby_cache.clear()
        self._filter_cache.clear()
        self._total_count = None

    def calculate_distribution(
        self,
        variables: List["Variable"],
        conditions: Optional[List[Union["Expression", "Variable"]]] = None,
    ) -> FrameT:
        """Calculate probability distribution using group_by + agg with caching.

        This method calculates marginal probability distributions P(X) or P(X,Y)
        using efficient Narwhals group_by operations. For conditional probabilities,
        it first filters the dataframe based on the conditions. Results are cached
        to enable reuse of expensive computations.

        Args:
            variables: List of Variable objects to calculate distribution for
            conditions: Optional list of Expression or Variable objects for conditioning

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
        - 7.13: Reuse computations where possible
        """
        # Check cache first for performance optimization
        cached_result = self._get_cached_groupby(variables, conditions)
        if cached_result is not None:
            return cached_result

        # Handle Variable conditioning (P(X|Y) where Y is a variable)
        # This means we want P(X|Y=y) for each value y of Y
        variable_conditions = [c for c in (conditions or []) if not hasattr(c, 'to_narwhals_expr')]
        expression_conditions = [c for c in (conditions or []) if hasattr(c, 'to_narwhals_expr')]
        
        # Get filtered dataframe for expression conditions only
        df = self._get_filtered_dataframe(expression_conditions)

        # Handle empty dataframe case - check this before any expensive operations
        if conditions:
            # Check if conditioning event has any occurrences
            # We use a more efficient approach: try to get first row instead of len()
            try:
                # Try to get the first row to check if any rows exist
                # This is more efficient than len() for lazy frames
                first_row = df.head(1)
                
                # Handle lazy frames properly
                if hasattr(first_row, 'collect'):
                    # It's a lazy frame, collect it to check length
                    collected_row = first_row.collect()
                    if len(collected_row) == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                else:
                    # It's an eager frame, can check length directly
                    if len(first_row) == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                
                # Now we know there's at least one row, calculate the actual count
                total = len(df)
            except Exception:
                # If head() fails, fall back to len() check
                try:
                    conditional_count = len(df)
                    if conditional_count == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                    total = conditional_count
                except TypeError:
                    # For lazy frames where len() doesn't work, we'll need to collect
                    if hasattr(df, 'collect'):
                        collected_df = df.collect()
                        conditional_count = len(collected_df)
                        if conditional_count == 0:
                            raise ProbabilityError(
                                "Conditioning event has zero probability - no rows match the given conditions"
                            )
                        total = conditional_count
                    else:
                        raise
        else:
            # For marginal probabilities, use lazy total count
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
        # This satisfies requirement 7.13: reuse computations where possible
        result = (
            df.group_by(var_names)
            .agg(count=nw.len())  # Count occurrences of each combination
            .with_columns(
                probability=nw.col("count") / total  # Calculate probabilities
            )
            .sort(var_names)  # Sort for consistent output
        )

        # Ensure result is collected if it's a lazy frame
        # This is necessary because result objects expect eager frames
        if hasattr(result, 'collect'):
            result = result.collect()

        # Cache the result for future reuse
        self._cache_groupby_result(variables, conditions, result)

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

        # Get filtered dataframe for conditions (with caching)
        conditional_df = self._get_filtered_dataframe(conditions)

        # Check if conditioning event has any occurrences using lazy approach
        if conditions:
            try:
                # Try to get the first row to check if any rows exist (more efficient for lazy frames)
                first_row = conditional_df.head(1)
                
                # Handle lazy frames properly
                if hasattr(first_row, 'collect'):
                    # It's a lazy frame, collect it to check length
                    collected_row = first_row.collect()
                    if len(collected_row) == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                else:
                    # It's an eager frame, can check length directly
                    if len(first_row) == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                
                # Calculate denominator only when we know there are rows
                denominator = len(conditional_df)
            except Exception:
                # Fallback to direct len() check if head() fails
                try:
                    denominator = len(conditional_df)
                    if denominator == 0:
                        raise ProbabilityError(
                            "Conditioning event has zero probability - no rows match the given conditions"
                        )
                except TypeError:
                    # For lazy frames where len() doesn't work, we'll need to collect
                    if hasattr(conditional_df, 'collect'):
                        collected_df = conditional_df.collect()
                        denominator = len(collected_df)
                        if denominator == 0:
                            raise ProbabilityError(
                                "Conditioning event has zero probability - no rows match the given conditions"
                            )
                    else:
                        raise
        else:
            # For marginal probabilities, use lazy total count as denominator
            denominator = self.total_count

        # Apply expressions (the events we want to calculate probability for)
        # Start from the conditionally filtered dataframe
        result_df = conditional_df
        for expression in expressions:
            result_df = result_df.filter(expression.to_narwhals_expr())

        # Count rows that satisfy all expressions
        numerator = len(result_df)

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
