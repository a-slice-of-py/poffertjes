"""ProbabilityCalculator for computing probabilities using Narwhals operations."""

from typing import Any, List, Optional, TYPE_CHECKING
import narwhals as nw

if TYPE_CHECKING:
    from poffertjes.variable import Variable
    from poffertjes.expression import Expression


class ProbabilityCalculator:
    """Calculates probabilities using Narwhals operations.
    
    This class implements frequency-based probability calculation using empirical
    counts from the data. It uses Narwhals for dataframe-agnostic operations
    that work with both Pandas and Polars.
    
    Requirements addressed:
    - 6.1: Use frequency counting as the estimation method
    - 6.2: Use Narwhals operations for dataframe-agnostic counting
    """
    
    def __init__(self, dataframe: Any) -> None:
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
        conditions: Optional[List["Expression"]] = None
    ) -> Any:
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
        - 6.3: Count rows where conditions hold and divide by total
        - 6.4: Handle joint probabilities P(X,Y)
        - 7.2: Use group_by operations followed by aggregations
        - 7.3: Use df.group_by(['X', 'Y']).agg(nw.len()) pattern
        """
        df = self.df
        
        # Apply conditions if present (for conditional probabilities)
        if conditions:
            for condition in conditions:
                df = df.filter(condition.to_narwhals_expr())
            
            # Check if conditioning event has any occurrences
            conditional_count = len(df)
            if conditional_count == 0:
                raise ValueError("Conditioning event has zero probability")
            
            # For conditional probabilities, normalize by the conditional count
            total = conditional_count
        else:
            # For marginal probabilities, normalize by the total count
            total = self.total_count
        
        # Extract variable names for group_by operation
        var_names = [var.name for var in variables]
        
        # Use efficient group_by + agg pattern as specified in requirements
        # This satisfies requirement 7.2: use group_by operations followed by aggregations
        result = (
            df.group_by(var_names)
            .agg(count=nw.len())  # Count occurrences of each combination
            .with_columns(
                probability=nw.col("count") / total  # Calculate probabilities
            )
            .sort(var_names)  # Sort for consistent output
        )
        
        return result
