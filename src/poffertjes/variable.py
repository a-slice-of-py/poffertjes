"""Variable and VariableBuilder classes for extracting variables from dataframes."""

from __future__ import annotations

from typing import Any, List, Union

import narwhals as nw
from narwhals.typing import IntoFrameT, FrameT

from poffertjes.expression import Expression
from poffertjes.exceptions import DataframeError, VariableError


class Variable:
    """Represents a random variable bound to a dataframe column.

    Note: Variables store the dataframe ID and a reference to the Narwhals frame.
    Multiple variables from the same dataframe share the same frame reference,
    avoiding duplication in memory.
    """

    def __init__(self, name: str, nw_frame: FrameT) -> None:
        """Initialize a Variable.

        Args:
            name: The column name this variable represents
            nw_frame: The Narwhals dataframe this variable belongs to
        """
        self.name = name
        # Store reference to Narwhals frame (shared among variables from same dataframe)
        self._nw_frame = nw_frame
        # Compute dataframe identity for validation
        self._frame_id = id(nw_frame)

    def __repr__(self) -> str:
        """Return string representation of the variable."""
        return f"Variable({self.name})"

    def __str__(self) -> str:
        """Return string representation of the variable."""
        return f"Variable({self.name})"

    @property
    def dataframe_id(self) -> int:
        """Return unique identifier for the source dataframe."""
        return self._frame_id

    def __eq__(self, value: Any) -> Expression:
        """Create equality expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable == value

        Examples:
            >>> x == 5  # Creates Expression(x, "==", 5)
        """
        return Expression(self, "==", value)

    def __ne__(self, value: Any) -> Expression:
        """Create inequality expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable != value

        Examples:
            >>> x != 5  # Creates Expression(x, "!=", 5)
        """
        return Expression(self, "!=", value)

    def __lt__(self, value: Any) -> Expression:
        """Create less-than expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable < value

        Examples:
            >>> x < 5  # Creates Expression(x, "<", 5)
        """
        return Expression(self, "<", value)

    def __le__(self, value: Any) -> Expression:
        """Create less-than-or-equal expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable <= value

        Examples:
            >>> x <= 5  # Creates Expression(x, "<=", 5)
        """
        return Expression(self, "<=", value)

    def __gt__(self, value: Any) -> Expression:
        """Create greater-than expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable > value

        Examples:
            >>> x > 5  # Creates Expression(x, ">", 5)
        """
        return Expression(self, ">", value)

    def __ge__(self, value: Any) -> Expression:
        """Create greater-than-or-equal expression.

        Args:
            value: The value to compare against

        Returns:
            Expression representing variable >= value

        Examples:
            >>> x >= 5  # Creates Expression(x, ">=", 5)
        """
        return Expression(self, ">=", value)

    def isin(self, values: List[Any]) -> Expression:
        """Create 'in' expression for categorical variables.

        This is useful for checking if a variable's value is in a set of values,
        particularly for categorical/string variables.

        Args:
            values: List of values to check membership against

        Returns:
            Expression representing variable in values

        Raises:
            VariableError: If values list is empty

        Examples:
            >>> x.isin(['cat1', 'cat2', 'cat3'])
            >>> x.isin([1, 2, 3])
        """
        if not values:
            raise VariableError(
                "Cannot create 'isin' expression with empty values list"
            )
        return Expression(self, "in", values)


class VariableBuilder:
    """Factory for creating Variable objects from dataframes.

    The builder stores the dataframe once, and all variables created from it
    reference the builder. This avoids duplicating the dataframe in memory.
    """

    def __init__(self, data: IntoFrameT) -> None:
        """Initialize a VariableBuilder with a dataframe.

        Args:
            data: A Pandas or Polars dataframe (or any Narwhals-compatible frame)

        Raises:
            ValueError: If the dataframe is empty
        """
        # Convert to Narwhals frame
        self._nw_frame = nw.from_native(data)

        # Validate dataframe is not empty using lazy-friendly approach
        # For lazy frames, we need to collect the head to check if it's empty
        try:
            # Try to get first row - for lazy frames this will need to be collected
            first_row = self._nw_frame.head(1)

            # Check if it's a lazy frame (Polars LazyFrame)
            if hasattr(first_row, "collect"):
                # It's a lazy frame, collect it to check length
                collected_row = first_row.collect()
                if len(collected_row) == 0:
                    raise DataframeError(
                        "Cannot create variables from an empty dataframe"
                    )
            else:
                # It's an eager frame, can check length directly
                if len(first_row) == 0:
                    raise DataframeError(
                        "Cannot create variables from an empty dataframe"
                    )
        except Exception:
            # Fallback to len() check if head() fails (for eager frames)
            try:
                if len(self._nw_frame) == 0:
                    raise DataframeError(
                        "Cannot create variables from an empty dataframe"
                    )
            except TypeError:
                # If len() also fails (lazy frame), we'll assume it's not empty
                # This is a reasonable assumption since most lazy operations wouldn't
                # be created on empty datasets
                pass

        # Cache the dataframe identity
        self._id = id(self._nw_frame)

    def _get_column_names(self) -> List[str]:
        """Get column names efficiently, handling lazy frames properly.

        For lazy frames (like Polars LazyFrame), this uses collect_schema().names()
        to avoid the performance warning. For eager frames, it uses .columns directly.

        Returns:
            List of column names
        """
        # Check if this is a lazy frame that has collect_schema method
        native_frame = self._nw_frame.to_native()
        if hasattr(native_frame, "collect_schema"):
            # This is a Polars LazyFrame, use collect_schema to avoid warning
            return native_frame.collect_schema().names()
        else:
            # This is an eager frame, safe to use .columns
            return self._nw_frame.columns

    @property
    def dataframe_id(self) -> int:
        """Return unique identifier for the dataframe."""
        return self._id

    def get_variables(self, *args: str) -> Union[List[Variable], Variable]:
        """Extract variables from dataframe columns.

        Args:
            *args: Column names. If empty, returns all columns.
                   If single column name, returns single Variable.
                   If multiple column names, returns list of Variables.

        Returns:
            Variable or List of Variable objects.

        Raises:
            ValueError: If a column name doesn't exist in the dataframe.

        Examples:
            >>> vb = VariableBuilder.from_data(df)
            >>> x, y, z = vb.get_variables('x', 'y', 'z')
            >>> all_vars = vb.get_variables()  # All columns
        """
        # Determine which columns to extract
        if args:
            columns = list(args)
        else:
            # Get columns efficiently, avoiding performance warnings for lazy frames
            columns = self._get_column_names()

        # Validate columns exist
        available_columns = self._get_column_names()
        missing = set(columns) - set(available_columns)
        if missing:
            raise VariableError(
                f"Columns not found in dataframe: {sorted(missing)}. "
                f"Available columns: {sorted(available_columns)}"
            )

        # Create Variable objects
        # All variables share the same frame reference (no duplication)
        variables = [Variable(name, self._nw_frame) for name in columns]

        # Return single Variable if only one requested, otherwise list
        if len(variables) == 1 and args:
            return variables[0]
        return variables

    @staticmethod
    def from_data(data: IntoFrameT) -> "VariableBuilder":
        """Create a VariableBuilder from a dataframe.

        Args:
            data: A Pandas or Polars dataframe (or any Narwhals-compatible frame)

        Returns:
            A new VariableBuilder instance

        Raises:
            ValueError: If the dataframe is empty

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            >>> vb = VariableBuilder.from_data(df)
            >>> x, y = vb.get_variables('x', 'y')
        """
        return VariableBuilder(data)
