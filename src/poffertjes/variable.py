"""Variable and VariableBuilder classes for extracting variables from dataframes."""

from typing import Any


class Variable:
    """Represents a random variable bound to a dataframe column.

    Note: Variables store the dataframe ID and a reference to the Narwhals frame.
    Multiple variables from the same dataframe share the same frame reference,
    avoiding duplication in memory.
    """

    def __init__(self, name: str, nw_frame: Any) -> None:
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
