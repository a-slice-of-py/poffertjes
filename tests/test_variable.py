"""Unit tests for Variable and VariableBuilder classes."""

import pandas as pd
import narwhals as nw
from poffertjes.variable import Variable


class TestVariable:
    """Tests for the Variable class."""

    def test_variable_creation(self):
        """Test that a Variable can be created with required parameters."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)

        var = Variable("x", nw_frame)

        assert var.name == "x"
        assert var._nw_frame is nw_frame
        assert var._frame_id == id(nw_frame)

    def test_variable_repr(self):
        """Test that Variable has a proper __repr__ method."""
        df = pd.DataFrame({"my_column": [1, 2, 3]})
        nw_frame = nw.from_native(df)

        var = Variable("my_column", nw_frame)

        assert repr(var) == "Variable(my_column)"

    def test_variable_str(self):
        """Test that Variable has a proper __str__ method."""
        df = pd.DataFrame({"test_var": [1, 2, 3]})
        nw_frame = nw.from_native(df)

        var = Variable("test_var", nw_frame)

        assert str(var) == "Variable(test_var)"

    def test_dataframe_id_property(self):
        """Test that dataframe_id property returns the correct frame ID."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)

        var = Variable("x", nw_frame)

        assert var.dataframe_id == id(nw_frame)

    def test_multiple_variables_same_dataframe(self):
        """Test that multiple variables from the same dataframe share the same frame reference."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        nw_frame = nw.from_native(df)

        var_x = Variable("x", nw_frame)
        var_y = Variable("y", nw_frame)

        # Both variables should reference the same frame object
        assert var_x._nw_frame is var_y._nw_frame
        assert var_x.dataframe_id == var_y.dataframe_id

    def test_variables_different_dataframes(self):
        """Test that variables from different dataframes have different IDs."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"y": [4, 5, 6]})

        nw_frame1 = nw.from_native(df1)
        nw_frame2 = nw.from_native(df2)

        var_x = Variable("x", nw_frame1)
        var_y = Variable("y", nw_frame2)

        # Variables from different dataframes should have different IDs
        assert var_x.dataframe_id != var_y.dataframe_id

    def test_variable_name_with_special_characters(self):
        """Test that Variable handles column names with special characters."""
        df = pd.DataFrame({"col_with_underscore": [1, 2, 3]})
        nw_frame = nw.from_native(df)

        var = Variable("col_with_underscore", nw_frame)

        assert var.name == "col_with_underscore"
        assert repr(var) == "Variable(col_with_underscore)"
