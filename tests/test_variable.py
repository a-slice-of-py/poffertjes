"""Unit tests for Variable and VariableBuilder classes."""

import pandas as pd
import narwhals as nw
from poffertjes.variable import Variable, VariableBuilder


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



class TestVariableBuilder:
    """Tests for the VariableBuilder class."""

    def test_variablebuilder_creation_pandas(self):
        """Test that VariableBuilder can be created with a Pandas dataframe."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder(df)

        assert vb._nw_frame is not None
        assert vb.dataframe_id == id(vb._nw_frame)

    def test_variablebuilder_creation_polars(self):
        """Test that VariableBuilder can be created with a Polars dataframe."""
        try:
            import polars as pl
            df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            vb = VariableBuilder(df)

            assert vb._nw_frame is not None
            assert vb.dataframe_id == id(vb._nw_frame)
        except ImportError:
            # Skip test if Polars is not installed
            pass

    def test_variablebuilder_empty_dataframe_error(self):
        """Test that VariableBuilder raises error for empty dataframe."""
        import pytest
        df = pd.DataFrame({"x": [], "y": []})

        with pytest.raises(ValueError, match="Cannot create variables from an empty dataframe"):
            VariableBuilder(df)

    def test_from_data_static_method(self):
        """Test that from_data static method creates a VariableBuilder."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)

        assert isinstance(vb, VariableBuilder)
        assert vb._nw_frame is not None

    def test_get_variables_all_columns(self):
        """Test that get_variables without arguments returns all columns."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        vb = VariableBuilder.from_data(df)

        variables = vb.get_variables()

        assert len(variables) == 3
        assert all(isinstance(v, Variable) for v in variables)
        assert {v.name for v in variables} == {"x", "y", "z"}

    def test_get_variables_specific_columns(self):
        """Test that get_variables with column names returns those specific columns."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        vb = VariableBuilder.from_data(df)

        variables = vb.get_variables("x", "z")

        assert len(variables) == 2
        assert all(isinstance(v, Variable) for v in variables)
        assert {v.name for v in variables} == {"x", "z"}

    def test_get_variables_single_column_returns_variable(self):
        """Test that get_variables with single column name returns a Variable (not list)."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder.from_data(df)

        variable = vb.get_variables("x")

        assert isinstance(variable, Variable)
        assert variable.name == "x"

    def test_get_variables_unpacking(self):
        """Test that get_variables can be unpacked into multiple variables."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        vb = VariableBuilder.from_data(df)

        x, y, z = vb.get_variables("x", "y", "z")

        assert x.name == "x"
        assert y.name == "y"
        assert z.name == "z"

    def test_get_variables_missing_column_error(self):
        """Test that get_variables raises error for non-existent column."""
        import pytest
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder.from_data(df)

        with pytest.raises(ValueError, match="Columns not found in dataframe: \\['z'\\]"):
            vb.get_variables("x", "z")

    def test_get_variables_multiple_missing_columns_error(self):
        """Test that get_variables shows all missing columns in error message."""
        import pytest
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)

        with pytest.raises(ValueError, match="Columns not found in dataframe"):
            vb.get_variables("a", "b", "c")

    def test_get_variables_error_shows_available_columns(self):
        """Test that error message includes available columns."""
        import pytest
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder.from_data(df)

        with pytest.raises(ValueError, match="Available columns"):
            vb.get_variables("z")

    def test_variables_share_same_frame_reference(self):
        """Test that all variables from same builder share the same frame reference."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        vb = VariableBuilder.from_data(df)

        x, y, z = vb.get_variables("x", "y", "z")

        # All variables should reference the same frame object
        assert x._nw_frame is y._nw_frame
        assert y._nw_frame is z._nw_frame
        assert x.dataframe_id == y.dataframe_id == z.dataframe_id

    def test_variables_from_different_builders_have_different_ids(self):
        """Test that variables from different builders have different dataframe IDs."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"y": [4, 5, 6]})

        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)

        x = vb1.get_variables("x")
        y = vb2.get_variables("y")

        assert x.dataframe_id != y.dataframe_id

    def test_dataframe_id_property(self):
        """Test that dataframe_id property returns correct ID."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)

        assert vb.dataframe_id == id(vb._nw_frame)

    def test_get_variables_preserves_column_order(self):
        """Test that get_variables returns variables in the order requested."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        vb = VariableBuilder.from_data(df)

        variables = vb.get_variables("c", "a", "b")

        assert [v.name for v in variables] == ["c", "a", "b"]

    def test_get_variables_with_various_dtypes(self):
        """Test that get_variables works with various column dtypes."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        vb = VariableBuilder.from_data(df)

        variables = vb.get_variables("int_col", "float_col", "str_col", "bool_col")

        assert len(variables) == 4
        assert all(isinstance(v, Variable) for v in variables)

    def test_variablebuilder_with_duplicate_column_names(self):
        """Test that requesting the same column multiple times works."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder.from_data(df)

        # Requesting same column multiple times should work
        variables = vb.get_variables("x", "x", "y")

        assert len(variables) == 3
        assert variables[0].name == "x"
        assert variables[1].name == "x"
        assert variables[2].name == "y"
        # All should reference the same frame
        assert variables[0]._nw_frame is variables[1]._nw_frame is variables[2]._nw_frame

