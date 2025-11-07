"""Unit tests for Variable and VariableBuilder classes."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.variable import Variable, VariableBuilder
from poffertjes.expression import Expression, ExpressionOp
from poffertjes.exceptions import DataframeError, VariableError


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


class TestVariableOperatorOverloading:
    """Tests for Variable operator overloading."""

    def test_eq_operator(self):
        """Test that == operator creates an Expression."""        
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var == 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.EQ
        assert expr.value == 5

    def test_ne_operator(self):
        """Test that != operator creates an Expression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var != 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.NE
        assert expr.value == 5

    def test_lt_operator(self):
        """Test that < operator creates an Expression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var < 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.LT
        assert expr.value == 5

    def test_le_operator(self):
        """Test that <= operator creates an Expression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var <= 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.LE
        assert expr.value == 5

    def test_gt_operator(self):
        """Test that > operator creates an Expression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var > 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.GT
        assert expr.value == 5

    def test_ge_operator(self):
        """Test that >= operator creates an Expression."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var >= 5

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.GE
        assert expr.value == 5

    def test_isin_method(self):
        """Test that isin() method creates an Expression."""
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var.isin(["a", "b", "c"])

        assert isinstance(expr, Expression)
        assert expr.variable is var
        assert expr.operator == ExpressionOp.IN
        assert expr.value == ["a", "b", "c"]

    def test_operators_with_integer_values(self):
        """Test operators work with integer values."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr_eq = var == 10
        expr_lt = var < 20
        expr_gt = var > 0

        assert expr_eq.value == 10
        assert expr_lt.value == 20
        assert expr_gt.value == 0

    def test_operators_with_float_values(self):
        """Test operators work with float values."""
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr_eq = var == 1.5
        expr_le = var <= 2.5
        expr_ge = var >= 0.5

        assert expr_eq.value == 1.5
        assert expr_le.value == 2.5
        assert expr_ge.value == 0.5

    def test_operators_with_string_values(self):
        """Test operators work with string values."""
        df = pd.DataFrame({"category": ["cat1", "cat2", "cat3"]})
        nw_frame = nw.from_native(df)
        var = Variable("category", nw_frame)

        expr_eq = var == "cat1"
        expr_ne = var != "cat2"

        assert expr_eq.value == "cat1"
        assert expr_ne.value == "cat2"

    def test_operators_with_boolean_values(self):
        """Test operators work with boolean values."""
        df = pd.DataFrame({"flag": [True, False, True]})
        nw_frame = nw.from_native(df)
        var = Variable("flag", nw_frame)

        expr_eq = var == True
        expr_ne = var != False

        assert expr_eq.value is True
        assert expr_ne.value is False

    def test_isin_with_integer_list(self):
        """Test isin() works with list of integers."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var.isin([1, 3, 5])

        assert expr.value == [1, 3, 5]
        assert expr.operator == ExpressionOp.IN

    def test_isin_with_string_list(self):
        """Test isin() works with list of strings."""
        df = pd.DataFrame({"category": ["cat1", "cat2", "cat3"]})
        nw_frame = nw.from_native(df)
        var = Variable("category", nw_frame)

        expr = var.isin(["cat1", "cat2"])

        assert expr.value == ["cat1", "cat2"]
        assert expr.operator == ExpressionOp.IN

    def test_isin_with_empty_list(self):
        """Test isin() raises error with empty list."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        with pytest.raises(VariableError, match="Cannot create 'isin' expression with empty values list"):
            var.isin([])

    def test_expression_repr_for_equality(self):
        """Test that Expression has proper repr for equality."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var == 5

        assert "x" in repr(expr)
        assert "==" in repr(expr)
        assert "5" in repr(expr)

    def test_expression_repr_for_comparison(self):
        """Test that Expression has proper repr for comparison operators."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr_lt = var < 10
        expr_gt = var > 0

        assert "x" in repr(expr_lt)
        assert "<" in repr(expr_lt)
        assert "x" in repr(expr_gt)
        assert ">" in repr(expr_gt)

    def test_expression_repr_for_isin(self):
        """Test that Expression has proper repr for isin."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var.isin([1, 2, 3])

        assert "x" in repr(expr)
        assert "in" in repr(expr)

    def test_multiple_expressions_from_same_variable(self):
        """Test that multiple expressions can be created from the same variable."""
        from poffertjes.expression import Expression
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr1 = var == 5
        expr2 = var < 10
        expr3 = var.isin([1, 2, 3])

        assert all(isinstance(e, Expression) for e in [expr1, expr2, expr3])
        assert expr1.variable is var
        assert expr2.variable is var
        assert expr3.variable is var

    def test_operators_preserve_variable_reference(self):
        """Test that operators preserve the variable reference."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr = var == 5

        # The expression should reference the same variable object
        assert expr.variable is var
        assert expr.variable.name == "x"
        assert expr.variable.dataframe_id == var.dataframe_id

    def test_operators_with_none_value(self):
        """Test operators work with None value."""
        df = pd.DataFrame({"x": [1, 2, None]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        expr_eq = var == None
        expr_ne = var != None

        assert expr_eq.value is None
        assert expr_ne.value is None

    def test_isin_with_mixed_types(self):
        """Test isin() with mixed types in list."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        nw_frame = nw.from_native(df)
        var = Variable("x", nw_frame)

        # This should work - the actual validation happens at execution time
        expr = var.isin([1, "2", 3.0])

        assert expr.value == [1, "2", 3.0]
        assert expr.operator == ExpressionOp.IN



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

        with pytest.raises(DataframeError, match="Cannot create variables from an empty dataframe"):
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

        with pytest.raises(VariableError, match="Columns not found in dataframe: \\['z'\\]"):
            vb.get_variables("x", "z")

    def test_get_variables_multiple_missing_columns_error(self):
        """Test that get_variables shows all missing columns in error message."""
        import pytest
        df = pd.DataFrame({"x": [1, 2, 3]})
        vb = VariableBuilder.from_data(df)

        with pytest.raises(VariableError, match="Columns not found in dataframe"):
            vb.get_variables("a", "b", "c")

    def test_get_variables_error_shows_available_columns(self):
        """Test that error message includes available columns."""
        import pytest
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        vb = VariableBuilder.from_data(df)

        with pytest.raises(VariableError, match="Available columns"):
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

