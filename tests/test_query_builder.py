"""Tests for QueryBuilder class."""

import pytest
import pandas as pd

from poffertjes.variable import VariableBuilder
from poffertjes.query_builder import QueryBuilder
from poffertjes.expression import Expression, CompositeExpression
from poffertjes.exceptions import VariableError, ExpressionError

# Try to import Polars, but make it optional
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class TestQueryBuilder:
    """Test cases for QueryBuilder argument parsing and query type detection."""

    @pytest.fixture
    def sample_df_pandas(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': ['a', 'b', 'c', 'd', 'e']
        })

    @pytest.fixture
    def sample_df_polars(self):
        """Create a sample Polars dataframe for testing."""
        if not HAS_POLARS:
            pytest.skip("Polars not available")
        return pl.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': ['a', 'b', 'c', 'd', 'e']
        })

    @pytest.fixture
    def variables_pandas(self, sample_df_pandas):
        """Create variables from Pandas dataframe."""
        vb = VariableBuilder.from_data(sample_df_pandas)
        return vb.get_variables('x', 'y', 'z')

    @pytest.fixture
    def variables_polars(self, sample_df_polars):
        """Create variables from Polars dataframe."""
        if not HAS_POLARS:
            pytest.skip("Polars not available")
        vb = VariableBuilder.from_data(sample_df_polars)
        return vb.get_variables('x', 'y', 'z')

    def test_init_with_single_variable(self, variables_pandas):
        """Test QueryBuilder initialization with a single variable."""
        x, y, z = variables_pandas
        
        qb = QueryBuilder((x,))
        
        assert len(qb.variables) == 1
        assert qb.variables[0] == x
        assert len(qb.expressions) == 0
        assert qb.args == (x,)

    def test_init_with_multiple_variables(self, variables_pandas):
        """Test QueryBuilder initialization with multiple variables."""
        x, y, z = variables_pandas
        
        qb = QueryBuilder((x, y, z))
        
        assert len(qb.variables) == 3
        assert x in qb.variables
        assert y in qb.variables
        assert z in qb.variables
        assert len(qb.expressions) == 0

    def test_init_with_single_expression(self, variables_pandas):
        """Test QueryBuilder initialization with a single expression."""
        x, y, z = variables_pandas
        expr = x == 5
        
        qb = QueryBuilder((expr,))
        
        assert len(qb.variables) == 1
        assert qb.variables[0] == x
        assert len(qb.expressions) == 1
        assert qb.expressions[0] == expr

    def test_init_with_multiple_expressions(self, variables_pandas):
        """Test QueryBuilder initialization with multiple expressions."""
        x, y, z = variables_pandas
        expr1 = x == 5
        expr2 = y > 20
        
        qb = QueryBuilder((expr1, expr2))
        
        assert len(qb.variables) == 2
        assert x in qb.variables
        assert y in qb.variables
        assert len(qb.expressions) == 2
        assert expr1 in qb.expressions
        assert expr2 in qb.expressions

    def test_init_with_mixed_variables_and_expressions(self, variables_pandas):
        """Test QueryBuilder initialization with both variables and expressions."""
        x, y, z = variables_pandas
        expr = x == 5
        
        qb = QueryBuilder((y, expr, z))
        
        # Should have all three variables (y, x from expr, z)
        assert len(qb.variables) == 3
        assert x in qb.variables  # From expression
        assert y in qb.variables  # Direct variable
        assert z in qb.variables  # Direct variable
        assert len(qb.expressions) == 1
        assert qb.expressions[0] == expr

    def test_init_with_composite_expression(self, variables_pandas):
        """Test QueryBuilder initialization with composite expressions."""
        x, y, z = variables_pandas
        expr1 = x == 5
        expr2 = y > 20
        composite = expr1 & expr2  # CompositeExpression
        
        qb = QueryBuilder((composite,))
        
        assert len(qb.variables) == 2
        assert x in qb.variables
        assert y in qb.variables
        assert len(qb.expressions) == 1
        assert qb.expressions[0] == composite

    def test_init_with_nested_composite_expressions(self, variables_pandas):
        """Test QueryBuilder with nested composite expressions."""
        x, y, z = variables_pandas
        expr1 = x == 5
        expr2 = y > 20
        expr3 = z == 'a'
        
        # Create nested composite: (x == 5 & y > 20) | z == 'a'
        composite1 = expr1 & expr2
        composite2 = composite1 | expr3
        
        qb = QueryBuilder((composite2,))
        
        assert len(qb.variables) == 3
        assert x in qb.variables
        assert y in qb.variables
        assert z in qb.variables
        assert len(qb.expressions) == 1

    def test_parse_args_distribution_query(self, variables_pandas):
        """Test _parse_args for distribution queries (variables only)."""
        x, y, z = variables_pandas
        
        qb = QueryBuilder((x, y))
        
        assert qb.is_distribution_query
        assert not qb.is_scalar_query
        assert len(qb.variables) == 2
        assert len(qb.expressions) == 0

    def test_parse_args_scalar_query(self, variables_pandas):
        """Test _parse_args for scalar queries (expressions only)."""
        x, y, z = variables_pandas
        expr1 = x == 5
        expr2 = y > 20
        
        qb = QueryBuilder((expr1, expr2))
        
        assert qb.is_scalar_query
        assert not qb.is_distribution_query
        assert len(qb.expressions) == 2

    def test_parse_args_duplicate_variables(self, variables_pandas):
        """Test that duplicate variables are not added twice."""
        x, y, z = variables_pandas
        expr1 = x == 5
        expr2 = x > 2  # Same variable as expr1
        
        qb = QueryBuilder((expr1, expr2))
        
        # Should only have x once in variables list
        assert len(qb.variables) == 1
        assert qb.variables[0] == x
        assert len(qb.expressions) == 2

    def test_parse_args_invalid_argument_type(self, variables_pandas):
        """Test that invalid argument types raise ExpressionError."""
        with pytest.raises(ExpressionError, match="Invalid argument type"):
            QueryBuilder(("invalid_string",))
        
        with pytest.raises(ExpressionError, match="Invalid argument type"):
            QueryBuilder((42,))

    def test_execute_distribution_query_placeholder(self, variables_pandas):
        """Test execute method for distribution queries (placeholder implementation)."""
        x, y, z = variables_pandas
        
        qb = QueryBuilder((x, y))
        result = qb.execute()
        
        # Should return placeholder string for now
        assert isinstance(result, str)
        assert "DistributionQuery" in result
        assert "variables=" in result

    def test_execute_scalar_query_placeholder(self, variables_pandas):
        """Test execute method for scalar queries (placeholder implementation)."""
        x, y, z = variables_pandas
        expr = x == 5
        
        qb = QueryBuilder((expr,))
        result = qb.execute()
        
        # Should return placeholder string for now
        assert isinstance(result, str)
        assert "ScalarQuery" in result
        assert "expressions=" in result

    def test_execute_no_variables_error(self):
        """Test that execute raises error when no variables are found."""
        # This shouldn't happen in normal usage, but test the error handling
        qb = QueryBuilder(())
        qb.variables = []  # Force empty variables list
        
        with pytest.raises(VariableError, match="No variables found in query arguments"):
            qb.execute()

    def test_query_type_properties(self, variables_pandas):
        """Test is_scalar_query and is_distribution_query properties."""
        x, y, z = variables_pandas
        
        # Distribution query
        qb_dist = QueryBuilder((x, y))
        assert qb_dist.is_distribution_query
        assert not qb_dist.is_scalar_query
        
        # Scalar query
        qb_scalar = QueryBuilder((x == 5,))
        assert qb_scalar.is_scalar_query
        assert not qb_scalar.is_distribution_query

    def test_repr(self, variables_pandas):
        """Test string representation of QueryBuilder."""
        x, y, z = variables_pandas
        
        # Distribution query
        qb_dist = QueryBuilder((x, y))
        repr_str = repr(qb_dist)
        assert "QueryBuilder" in repr_str
        assert "distribution" in repr_str
        assert "variables=2" in repr_str
        assert "expressions=0" in repr_str
        
        # Scalar query
        qb_scalar = QueryBuilder((x == 5, y > 20))
        repr_str = repr(qb_scalar)
        assert "QueryBuilder" in repr_str
        assert "scalar" in repr_str
        assert "variables=2" in repr_str
        assert "expressions=2" in repr_str

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_works_with_polars_dataframe(self, variables_polars):
        """Test that QueryBuilder works with Polars dataframes."""
        x, y, z = variables_polars
        
        # Test distribution query
        qb_dist = QueryBuilder((x, y))
        assert qb_dist.is_distribution_query
        assert len(qb_dist.variables) == 2
        
        # Test scalar query
        expr = x == 3
        qb_scalar = QueryBuilder((expr,))
        assert qb_scalar.is_scalar_query
        assert len(qb_scalar.expressions) == 1

    def test_extract_variables_from_composite_complex(self, variables_pandas):
        """Test complex composite expression variable extraction."""
        x, y, z = variables_pandas
        
        # Create complex nested expression: ((x == 1) & (y > 10)) | ((z == 'a') & (x < 5))
        expr1 = x == 1
        expr2 = y > 10
        expr3 = z == 'a'
        expr4 = x < 5
        
        composite1 = expr1 & expr2
        composite2 = expr3 & expr4
        final_composite = composite1 | composite2
        
        qb = QueryBuilder((final_composite,))
        
        # Should extract all three variables
        assert len(qb.variables) == 3
        assert x in qb.variables
        assert y in qb.variables
        assert z in qb.variables

    def test_argument_order_preservation(self, variables_pandas):
        """Test that argument order is preserved in args tuple."""
        x, y, z = variables_pandas
        expr = x == 5
        
        qb = QueryBuilder((y, expr, z))
        
        assert qb.args == (y, expr, z)
        # Variables list order may differ due to parsing logic, but args should be preserved

    def test_empty_args_tuple(self):
        """Test QueryBuilder with empty arguments tuple."""
        qb = QueryBuilder(())
        
        assert len(qb.variables) == 0
        assert len(qb.expressions) == 0
        assert qb.args == ()
        
        # Should raise error on execute
        with pytest.raises(VariableError, match="No variables found"):
            qb.execute()


class TestQueryBuilderIntegration:
    """Integration tests for QueryBuilder with real dataframe operations."""

    def test_dataframe_access(self):
        """Test that QueryBuilder can access the underlying dataframe."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        qb = QueryBuilder((x,))
        
        # Should be able to access the Narwhals frame
        assert qb.variables[0]._nw_frame is not None
        assert len(qb.variables[0]._nw_frame) == 3

    def test_variable_dataframe_consistency(self):
        """Test that all variables in a query come from the same dataframe."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        qb = QueryBuilder((x, y, x == 1, y > 4))
        
        # All variables should have the same dataframe ID
        frame_ids = [var.dataframe_id for var in qb.variables]
        assert len(set(frame_ids)) == 1  # All should be the same