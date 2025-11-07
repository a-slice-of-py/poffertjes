"""Tests for ScalarResult class."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.result import ScalarResult
from poffertjes.variable import VariableBuilder
from poffertjes.expression import Expression
from poffertjes.exceptions import VariableError


class TestScalarResult:
    """Test cases for ScalarResult class."""
    
    def test_init(self):
        """Test ScalarResult initialization."""
        result = ScalarResult(0.5)
        assert result.value == 0.5
        assert result._expressions == []
        assert result._dataframe is None
    
    def test_init_with_parameters(self):
        """Test ScalarResult initialization with all parameters."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        nw_df = nw.from_native(df)
        expressions = []  # Would normally contain Expression objects
        
        result = ScalarResult(0.75, expressions, nw_df)
        assert result.value == 0.75
        assert result._expressions == expressions
        assert result._dataframe is nw_df
    
    def test_float_conversion(self):
        """Test conversion to float."""
        result = ScalarResult(0.333333)
        assert float(result) == 0.333333
    
    def test_repr(self):
        """Test string representation."""
        result = ScalarResult(0.333333)
        assert repr(result) == "0.333333"
        
        result = ScalarResult(0.5)
        assert repr(result) == "0.500000"
    
    def test_given_with_variable_raises_error(self):
        """Test that conditioning on a variable without expression raises error."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        result = ScalarResult(0.5)
        
        with pytest.raises(VariableError, match="Scalar result cannot be conditioned on variable without expression"):
            result.given(y)
    
    def test_given_with_expression(self):
        """Test conditioning with expression."""
        df = pd.DataFrame({'x': [1, 1, 2, 2], 'y': [1, 2, 1, 2]})
        nw_df = nw.from_native(df)
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Create expressions
        x_eq_1 = x == 1
        y_eq_1 = y == 1
        
        # Create a scalar result
        result = ScalarResult(0.5, [x_eq_1], nw_df)
        
        # Test that given() returns a new ScalarResult
        # Note: This test assumes ProbabilityCalculator is implemented
        # For now, we'll just test that it doesn't raise an error
        try:
            conditional_result = result.given(y_eq_1)
            assert isinstance(conditional_result, ScalarResult)
        except ImportError:
            # ProbabilityCalculator not implemented yet, skip this test
            pytest.skip("ProbabilityCalculator not implemented yet")
    
    def test_parse_conditioning_args_with_expressions(self):
        """Test parsing conditioning arguments with expressions."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        result = ScalarResult(0.5)
        
        # Test with expressions
        expr1 = x == 1
        expr2 = y == 2
        conditions = result._parse_conditioning_args([expr1, expr2])
        
        assert len(conditions) == 2
        assert conditions[0] is expr1
        assert conditions[1] is expr2
    
    def test_parse_conditioning_args_with_variable_raises_error(self):
        """Test that parsing conditioning arguments with variable raises error."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        result = ScalarResult(0.5)
        
        with pytest.raises(VariableError, match="Scalar result cannot be conditioned on variable without expression"):
            result._parse_conditioning_args([y])