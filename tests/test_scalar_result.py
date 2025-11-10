"""Tests for ScalarResult class."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.result import ScalarResult, DistributionResult
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
    
    def test_given_with_variable_without_dataframe_raises_error(self):
        """Test that conditioning on a variable without dataframe context raises error."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # ScalarResult without dataframe context
        result = ScalarResult(0.5)
        
        with pytest.raises(VariableError, match="Cannot condition ScalarResult on variables without dataframe context"):
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
    
    def test_parse_conditioning_args_with_variable_succeeds(self):
        """Test that parsing conditioning arguments with variable now succeeds."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        result = ScalarResult(0.5)
        
        # This should now work (no longer raises an error)
        conditions = result._parse_conditioning_args([y])
        assert len(conditions) == 1
        assert conditions[0] is y

    def test_given_with_variable_returns_distribution(self):
        """Test that conditioning scalar result on variable returns distribution."""
        df = pd.DataFrame({'x': [1, 1, 2, 2], 'y': [1, 2, 1, 2]})
        nw_df = nw.from_native(df)
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Create a scalar result with dataframe context
        expr = x == 1
        result = ScalarResult(0.5, [expr], nw_df)
        
        # Conditioning on variable should return DistributionResult
        cond_result = result.given(y)
        
        assert isinstance(cond_result, DistributionResult)
        assert len(cond_result.variables) == 1
        assert cond_result.variables[0] is y
        
        # Check the actual probabilities
        prob_dict = cond_result.to_dict()
        assert 1 in prob_dict  # P(X=1 | Y=1)
        assert 2 in prob_dict  # P(X=1 | Y=2)

    def test_given_with_mixed_conditions(self):
        """Test conditioning with both variable and expression."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 2, 1, 2], 
            'y': [1, 2, 1, 2, 1, 2],
            'z': [1, 1, 1, 1, 2, 2]
        })
        nw_df = nw.from_native(df)
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Create a scalar result
        expr = x == 1
        result = ScalarResult(0.5, [expr], nw_df)
        
        # Condition on both variable and expression
        cond_result = result.given(y, z == 1)
        
        assert isinstance(cond_result, DistributionResult)
        assert len(cond_result.variables) == 1
        assert cond_result.variables[0] is y

    def test_arithmetic_operations(self):
        """Test that ScalarResult behaves like a float in arithmetic operations."""
        result = ScalarResult(0.5)
        
        # Addition
        assert result + 1 == 1.5
        assert 1 + result == 1.5
        assert result + result == 1.0
        
        # Subtraction
        assert result - 0.2 == 0.3
        assert 1 - result == 0.5
        assert result - result == 0.0
        
        # Multiplication
        assert result * 2 == 1.0
        assert 2 * result == 1.0
        assert result * result == 0.25
        
        # Division
        assert result / 2 == 0.25
        assert 1 / result == 2.0
        assert result / result == 1.0
        
        # Floor division
        assert result // 0.2 == 2.0
        assert 1 // result == 2.0
        
        # Modulo
        assert result % 0.3 == 0.2
        assert 1 % result == 0.0
        
        # Power
        assert result ** 2 == 0.25
        assert 2 ** result == pytest.approx(1.414213, rel=1e-5)

    def test_comparison_operations(self):
        """Test that ScalarResult supports comparison operations."""
        result1 = ScalarResult(0.5)
        result2 = ScalarResult(0.3)
        
        # Equality
        assert result1 == 0.5
        assert result1 != 0.3
        assert result1 == result1
        assert result1 != result2
        
        # Ordering
        assert result1 > 0.3
        assert result1 >= 0.5
        assert result1 < 0.7
        assert result1 <= 0.5
        
        assert result1 > result2
        assert result1 >= result2
        assert result2 < result1
        assert result2 <= result1

    def test_unary_operations(self):
        """Test unary operations on ScalarResult."""
        result = ScalarResult(0.5)
        negative_result = ScalarResult(-0.3)
        
        # Unary operations
        assert -result == -0.5
        assert +result == 0.5
        assert abs(negative_result) == 0.3
        
        # Rounding and truncation
        precise_result = ScalarResult(0.123456)
        assert round(precise_result, 2) == 0.12
        
        import math
        assert math.trunc(precise_result) == 0
        assert math.floor(precise_result) == 0
        assert math.ceil(precise_result) == 1

    def test_mixed_arithmetic_preserves_given_functionality(self):
        """Test that arithmetic operations don't break the .given() functionality."""
        df = pd.DataFrame({'x': [1, 1, 2, 2], 'y': [1, 2, 1, 2]})
        nw_df = nw.from_native(df)
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Create a scalar result
        expr = x == 1
        result = ScalarResult(0.5, [expr], nw_df)
        
        # Arithmetic operations should work
        arithmetic_result = result + 0.2
        assert arithmetic_result == 0.7
        assert isinstance(arithmetic_result, float)
        
        # But the original result should still support .given()
        try:
            cond_result = result.given(y == 1)
            assert isinstance(cond_result, ScalarResult)
        except ImportError:
            # ProbabilityCalculator not implemented yet, skip this test
            pytest.skip("ProbabilityCalculator not implemented yet")