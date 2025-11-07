"""Tests for exception hierarchy and error handling."""

import pytest
import pandas as pd
from poffertjes.exceptions import (
    PoffertjesError,
    DataframeError,
    VariableError,
    ExpressionError,
    ProbabilityError,
)
from poffertjes.variable import VariableBuilder
from poffertjes.expression import Expression, TernaryExpression, CompositeExpression
from poffertjes.p_interface import p


class TestExceptionHierarchy:
    """Test the exception hierarchy structure."""
    
    def test_base_exception(self):
        """Test that PoffertjesError is the base exception."""
        exc = PoffertjesError("test message")
        assert str(exc) == "test message"
        assert isinstance(exc, Exception)
    
    def test_dataframe_error_inheritance(self):
        """Test that DataframeError inherits from PoffertjesError."""
        exc = DataframeError("dataframe error")
        assert isinstance(exc, PoffertjesError)
        assert isinstance(exc, Exception)
        assert str(exc) == "dataframe error"
    
    def test_variable_error_inheritance(self):
        """Test that VariableError inherits from PoffertjesError."""
        exc = VariableError("variable error")
        assert isinstance(exc, PoffertjesError)
        assert isinstance(exc, Exception)
        assert str(exc) == "variable error"
    
    def test_expression_error_inheritance(self):
        """Test that ExpressionError inherits from PoffertjesError."""
        exc = ExpressionError("expression error")
        assert isinstance(exc, PoffertjesError)
        assert isinstance(exc, Exception)
        assert str(exc) == "expression error"
    
    def test_probability_error_inheritance(self):
        """Test that ProbabilityError inherits from PoffertjesError."""
        exc = ProbabilityError("probability error")
        assert isinstance(exc, PoffertjesError)
        assert isinstance(exc, Exception)
        assert str(exc) == "probability error"
    
    def test_exception_catching(self):
        """Test that specific exceptions can be caught as PoffertjesError."""
        # Test that we can catch all poffertjes exceptions with the base class
        with pytest.raises(PoffertjesError):
            raise DataframeError("test")
        
        with pytest.raises(PoffertjesError):
            raise VariableError("test")
        
        with pytest.raises(PoffertjesError):
            raise ExpressionError("test")
        
        with pytest.raises(PoffertjesError):
            raise ProbabilityError("test")
    
    def test_exception_types_are_distinct(self):
        """Test that different exception types are distinct."""
        dataframe_exc = DataframeError("test")
        variable_exc = VariableError("test")
        expression_exc = ExpressionError("test")
        probability_exc = ProbabilityError("test")
        
        assert type(dataframe_exc) != type(variable_exc)
        assert type(variable_exc) != type(expression_exc)
        assert type(expression_exc) != type(probability_exc)
        assert type(probability_exc) != type(dataframe_exc)


class TestVariableErrorHandling:
    """Test error handling in Variable and VariableBuilder classes."""
    
    def test_empty_dataframe_error(self):
        """Test that empty dataframe raises DataframeError."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(DataframeError, match="Cannot create variables from an empty dataframe"):
            VariableBuilder.from_data(empty_df)
    
    def test_missing_column_error(self):
        """Test that missing column raises VariableError."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        
        with pytest.raises(VariableError, match="Columns not found in dataframe: \\['z'\\]"):
            vb.get_variables('z')
        
        with pytest.raises(VariableError, match="Available columns: \\['x', 'y'\\]"):
            vb.get_variables('missing1', 'missing2')
    
    def test_empty_isin_values_error(self):
        """Test that empty values list in isin raises VariableError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        with pytest.raises(VariableError, match="Cannot create 'isin' expression with empty values list"):
            x.isin([])


class TestExpressionErrorHandling:
    """Test error handling in Expression classes."""
    
    def test_unsupported_operator_error(self):
        """Test that unsupported operator raises ExpressionError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Create expression with invalid operator
        expr = Expression(x, "==", 5)
        expr.operator = "INVALID"  # Manually set invalid operator
        
        with pytest.raises(ExpressionError, match="Unsupported operator"):
            expr.to_narwhals_expr()
    
    def test_empty_in_values_error(self):
        """Test that empty list in 'in' operator raises ExpressionError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Create expression with empty list
        expr = Expression(x, "in", [])
        
        with pytest.raises(ExpressionError, match="Cannot use 'in' operator with empty list"):
            expr.to_narwhals_expr()
    
    def test_ternary_expression_invalid_closed_error(self):
        """Test that invalid closed parameter raises ExpressionError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        with pytest.raises(ExpressionError, match="closed must be one of"):
            TernaryExpression(x, 1, 5, closed="invalid")
    
    def test_ternary_expression_invalid_bounds_error(self):
        """Test that invalid bounds raise ExpressionError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        with pytest.raises(ExpressionError, match="Lower bound \\(5\\) must be less than upper bound \\(1\\)"):
            TernaryExpression(x, 5, 1)  # lower > upper
    
    def test_composite_expression_invalid_logic_error(self):
        """Test that invalid logic raises ExpressionError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        expr1 = x == 1
        expr2 = x == 2
        
        with pytest.raises(ExpressionError, match="Logic must be 'AND' or 'OR'"):
            CompositeExpression([expr1, expr2], "INVALID")
    
    def test_composite_expression_empty_expressions_error(self):
        """Test that empty expressions list raises ExpressionError."""
        with pytest.raises(ExpressionError, match="CompositeExpression requires at least one expression"):
            CompositeExpression([], "AND")


class TestPInterfaceErrorHandling:
    """Test error handling in P interface."""
    
    def test_no_arguments_error(self):
        """Test that p() with no arguments raises VariableError."""
        with pytest.raises(VariableError, match="p\\(\\) requires at least one argument"):
            p()
    
    def test_mixed_dataframes_error(self):
        """Test that mixing variables from different dataframes raises DataframeError."""
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'y': [4, 5, 6]})
        
        vb1 = VariableBuilder.from_data(df1)
        vb2 = VariableBuilder.from_data(df2)
        
        x = vb1.get_variables('x')
        y = vb2.get_variables('y')
        
        with pytest.raises(DataframeError, match="Variables from different dataframes cannot be mixed"):
            p(x, y)


class TestResultErrorHandling:
    """Test error handling in result classes."""
    
    def test_scalar_result_variable_conditioning_error(self):
        """Test that conditioning scalar result on variable without expression raises VariableError."""
        from poffertjes.result import ScalarResult
        
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Create a scalar result
        scalar_result = ScalarResult(0.5, [x == 1], None)
        
        with pytest.raises(VariableError, match="Scalar result cannot be conditioned on variable without expression"):
            scalar_result.given(y)  # Should be y == value
    
    def test_distribution_result_invalid_conditioning_error(self):
        """Test that invalid conditioning argument raises VariableError."""
        from poffertjes.result import DistributionResult
        
        df = pd.DataFrame({'x': [1, 2, 3]})
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Create a distribution result
        dist_result = DistributionResult(None, [x], None)
        
        with pytest.raises(VariableError, match="Invalid conditioning argument"):
            dist_result.given("invalid_string")