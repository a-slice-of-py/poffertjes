"""Tests for DistributionResult class."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.result import DistributionResult
from poffertjes.variable import VariableBuilder
from poffertjes.exceptions import VariableError


class TestDistributionResult:
    """Test cases for DistributionResult class."""

    def test_init(self):
        """Test DistributionResult initialization."""
        df = pd.DataFrame({"x": [1, 2, 3], "probability": [0.3, 0.4, 0.3]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3]})
        vb = VariableBuilder.from_data(source_df)
        x = vb.get_variables("x")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        assert result.distribution is nw_df
        assert result.variables == [x]
        assert result._conditions == []

    def test_init_with_conditions(self):
        """Test DistributionResult initialization with conditions."""
        df = pd.DataFrame({"x": [1, 2], "probability": [0.6, 0.4]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]})
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        condition = y == 1
        result = DistributionResult(nw_df, [x], nw.from_native(source_df), [condition])

        assert result._conditions == [condition]

    def test_given_with_expression(self):
        """Test conditioning with expression."""
        df = pd.DataFrame({"x": [1, 2], "probability": [0.5, 0.5]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]})
        nw_source = nw.from_native(source_df)
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        result = DistributionResult(nw_df, [x], nw_source)

        # Test that given() returns a new DistributionResult
        # Note: This test assumes ProbabilityCalculator is implemented
        try:
            conditional_result = result.given(y == 1)
            assert isinstance(conditional_result, DistributionResult)
        except ImportError:
            # ProbabilityCalculator not implemented yet, skip this test
            pytest.skip("ProbabilityCalculator not implemented yet")

    def test_given_with_variable(self):
        """Test conditioning with variable expression."""
        # Create test data with clear conditional relationships
        source_df = pd.DataFrame(
            {"x": [1, 1, 2, 2, 3, 3, 1, 2, 3], "y": [1, 1, 1, 2, 2, 2, 2, 2, 1]}
        )
        nw_source = nw.from_native(source_df)
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        # Create initial distribution (marginal P(X))
        initial_dist_df = pd.DataFrame(
            {"x": [1, 2, 3], "count": [3, 3, 3], "probability": [1 / 3, 1 / 3, 1 / 3]}
        )
        nw_initial = nw.from_native(initial_dist_df)

        result = DistributionResult(nw_initial, [x], nw_source)

        # Test conditioning with a variable expression P(X | Y=1)
        # This should return a conditional distribution P(X | Y=1)
        conditional_result = result.given(y == 1)

        # Verify it returns a DistributionResult
        assert isinstance(conditional_result, DistributionResult)

        # The result should have the same variables as the original
        assert conditional_result.variables == [x]

        # The conditioning expression should be in the conditions
        assert len(conditional_result._conditions) == 1
        assert conditional_result._conditions[0].variable is y
        assert conditional_result._conditions[0].value == 1

        # Verify the result has valid probability distribution
        result_dict = conditional_result.to_dict()
        assert len(result_dict) > 0

        # All probabilities should be non-negative
        for prob in result_dict.values():
            assert prob >= 0.0

        # Probabilities should sum to 1.0 (within floating point precision)
        total_prob = sum(result_dict.values())
        assert abs(total_prob - 1.0) < 1e-10

        # Test conditioning with multiple expressions
        multi_conditional_result = result.given(y == 1, x != 3)
        assert isinstance(multi_conditional_result, DistributionResult)
        assert len(multi_conditional_result._conditions) == 2

        # Verify the multi-conditional result is valid
        multi_dict = multi_conditional_result.to_dict()
        assert len(multi_dict) > 0
        multi_total_prob = sum(multi_dict.values())
        assert abs(multi_total_prob - 1.0) < 1e-10

    def test_parse_conditioning_args_with_expressions(self):
        """Test parsing conditioning arguments with expressions."""
        df = pd.DataFrame({"x": [1, 2], "probability": [0.5, 0.5]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        # Test with expressions
        expr1 = x == 1
        expr2 = y == 2
        conditions = result._parse_conditioning_args([expr1, expr2])

        assert len(conditions) == 2
        assert conditions[0] is expr1
        assert conditions[1] is expr2

    def test_parse_conditioning_args_with_variables(self):
        """Test parsing conditioning arguments with variables."""
        df = pd.DataFrame({"x": [1, 2], "probability": [0.5, 0.5]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        # Test with variables
        conditions = result._parse_conditioning_args([y])

        assert len(conditions) == 1
        assert conditions[0] is y

    def test_parse_conditioning_args_with_invalid_type(self):
        """Test that parsing conditioning arguments with invalid type raises error."""
        df = pd.DataFrame({"x": [1, 2], "probability": [0.5, 0.5]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 2]})
        vb = VariableBuilder.from_data(source_df)
        x = vb.get_variables("x")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        with pytest.raises(VariableError, match="Invalid conditioning argument"):
            result._parse_conditioning_args(["invalid"])

    def test_to_dict_fallback(self):
        """Test to_dict fallback implementation."""
        # Create a simple distribution dataframe
        df = pd.DataFrame({"x": [1, 2, 3], "probability": [0.3, 0.4, 0.3]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3]})
        vb = VariableBuilder.from_data(source_df)
        x = vb.get_variables("x")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        # Test to_dict
        result_dict = result.to_dict()
        expected = {1: 0.3, 2: 0.4, 3: 0.3}
        assert result_dict == expected

    def test_to_dict_multivariate(self):
        """Test to_dict with multiple variables."""
        # Create a joint distribution dataframe
        df = pd.DataFrame(
            {
                "x": [1, 1, 2, 2],
                "y": [1, 2, 1, 2],
                "probability": [0.25, 0.25, 0.25, 0.25],
            }
        )
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]})
        vb = VariableBuilder.from_data(source_df)
        x, y = vb.get_variables("x", "y")

        result = DistributionResult(nw_df, [x, y], nw.from_native(source_df))

        # Test to_dict
        result_dict = result.to_dict()
        expected = {(1, 1): 0.25, (1, 2): 0.25, (2, 1): 0.25, (2, 2): 0.25}
        assert result_dict == expected

    def test_to_dataframe(self):
        """Test to_dataframe method."""
        df = pd.DataFrame({"x": [1, 2, 3], "probability": [0.3, 0.4, 0.3]})
        nw_df = nw.from_native(df)

        source_df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3]})
        vb = VariableBuilder.from_data(source_df)
        x = vb.get_variables("x")

        result = DistributionResult(nw_df, [x], nw.from_native(source_df))

        # Test to_dataframe
        native_df = result.to_dataframe()

        # Should return the original pandas dataframe
        pd.testing.assert_frame_equal(native_df, df)
