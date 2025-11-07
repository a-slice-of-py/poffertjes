"""End-to-end integration tests."""

import pytest
import pandas as pd
import polars as pl
from poffertjes import p
from poffertjes.variable import VariableBuilder
from poffertjes.result import ScalarResult, DistributionResult


class TestMarginalProbabilityWorkflows:
    """Test end-to-end marginal probability workflows.
    
    Requirements tested:
    - 1.1: Support Pandas dataframes
    - 1.2: Support Polars dataframes  
    - 4.1: Return probability distribution for p(x)
    - 4.2: Return scalar probability for p(x == value)
    - 4.3: Include all observed values and their probabilities
    - 4.4: Probabilities sum to 1.0
    """

    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 2, 2, 3, 3, 3],
            'y': ['a', 'a', 'b', 'b', 'c', 'c'],
            'z': [10.5, 20.5, 20.5, 30.5, 30.5, 30.5]
        })

    @pytest.fixture
    def sample_polars_df(self):
        """Create a sample Polars dataframe for testing."""
        return pl.DataFrame({
            'x': [1, 2, 2, 3, 3, 3],
            'y': ['a', 'a', 'b', 'b', 'c', 'c'],
            'z': [10.5, 20.5, 20.5, 30.5, 30.5, 30.5]
        })

    def test_marginal_distribution_pandas(self, sample_pandas_df):
        """Test p(x) returning distributions with Pandas.
        
        Requirements: 1.1, 4.1, 4.3, 4.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Check all values are present with correct probabilities
        # x=1 appears 1/6 times, x=2 appears 2/6 times, x=3 appears 3/6 times
        expected = {1: 1/6, 2: 2/6, 3: 3/6}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_marginal_distribution_polars(self, sample_polars_df):
        """Test p(x) returning distributions with Polars.
        
        Requirements: 1.2, 4.1, 4.3, 4.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Check all values are present with correct probabilities
        expected = {1: 1/6, 2: 2/6, 3: 3/6}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_marginal_scalar_pandas(self, sample_pandas_df):
        """Test p(x == value) returning scalars with Pandas.
        
        Requirements: 1.1, 4.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test scalar probabilities
        result1 = p(x == 1)
        result2 = p(x == 2) 
        result3 = p(x == 3)
        result4 = p(x == 4)  # Non-existent value
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        assert isinstance(result4, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/6) < 1e-10
        assert abs(float(result2) - 2/6) < 1e-10
        assert abs(float(result3) - 3/6) < 1e-10
        assert abs(float(result4) - 0.0) < 1e-10

    def test_marginal_scalar_polars(self, sample_polars_df):
        """Test p(x == value) returning scalars with Polars.
        
        Requirements: 1.2, 4.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x = vb.get_variables('x')
        
        # Test scalar probabilities
        result1 = p(x == 1)
        result2 = p(x == 2)
        result3 = p(x == 3)
        result4 = p(x == 4)  # Non-existent value
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        assert isinstance(result4, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/6) < 1e-10
        assert abs(float(result2) - 2/6) < 1e-10
        assert abs(float(result3) - 3/6) < 1e-10
        assert abs(float(result4) - 0.0) < 1e-10

    def test_categorical_distribution_pandas(self, sample_pandas_df):
        """Test marginal distribution with categorical variables using Pandas.
        
        Requirements: 1.1, 4.1, 4.3, 4.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        y = vb.get_variables('y')
        
        # Test marginal distribution
        result = p(y)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Check all values are present with correct probabilities
        # y='a' appears 2/6 times, y='b' appears 2/6 times, y='c' appears 2/6 times
        expected = {'a': 2/6, 'b': 2/6, 'c': 2/6}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_float_distribution_pandas(self, sample_pandas_df):
        """Test marginal distribution with float variables using Pandas.
        
        Requirements: 1.1, 4.1, 4.3, 4.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        z = vb.get_variables('z')
        
        # Test marginal distribution
        result = p(z)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Check all values are present with correct probabilities
        # z=10.5 appears 1/6 times, z=20.5 appears 2/6 times, z=30.5 appears 3/6 times
        expected = {10.5: 1/6, 20.5: 2/6, 30.5: 3/6}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_comparison_operators_pandas(self, sample_pandas_df):
        """Test comparison operators with Pandas.
        
        Requirements: 1.1, 4.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test various comparison operators
        result_lt = p(x < 3)    # x=1,2,2 -> 3/6
        result_le = p(x <= 2)   # x=1,2,2 -> 3/6
        result_gt = p(x > 1)    # x=2,2,3,3,3 -> 5/6
        result_ge = p(x >= 2)   # x=2,2,3,3,3 -> 5/6
        result_ne = p(x != 2)   # x=1,3,3,3 -> 4/6
        
        # Check results
        assert abs(float(result_lt) - 3/6) < 1e-10
        assert abs(float(result_le) - 3/6) < 1e-10
        assert abs(float(result_gt) - 5/6) < 1e-10
        assert abs(float(result_ge) - 5/6) < 1e-10
        assert abs(float(result_ne) - 4/6) < 1e-10

    def test_isin_operator_pandas(self, sample_pandas_df):
        """Test isin operator with categorical variables using Pandas.
        
        Requirements: 1.1, 4.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        y = vb.get_variables('y')
        
        # Test isin operator
        result1 = p(y.isin(['a', 'b']))  # Should be 4/6
        result2 = p(y.isin(['c']))       # Should be 2/6
        result3 = p(y.isin(['d']))       # Should be 0/6
        
        # Check results
        assert abs(float(result1) - 4/6) < 1e-10
        assert abs(float(result2) - 2/6) < 1e-10
        assert abs(float(result3) - 0.0) < 1e-10


class TestConditionalProbabilityWorkflows:
    """Test end-to-end conditional probability workflows.
    
    Requirements tested:
    - 5.1: Return conditional probability distribution P(X|Y)
    - 5.2: Return scalar conditional probability P(X=value1|Y=value2)
    - 5.3: Return distribution of x conditioned on y equals value
    - 5.4: Return P(X|Y,Z) for multiple conditioning variables
    - 5.5: Return P(X|Y=value1 AND Z=value2) for multiple conditions
    - 5.6: Return P(X=x_i|Y=y_j) for all combinations
    - 5.7: Raise clear error when conditioning event has zero occurrences
    - 5.8: Conditional probabilities sum to 1.0
    """

    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 1, 2, 2, 3, 3, 3, 3],
            'y': [1, 2, 1, 2, 1, 1, 2, 2],
            'z': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']
        })

    @pytest.fixture
    def sample_polars_df(self):
        """Create a sample Polars dataframe for testing."""
        return pl.DataFrame({
            'x': [1, 1, 2, 2, 3, 3, 3, 3],
            'y': [1, 2, 1, 2, 1, 1, 2, 2],
            'z': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']
        })

    def test_conditional_distribution_pandas(self, sample_pandas_df):
        """Test p(x).given(y == value) returning conditional distributions with Pandas.
        
        Requirements: 5.1, 5.3, 5.8
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditional distribution P(X | Y=1)
        result = p(x).given(y == 1)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # When Y=1, we have rows: (1,1), (2,1), (3,1), (3,1)
        # So P(X=1|Y=1) = 1/4, P(X=2|Y=1) = 1/4, P(X=3|Y=1) = 2/4
        expected = {1: 1/4, 2: 1/4, 3: 2/4}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_conditional_distribution_polars(self, sample_polars_df):
        """Test p(x).given(y == value) returning conditional distributions with Polars.
        
        Requirements: 5.1, 5.3, 5.8
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditional distribution P(X | Y=2)
        result = p(x).given(y == 2)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # When Y=2, we have rows: (1,2), (2,2), (3,2), (3,2)
        # So P(X=1|Y=2) = 1/4, P(X=2|Y=2) = 1/4, P(X=3|Y=2) = 2/4
        expected = {1: 1/4, 2: 1/4, 3: 2/4}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_conditional_scalar_pandas(self, sample_pandas_df):
        """Test p(x == value1).given(y == value2) returning scalar conditional probabilities with Pandas.
        
        Requirements: 5.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test scalar conditional probabilities
        result1 = p(x == 1).given(y == 1)  # P(X=1|Y=1) = 1/4
        result2 = p(x == 3).given(y == 1)  # P(X=3|Y=1) = 2/4
        result3 = p(x == 1).given(y == 2)  # P(X=1|Y=2) = 1/4
        result4 = p(x == 4).given(y == 1)  # P(X=4|Y=1) = 0/4 (non-existent)
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        assert isinstance(result4, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/4) < 1e-10
        assert abs(float(result2) - 2/4) < 1e-10
        assert abs(float(result3) - 1/4) < 1e-10
        assert abs(float(result4) - 0.0) < 1e-10

    def test_multiple_conditioning_variables_pandas(self, sample_pandas_df):
        """Test p(x).given(y == value1, z == value2) with multiple conditioning variables using Pandas.
        
        Requirements: 5.4, 5.5
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Test conditional distribution P(X | Y=1, Z='a')
        result = p(x).given(y == 1, z == 'a')
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # When Y=1 AND Z='a', we have rows: (1,1,'a'), (3,1,'a')
        # So P(X=1|Y=1,Z='a') = 1/2, P(X=3|Y=1,Z='a') = 1/2
        expected = {1: 1/2, 3: 1/2}
        
        assert len(dist_dict) == 2
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_multiple_conditioning_scalar_pandas(self, sample_pandas_df):
        """Test p(x == value).given(y == value1, z == value2) with multiple conditions using Pandas.
        
        Requirements: 5.5
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Test scalar conditional probabilities with multiple conditions
        result1 = p(x == 1).given(y == 1, z == 'a')  # P(X=1|Y=1,Z='a') = 1/2
        result2 = p(x == 3).given(y == 1, z == 'a')  # P(X=3|Y=1,Z='a') = 1/2
        result3 = p(x == 2).given(y == 1, z == 'a')  # P(X=2|Y=1,Z='a') = 0/2
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/2) < 1e-10
        assert abs(float(result2) - 1/2) < 1e-10
        assert abs(float(result3) - 0.0) < 1e-10

    def test_zero_probability_conditioning_pandas(self, sample_pandas_df):
        """Test error handling when conditioning event has zero probability using Pandas.
        
        Requirements: 5.7
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditioning on non-existent value
        with pytest.raises(Exception) as exc_info:
            p(x).given(y == 999)  # Y=999 doesn't exist
        
        # Should raise a clear error about zero probability
        assert "zero probability" in str(exc_info.value).lower()

    def test_conditional_with_comparison_operators_pandas(self, sample_pandas_df):
        """Test conditional probabilities with comparison operators using Pandas.
        
        Requirements: 5.2, 5.3
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditional distribution P(X | Y > 1)
        result_dist = p(x).given(y > 1)
        
        # Should return DistributionResult
        assert isinstance(result_dist, DistributionResult)
        
        # When Y > 1 (i.e., Y=2), we have rows: (1,2), (2,2), (3,2), (3,2)
        dist_dict = result_dist.to_dict()
        expected = {1: 1/4, 2: 1/4, 3: 2/4}
        
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Test scalar conditional probability P(X=3 | Y > 1)
        result_scalar = p(x == 3).given(y > 1)
        assert isinstance(result_scalar, ScalarResult)
        assert abs(float(result_scalar) - 2/4) < 1e-10

    def test_conditional_with_isin_pandas(self, sample_pandas_df):
        """Test conditional probabilities with isin operator using Pandas.
        
        Requirements: 5.2, 5.3
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, z = vb.get_variables('x', 'z')
        
        # Test conditional distribution P(X | Z in ['a'])
        result = p(x).given(z.isin(['a']))
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # When Z='a', we have rows: (1,'a'), (1,'a'), (3,'a'), (3,'a')
        dist_dict = result.to_dict()
        expected = {1: 2/4, 3: 2/4}
        
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10


class TestJointProbabilityWorkflows:
    """Test end-to-end joint probability workflows.
    
    Requirements tested:
    - 11.1: Return joint probability distribution for multiple variables
    - 11.2: Return P(X=value1 AND Y=value2) when conditions are provided
    - 11.3: Include all observed combinations of values
    - 11.4: Support conditional joint distributions P(X,Y|Z)
    - 11.5: Joint probabilities sum to 1.0 across all combinations
    """

    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 1, 2, 2, 3, 3],
            'y': [1, 2, 1, 2, 1, 2],
            'z': ['a', 'a', 'b', 'b', 'a', 'b']
        })

    @pytest.fixture
    def sample_polars_df(self):
        """Create a sample Polars dataframe for testing."""
        return pl.DataFrame({
            'x': [1, 1, 2, 2, 3, 3],
            'y': [1, 2, 1, 2, 1, 2],
            'z': ['a', 'a', 'b', 'b', 'a', 'b']
        })

    def test_joint_distribution_pandas(self, sample_pandas_df):
        """Test p(x, y) returning joint distributions with Pandas.
        
        Requirements: 11.1, 11.3, 11.5
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution P(X, Y)
        result = p(x, y)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Each combination appears once out of 6 total rows
        # Expected combinations: (1,1), (1,2), (2,1), (2,2), (3,1), (3,2)
        expected = {
            (1, 1): 1/6, (1, 2): 1/6, (2, 1): 1/6,
            (2, 2): 1/6, (3, 1): 1/6, (3, 2): 1/6
        }
        
        assert len(dist_dict) == 6
        for combination, expected_prob in expected.items():
            assert abs(dist_dict[combination] - expected_prob) < 1e-10
        
        # Joint probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_joint_distribution_polars(self, sample_polars_df):
        """Test p(x, y) returning joint distributions with Polars.
        
        Requirements: 11.1, 11.3, 11.5
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution P(X, Y)
        result = p(x, y)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Each combination appears once out of 6 total rows
        expected = {
            (1, 1): 1/6, (1, 2): 1/6, (2, 1): 1/6,
            (2, 2): 1/6, (3, 1): 1/6, (3, 2): 1/6
        }
        
        assert len(dist_dict) == 6
        for combination, expected_prob in expected.items():
            assert abs(dist_dict[combination] - expected_prob) < 1e-10
        
        # Joint probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_joint_scalar_pandas(self, sample_pandas_df):
        """Test p(x == v1, y == v2) returning scalar joint probabilities with Pandas.
        
        Requirements: 11.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test scalar joint probabilities
        result1 = p(x == 1, y == 1)  # P(X=1, Y=1) = 1/6
        result2 = p(x == 2, y == 2)  # P(X=2, Y=2) = 1/6
        result3 = p(x == 1, y == 3)  # P(X=1, Y=3) = 0/6 (non-existent)
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/6) < 1e-10
        assert abs(float(result2) - 1/6) < 1e-10
        assert abs(float(result3) - 0.0) < 1e-10

    def test_three_variable_joint_pandas(self, sample_pandas_df):
        """Test p(x, y, z) with three variables using Pandas.
        
        Requirements: 11.1, 11.3, 11.5
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Test three-variable joint distribution P(X, Y, Z)
        result = p(x, y, z)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # Each combination appears once out of 6 total rows
        expected = {
            (1, 1, 'a'): 1/6, (1, 2, 'a'): 1/6, (2, 1, 'b'): 1/6,
            (2, 2, 'b'): 1/6, (3, 1, 'a'): 1/6, (3, 2, 'b'): 1/6
        }
        
        assert len(dist_dict) == 6
        for combination, expected_prob in expected.items():
            assert abs(dist_dict[combination] - expected_prob) < 1e-10
        
        # Joint probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_conditional_joint_distribution_pandas(self, sample_pandas_df):
        """Test p(x, y).given(z == value) for conditional joint distributions using Pandas.
        
        Requirements: 11.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Test conditional joint distribution P(X, Y | Z='a')
        result = p(x, y).given(z == 'a')
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # When Z='a', we have rows: (1,1,'a'), (1,2,'a'), (3,1,'a')
        # So P(X,Y|Z='a') for each combination is 1/3
        expected = {(1, 1): 1/3, (1, 2): 1/3, (3, 1): 1/3}
        
        assert len(dist_dict) == 3
        for combination, expected_prob in expected.items():
            assert abs(dist_dict[combination] - expected_prob) < 1e-10
        
        # Conditional joint probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_conditional_joint_scalar_pandas(self, sample_pandas_df):
        """Test p(x == v1, y == v2).given(z == value) for conditional joint scalars using Pandas.
        
        Requirements: 11.4
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Test conditional joint scalar probabilities
        result1 = p(x == 1, y == 1).given(z == 'a')  # P(X=1,Y=1|Z='a') = 1/3
        result2 = p(x == 1, y == 2).given(z == 'a')  # P(X=1,Y=2|Z='a') = 1/3
        result3 = p(x == 2, y == 1).given(z == 'a')  # P(X=2,Y=1|Z='a') = 0/3
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        assert isinstance(result3, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 1/3) < 1e-10
        assert abs(float(result2) - 1/3) < 1e-10
        assert abs(float(result3) - 0.0) < 1e-10

    def test_joint_with_comparison_operators_pandas(self, sample_pandas_df):
        """Test joint probabilities with comparison operators using Pandas.
        
        Requirements: 11.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint probability with comparison operators
        result1 = p(x > 1, y == 1)  # P(X>1, Y=1) -> rows (2,1), (3,1) = 2/6
        result2 = p(x <= 2, y >= 2)  # P(X<=2, Y>=2) -> rows (1,2), (2,2) = 2/6
        
        # Should return ScalarResult
        assert isinstance(result1, ScalarResult)
        assert isinstance(result2, ScalarResult)
        
        # Check probabilities
        assert abs(float(result1) - 2/6) < 1e-10
        assert abs(float(result2) - 2/6) < 1e-10

    def test_joint_with_isin_pandas(self, sample_pandas_df):
        """Test joint probabilities with isin operator using Pandas.
        
        Requirements: 11.2
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, z = vb.get_variables('x', 'z')
        
        # Test joint probability with isin
        result = p(x.isin([1, 2]), z == 'a')  # P(X in [1,2], Z='a') -> rows (1,'a'), (1,'a') = 2/6
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # Check probability
        assert abs(float(result) - 2/6) < 1e-10


class TestTernaryConditions:
    """Test ternary conditions like p(a < x < b).
    
    Requirements tested:
    - 9.4: Support ternary conditions like p(a < x < b)
    """

    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            'z': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        })

    @pytest.fixture
    def sample_polars_df(self):
        """Create a sample Polars dataframe for testing."""
        return pl.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            'z': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        })

    def test_ternary_expression_creation_pandas(self, sample_pandas_df):
        """Test creating ternary expressions using TernaryExpression class with Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test ternary expression P(3 < X < 7)
        # This should match x = 4, 5, 6 (3 values out of 10)
        ternary_expr = TernaryExpression(x, 3, 7, closed="none")
        result = p(ternary_expr)
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # Check probability: 3 values (4, 5, 6) out of 10 total
        assert abs(float(result) - 3/10) < 1e-10

    def test_ternary_expression_inclusive_bounds_pandas(self, sample_pandas_df):
        """Test ternary expressions with inclusive bounds using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test different closed options
        # P(3 <= X <= 7) should match x = 3, 4, 5, 6, 7 (5 values out of 10)
        ternary_both = TernaryExpression(x, 3, 7, closed="both")
        result_both = p(ternary_both)
        assert abs(float(result_both) - 5/10) < 1e-10
        
        # P(3 <= X < 7) should match x = 3, 4, 5, 6 (4 values out of 10)
        ternary_left = TernaryExpression(x, 3, 7, closed="left")
        result_left = p(ternary_left)
        assert abs(float(result_left) - 4/10) < 1e-10
        
        # P(3 < X <= 7) should match x = 4, 5, 6, 7 (4 values out of 10)
        ternary_right = TernaryExpression(x, 3, 7, closed="right")
        result_right = p(ternary_right)
        assert abs(float(result_right) - 4/10) < 1e-10

    def test_ternary_expression_with_floats_pandas(self, sample_pandas_df):
        """Test ternary expressions with float variables using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        y = vb.get_variables('y')
        
        # Test ternary expression P(3.0 < Y < 7.0)
        # This should match y = 3.3, 4.4, 5.5, 6.6 (4 values out of 10)
        ternary_expr = TernaryExpression(y, 3.0, 7.0, closed="none")
        result = p(ternary_expr)
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # Check probability: 4 values out of 10 total
        assert abs(float(result) - 4/10) < 1e-10

    def test_ternary_expression_polars(self, sample_polars_df):
        """Test ternary expressions with Polars.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x = vb.get_variables('x')
        
        # Test ternary expression P(2 < X < 8)
        # This should match x = 3, 4, 5, 6, 7 (5 values out of 10)
        ternary_expr = TernaryExpression(x, 2, 8, closed="none")
        result = p(ternary_expr)
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # Check probability: 5 values out of 10 total
        assert abs(float(result) - 5/10) < 1e-10

    def test_ternary_expression_with_given_pandas(self, sample_pandas_df):
        """Test ternary expressions with conditional probabilities using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, z = vb.get_variables('x', 'z')
        
        # Test conditional ternary expression P(3 < X < 8 | Z = 'a')
        ternary_expr = TernaryExpression(x, 3, 8, closed="none")
        result = p(ternary_expr).given(z == 'a')
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # When Z='a', we have x = 1, 3, 5, 7, 9 (5 values)
        # P(3 < X < 8 | Z='a') matches x = 5, 7 (2 out of 5)
        assert abs(float(result) - 2/5) < 1e-10

    def test_ternary_expression_distribution_given_pandas(self, sample_pandas_df):
        """Test ternary expressions in conditional distributions using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, z = vb.get_variables('x', 'z')
        
        # Test conditional distribution P(X | 3 < X < 8)
        ternary_expr = TernaryExpression(x, 3, 8, closed="none")
        result = p(x).given(ternary_expr)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict for easier testing
        dist_dict = result.to_dict()
        
        # When 3 < X < 8, we have x = 4, 5, 6, 7 (4 values)
        # Each should have probability 1/4
        expected = {4: 1/4, 5: 1/4, 6: 1/4, 7: 1/4}
        
        assert len(dist_dict) == 4
        for value, expected_prob in expected.items():
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_ternary_expression_combined_with_other_conditions_pandas(self, sample_pandas_df):
        """Test ternary expressions combined with other conditions using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, z = vb.get_variables('x', 'z')
        
        # Test combined condition P(3 < X < 8 AND Z = 'a')
        ternary_expr = TernaryExpression(x, 3, 8, closed="none")
        combined_expr = ternary_expr & (z == 'a')
        result = p(combined_expr)
        
        # Should return ScalarResult
        assert isinstance(result, ScalarResult)
        
        # 3 < X < 8 matches x = 4, 5, 6, 7
        # Z = 'a' matches rows with odd indices: x = 1, 3, 5, 7, 9
        # Combined: x = 5, 7 (2 out of 10 total rows)
        assert abs(float(result) - 2/10) < 1e-10

    def test_ternary_expression_edge_cases_pandas(self, sample_pandas_df):
        """Test ternary expression edge cases using Pandas.
        
        Requirements: 9.4
        """
        from poffertjes.expression import TernaryExpression
        
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test ternary expression with no matches P(15 < X < 20)
        ternary_no_match = TernaryExpression(x, 15, 20, closed="none")
        result_no_match = p(ternary_no_match)
        assert abs(float(result_no_match) - 0.0) < 1e-10
        
        # Test ternary expression that matches all P(0 < X < 15)
        ternary_all_match = TernaryExpression(x, 0, 15, closed="none")
        result_all_match = p(ternary_all_match)
        assert abs(float(result_all_match) - 1.0) < 1e-10


class TestResultExportFunctionality:
    """Test result export functionality (.to_dict(), .to_dataframe()).
    
    Requirements tested:
    - 15.1: Convert probability distribution to dictionary
    - 15.2: Convert probability distribution to dataframe
    - 15.6: Export results to native format conversion
    """

    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            'x': [1, 1, 2, 2, 3, 3, 3],
            'y': ['a', 'b', 'a', 'b', 'a', 'b', 'a'],
            'z': [10.5, 20.5, 10.5, 20.5, 10.5, 20.5, 30.5]
        })

    @pytest.fixture
    def sample_polars_df(self):
        """Create a sample Polars dataframe for testing."""
        return pl.DataFrame({
            'x': [1, 1, 2, 2, 3, 3, 3],
            'y': ['a', 'b', 'a', 'b', 'a', 'b', 'a'],
            'z': [10.5, 20.5, 10.5, 20.5, 10.5, 20.5, 30.5]
        })

    def test_distribution_to_dict_pandas(self, sample_pandas_df):
        """Test converting distribution results to dictionary with Pandas.
        
        Requirements: 15.1
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict
        dist_dict = result.to_dict()
        
        # Check that it's a proper dictionary
        assert isinstance(dist_dict, dict)
        
        # Check expected values and probabilities
        # x=1 appears 2/7 times, x=2 appears 2/7 times, x=3 appears 3/7 times
        expected = {1: 2/7, 2: 2/7, 3: 3/7}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert value in dist_dict
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_distribution_to_dict_polars(self, sample_polars_df):
        """Test converting distribution results to dictionary with Polars.
        
        Requirements: 15.1
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict
        dist_dict = result.to_dict()
        
        # Check that it's a proper dictionary
        assert isinstance(dist_dict, dict)
        
        # Check expected values and probabilities
        expected = {1: 2/7, 2: 2/7, 3: 3/7}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert value in dist_dict
            assert abs(dist_dict[value] - expected_prob) < 1e-10

    def test_joint_distribution_to_dict_pandas(self, sample_pandas_df):
        """Test converting joint distribution results to dictionary with Pandas.
        
        Requirements: 15.1
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution
        result = p(x, y)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict
        dist_dict = result.to_dict()
        
        # Check that it's a proper dictionary with tuple keys
        assert isinstance(dist_dict, dict)
        
        # Check that keys are tuples for joint distributions
        for key in dist_dict.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2  # Two variables
        
        # Check some expected combinations
        # Each combination should appear once out of 7 total rows
        expected_combinations = [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]
        
        for combo in expected_combinations:
            if combo == (3, 'a'):
                # (3, 'a') appears twice: rows 4 and 6
                assert abs(dist_dict[combo] - 2/7) < 1e-10
            else:
                # Other combinations appear once
                assert abs(dist_dict[combo] - 1/7) < 1e-10

    def test_distribution_to_dataframe_pandas(self, sample_pandas_df):
        """Test converting distribution results to dataframe with Pandas.
        
        Requirements: 15.2, 15.6
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dataframe
        result_df = result.to_dataframe()
        
        # Should return a Pandas DataFrame (native format)
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = {'x', 'count', 'probability'}
        assert set(result_df.columns) == expected_columns
        
        # Check that it has the right number of rows
        assert len(result_df) == 3  # Three unique values of x
        
        # Check that probabilities sum to 1.0
        assert abs(result_df['probability'].sum() - 1.0) < 1e-10
        
        # Check specific values
        x1_row = result_df[result_df['x'] == 1]
        assert len(x1_row) == 1
        assert abs(x1_row['probability'].iloc[0] - 2/7) < 1e-10
        assert x1_row['count'].iloc[0] == 2

    def test_distribution_to_dataframe_polars(self, sample_polars_df):
        """Test converting distribution results to dataframe with Polars.
        
        Requirements: 15.2, 15.6
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_polars_df)
        x = vb.get_variables('x')
        
        # Test marginal distribution
        result = p(x)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dataframe
        result_df = result.to_dataframe()
        
        # Should return a Polars DataFrame (native format)
        assert isinstance(result_df, pl.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = {'x', 'count', 'probability'}
        assert set(result_df.columns) == expected_columns
        
        # Check that it has the right number of rows
        assert len(result_df) == 3  # Three unique values of x
        
        # Check that probabilities sum to 1.0
        assert abs(result_df['probability'].sum() - 1.0) < 1e-10

    def test_joint_distribution_to_dataframe_pandas(self, sample_pandas_df):
        """Test converting joint distribution results to dataframe with Pandas.
        
        Requirements: 15.2, 15.6
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution
        result = p(x, y)
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dataframe
        result_df = result.to_dataframe()
        
        # Should return a Pandas DataFrame (native format)
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = {'x', 'y', 'count', 'probability'}
        assert set(result_df.columns) == expected_columns
        
        # Check that it has the right number of rows (unique combinations)
        assert len(result_df) == 6  # Six unique combinations
        
        # Check that probabilities sum to 1.0
        assert abs(result_df['probability'].sum() - 1.0) < 1e-10

    def test_conditional_distribution_to_dict_pandas(self, sample_pandas_df):
        """Test converting conditional distribution results to dictionary with Pandas.
        
        Requirements: 15.1
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditional distribution P(X | Y='a')
        result = p(x).given(y == 'a')
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dict
        dist_dict = result.to_dict()
        
        # Check that it's a proper dictionary
        assert isinstance(dist_dict, dict)
        
        # When Y='a', we have rows: (1,'a'), (2,'a'), (3,'a'), (3,'a')
        # So P(X=1|Y='a') = 1/4, P(X=2|Y='a') = 1/4, P(X=3|Y='a') = 2/4
        expected = {1: 1/4, 2: 1/4, 3: 2/4}
        
        assert len(dist_dict) == 3
        for value, expected_prob in expected.items():
            assert value in dist_dict
            assert abs(dist_dict[value] - expected_prob) < 1e-10
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_conditional_distribution_to_dataframe_pandas(self, sample_pandas_df):
        """Test converting conditional distribution results to dataframe with Pandas.
        
        Requirements: 15.2, 15.6
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        x, y = vb.get_variables('x', 'y')
        
        # Test conditional distribution P(X | Y='b')
        result = p(x).given(y == 'b')
        
        # Should return DistributionResult
        assert isinstance(result, DistributionResult)
        
        # Convert to dataframe
        result_df = result.to_dataframe()
        
        # Should return a Pandas DataFrame (native format)
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = {'x', 'count', 'probability'}
        assert set(result_df.columns) == expected_columns
        
        # When Y='b', we have rows: (1,'b'), (2,'b'), (3,'b')
        # So we should have 3 rows in the result
        assert len(result_df) == 3
        
        # Check that conditional probabilities sum to 1.0
        assert abs(result_df['probability'].sum() - 1.0) < 1e-10

    def test_export_with_different_dtypes_pandas(self, sample_pandas_df):
        """Test exporting distributions with different data types using Pandas.
        
        Requirements: 15.1, 15.2, 15.6
        """
        # Create variables
        vb = VariableBuilder.from_data(sample_pandas_df)
        y, z = vb.get_variables('y', 'z')
        
        # Test string variable distribution
        result_str = p(y)
        dict_str = result_str.to_dict()
        df_str = result_str.to_dataframe()
        
        # Check string keys in dictionary
        assert isinstance(dict_str, dict)
        for key in dict_str.keys():
            assert isinstance(key, str)
        
        # Check string column in dataframe
        assert isinstance(df_str, pd.DataFrame)
        assert 'y' in df_str.columns
        
        # Test float variable distribution
        result_float = p(z)
        dict_float = result_float.to_dict()
        df_float = result_float.to_dataframe()
        
        # Check float keys in dictionary
        assert isinstance(dict_float, dict)
        for key in dict_float.keys():
            assert isinstance(key, float)
        
        # Check float column in dataframe
        assert isinstance(df_float, pd.DataFrame)
        assert 'z' in df_float.columns