"""Tests for Distribution class."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.result import Distribution


class TestDistribution:
    """Test cases for Distribution class."""
    
    def test_init(self):
        """Test Distribution initialization."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        assert dist.data is nw_df
        assert dist.variables == ['x']
    
    def test_iter_single_variable(self):
        """Test iteration over single variable distribution."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        # Test iteration
        pairs = list(dist)
        expected = [(1, 0.3), (2, 0.4), (3, 0.3)]
        
        assert pairs == expected
    
    def test_iter_multiple_variables(self):
        """Test iteration over multiple variable distribution."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 2],
            'y': [1, 2, 1, 2],
            'count': [1, 1, 1, 1],
            'probability': [0.25, 0.25, 0.25, 0.25]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x', 'y'])
        
        # Test iteration
        pairs = list(dist)
        expected = [
            ((1, 1), 0.25),
            ((1, 2), 0.25),
            ((2, 1), 0.25),
            ((2, 2), 0.25)
        ]
        
        assert pairs == expected
    
    def test_repr_single_variable(self):
        """Test string representation for single variable."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        repr_str = repr(dist)
        
        assert "Distribution over x:" in repr_str
        assert "1: 0.300000" in repr_str
        assert "2: 0.400000" in repr_str
        assert "3: 0.300000" in repr_str
    
    def test_repr_multiple_variables(self):
        """Test string representation for multiple variables."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 2],
            'y': [1, 2, 1, 2],
            'count': [1, 1, 1, 1],
            'probability': [0.25, 0.25, 0.25, 0.25]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x', 'y'])
        
        repr_str = repr(dist)
        
        assert "Distribution over x, y:" in repr_str
        assert "(1, 1): 0.250000" in repr_str
        assert "(1, 2): 0.250000" in repr_str
        assert "(2, 1): 0.250000" in repr_str
        assert "(2, 2): 0.250000" in repr_str
    
    def test_repr_truncation(self):
        """Test string representation truncation for many values."""
        # Create a distribution with more than 10 values
        values = list(range(15))
        probs = [1/15] * 15
        df = pd.DataFrame({
            'x': values,
            'count': [1] * 15,
            'probability': probs
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        repr_str = repr(dist)
        
        assert "Distribution over x:" in repr_str
        assert "... (5 more values)" in repr_str
    
    def test_to_dict_single_variable(self):
        """Test conversion to dictionary for single variable."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        result_dict = dist.to_dict()
        expected = {1: 0.3, 2: 0.4, 3: 0.3}
        
        assert result_dict == expected
    
    def test_to_dict_multiple_variables(self):
        """Test conversion to dictionary for multiple variables."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 2],
            'y': [1, 2, 1, 2],
            'count': [1, 1, 1, 1],
            'probability': [0.25, 0.25, 0.25, 0.25]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x', 'y'])
        
        result_dict = dist.to_dict()
        expected = {
            (1, 1): 0.25,
            (1, 2): 0.25,
            (2, 1): 0.25,
            (2, 2): 0.25
        }
        
        assert result_dict == expected
    
    def test_to_dataframe(self):
        """Test conversion to native dataframe."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        nw_df = nw.from_native(df)
        
        dist = Distribution(nw_df, ['x'])
        
        native_df = dist.to_dataframe()
        
        # Should return the original pandas dataframe
        pd.testing.assert_frame_equal(native_df, df)
    
    def test_eq_same_distributions(self):
        """Test equality comparison for identical distributions."""
        df1 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        df2 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        
        dist1 = Distribution(nw.from_native(df1), ['x'])
        dist2 = Distribution(nw.from_native(df2), ['x'])
        
        assert dist1 == dist2
    
    def test_eq_different_variables(self):
        """Test equality comparison for distributions with different variables."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        
        dist1 = Distribution(nw.from_native(df), ['x'])
        dist2 = Distribution(nw.from_native(df), ['y'])
        
        assert dist1 != dist2
    
    def test_eq_different_probabilities(self):
        """Test equality comparison for distributions with different probabilities."""
        df1 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        df2 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.2, 0.5, 0.3]
        })
        
        dist1 = Distribution(nw.from_native(df1), ['x'])
        dist2 = Distribution(nw.from_native(df2), ['x'])
        
        assert dist1 != dist2
    
    def test_eq_different_values(self):
        """Test equality comparison for distributions with different values."""
        df1 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        df2 = pd.DataFrame({
            'x': [1, 2, 4],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        
        dist1 = Distribution(nw.from_native(df1), ['x'])
        dist2 = Distribution(nw.from_native(df2), ['x'])
        
        assert dist1 != dist2
    
    def test_eq_floating_point_tolerance(self):
        """Test equality comparison with floating point tolerance."""
        df1 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        df2 = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3 + 1e-12, 0.4, 0.3]  # Very small difference
        })
        
        dist1 = Distribution(nw.from_native(df1), ['x'])
        dist2 = Distribution(nw.from_native(df2), ['x'])
        
        assert dist1 == dist2  # Should be equal within tolerance
    
    def test_eq_not_distribution(self):
        """Test equality comparison with non-Distribution object."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'count': [3, 4, 3],
            'probability': [0.3, 0.4, 0.3]
        })
        
        dist = Distribution(nw.from_native(df), ['x'])
        
        assert dist != "not a distribution"
        assert dist != 42
        assert dist != None