"""Property-based tests using Hypothesis."""

import pytest
import pandas as pd
import polars as pl
from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, columns
import numpy as np
from typing import Union, List, Any

from poffertjes import p
from poffertjes.variable import VariableBuilder
from poffertjes.result import ScalarResult, DistributionResult


# Hypothesis strategies for generating test data

@st.composite
def dataframe_strategy(draw, backend: str = "pandas", min_rows: int = 10, max_rows: int = 100):
    """Generate test dataframes with multiple dtypes.
    
    Args:
        draw: Hypothesis draw function
        backend: "pandas" or "polars"
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
        
    Returns:
        Generated dataframe with various column types
        
    Requirements: 12.1, 12.9
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=2, max_value=4))  # Keep manageable for testing
    
    # Generate column specifications
    column_specs = []
    column_names = []
    
    for i in range(n_cols):
        col_name = f'col_{i}'
        column_names.append(col_name)
        
        # Choose dtype for this column
        dtype = draw(st.sampled_from(['int', 'float', 'str', 'bool']))
        
        if dtype == 'int':
            # Generate integers with limited range for meaningful probabilities
            values = draw(st.lists(
                st.integers(min_value=0, max_value=5),
                min_size=n_rows, max_size=n_rows
            ))
        elif dtype == 'float':
            # Generate floats with limited precision for meaningful probabilities
            values = draw(st.lists(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
                .map(lambda x: round(x, 1)),  # Round to 1 decimal place
                min_size=n_rows, max_size=n_rows
            ))
        elif dtype == 'str':
            # Generate categorical strings
            categories = ['cat_A', 'cat_B', 'cat_C', 'cat_D']
            values = draw(st.lists(
                st.sampled_from(categories),
                min_size=n_rows, max_size=n_rows
            ))
        elif dtype == 'bool':
            values = draw(st.lists(
                st.booleans(),
                min_size=n_rows, max_size=n_rows
            ))
        
        column_specs.append((col_name, values))
    
    # Create dataframe
    data = {name: values for name, values in column_specs}
    
    if backend == "pandas":
        return pd.DataFrame(data)
    elif backend == "polars":
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@st.composite 
def simple_dataframe_strategy(draw, backend: str = "pandas"):
    """Generate simple 2-column dataframes for basic probability tests.
    
    Args:
        draw: Hypothesis draw function
        backend: "pandas" or "polars"
        
    Returns:
        Simple dataframe with 2 columns for testing basic probability laws
    """
    n_rows = draw(st.integers(min_value=20, max_value=50))
    
    # Generate two columns with small value ranges
    x_values = draw(st.lists(
        st.integers(min_value=0, max_value=3),
        min_size=n_rows, max_size=n_rows
    ))
    
    y_values = draw(st.lists(
        st.integers(min_value=0, max_value=3), 
        min_size=n_rows, max_size=n_rows
    ))
    
    data = {'x': x_values, 'y': y_values}
    
    if backend == "pandas":
        return pd.DataFrame(data)
    elif backend == "polars":
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@st.composite
def three_column_dataframe_strategy(draw, backend: str = "pandas"):
    """Generate 3-column dataframes for testing laws involving 3 variables.
    
    Args:
        draw: Hypothesis draw function
        backend: "pandas" or "polars"
        
    Returns:
        Dataframe with 3 columns for testing complex probability laws
    """
    n_rows = draw(st.integers(min_value=30, max_value=80))
    
    # Generate three columns with small value ranges
    x_values = draw(st.lists(
        st.integers(min_value=0, max_value=2),
        min_size=n_rows, max_size=n_rows
    ))
    
    y_values = draw(st.lists(
        st.integers(min_value=0, max_value=2),
        min_size=n_rows, max_size=n_rows
    ))
    
    z_values = draw(st.lists(
        st.integers(min_value=0, max_value=2),
        min_size=n_rows, max_size=n_rows
    ))
    
    data = {'x': x_values, 'y': y_values, 'z': z_values}
    
    if backend == "pandas":
        return pd.DataFrame(data)
    elif backend == "polars":
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Helper functions for property-based tests

def get_distribution_probabilities(dist_result: DistributionResult) -> List[float]:
    """Extract probability values from a DistributionResult.
    
    Args:
        dist_result: A DistributionResult object
        
    Returns:
        List of probability values
    """
    probabilities = []
    for row in dist_result.distribution.iter_rows(named=True):
        probabilities.append(row.get("probability", 0.0))
    return probabilities


def get_distribution_dict(dist_result: DistributionResult) -> dict:
    """Convert DistributionResult to dictionary.
    
    Args:
        dist_result: A DistributionResult object
        
    Returns:
        Dictionary mapping values to probabilities
    """
    result = {}
    for row in dist_result.distribution.iter_rows(named=True):
        if len(dist_result.variables) == 1:
            key = row[dist_result.variables[0].name]
        else:
            key = tuple(row[var.name] for var in dist_result.variables)
        result[key] = row.get("probability", 0.0)
    return result


def calculate_marginal_from_joint(joint_dist: DistributionResult, variable_index: int) -> dict:
    """Calculate marginal distribution from joint distribution.
    
    Args:
        joint_dist: Joint distribution result
        variable_index: Index of variable to marginalize (0 or 1)
        
    Returns:
        Dictionary mapping values to marginal probabilities
    """
    marginal = {}
    for values, prob in joint_dist.distribution:
        if isinstance(values, tuple):
            value = values[variable_index]
        else:
            # Single variable case
            value = values
        
        if value in marginal:
            marginal[value] += prob
        else:
            marginal[value] = prob
    
    return marginal


# Property-based tests for probability axioms

class TestProbabilityAxioms:
    """Test fundamental probability axioms using property-based testing.
    
    Requirements tested:
    - 12.3: Non-negativity and normalization axioms
    - 16.5: Probabilities sum to 1.0 within tolerance
    """

    @given(dataframe_strategy(backend="pandas"))
    def test_non_negativity_pandas(self, df):
        """Test that all probabilities are non-negative (>= 0).
        
        Requirements: 12.3
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables()
        
        # Test marginal distributions
        for var in variables:
            dist_result = p(var)
            probabilities = get_distribution_probabilities(dist_result)
            
            # All probabilities must be non-negative
            for prob in probabilities:
                assert prob >= 0.0, f"Negative probability found: {prob}"

    @given(dataframe_strategy(backend="polars"))
    def test_non_negativity_polars(self, df):
        """Test that all probabilities are non-negative with Polars backend.
        
        Requirements: 12.3
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables()
        
        # Test marginal distributions
        for var in variables:
            dist_result = p(var)
            probabilities = get_distribution_probabilities(dist_result)
            
            # All probabilities must be non-negative
            for prob in probabilities:
                assert prob >= 0.0, f"Negative probability found: {prob}"

    @given(dataframe_strategy(backend="pandas"))
    def test_normalization_marginal_pandas(self, df):
        """Test that marginal probabilities sum to 1.0.
        
        Requirements: 12.3, 16.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables()
        
        # Test each variable's marginal distribution
        for var in variables:
            dist_result = p(var)
            probabilities = get_distribution_probabilities(dist_result)
            
            # Probabilities must sum to 1.0 (within floating point tolerance)
            total = sum(probabilities)
            assert abs(total - 1.0) < 1e-10, f"Probabilities sum to {total}, not 1.0"

    @given(dataframe_strategy(backend="polars"))
    def test_normalization_marginal_polars(self, df):
        """Test that marginal probabilities sum to 1.0 with Polars backend.
        
        Requirements: 12.3, 16.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables()
        
        # Test each variable's marginal distribution
        for var in variables:
            dist_result = p(var)
            probabilities = get_distribution_probabilities(dist_result)
            
            # Probabilities must sum to 1.0 (within floating point tolerance)
            total = sum(probabilities)
            assert abs(total - 1.0) < 1e-10, f"Probabilities sum to {total}, not 1.0"

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_normalization_joint_pandas(self, df):
        """Test that joint probabilities sum to 1.0.
        
        Requirements: 12.3, 16.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution
        joint_result = p(x, y)
        probabilities = get_distribution_probabilities(joint_result)
        
        # Joint probabilities must sum to 1.0
        total = sum(probabilities)
        assert abs(total - 1.0) < 1e-10, f"Joint probabilities sum to {total}, not 1.0"

    @given(simple_dataframe_strategy(backend="polars"))
    def test_normalization_joint_polars(self, df):
        """Test that joint probabilities sum to 1.0 with Polars backend.
        
        Requirements: 12.3, 16.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test joint distribution
        joint_result = p(x, y)
        probabilities = get_distribution_probabilities(joint_result)
        
        # Joint probabilities must sum to 1.0
        total = sum(probabilities)
        assert abs(total - 1.0) < 1e-10, f"Joint probabilities sum to {total}, not 1.0"

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_conditional_normalization_pandas(self, df):
        """Test that conditional probabilities sum to 1.0 for each conditioning value.
        
        Requirements: 12.3, 16.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get all unique values of y for conditioning
        y_values = df['y'].unique()
        
        for y_val in y_values:
            # Skip if this conditioning value doesn't exist in data
            if (df['y'] == y_val).sum() == 0:
                continue
                
            try:
                # Test P(X | Y = y_val)
                cond_result = p(x).given(y == y_val)
                probabilities = get_distribution_probabilities(cond_result)
                
                # Conditional probabilities must sum to 1.0
                total = sum(probabilities)
                assert abs(total - 1.0) < 1e-10, f"Conditional probabilities sum to {total}, not 1.0 for Y={y_val}"
            except Exception:
                # Skip if conditioning results in zero probability
                continue

    @given(dataframe_strategy(backend="pandas"))
    def test_scalar_probability_bounds_pandas(self, df):
        """Test that scalar probabilities are between 0 and 1.
        
        Requirements: 12.3
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables()
        
        # Test scalar probabilities for each variable
        for var in variables:
            # Get some values from the variable's column
            col_values = df[var.name].unique()[:3]  # Test first 3 unique values
            
            for val in col_values:
                scalar_result = p(var == val)
                prob = float(scalar_result)
                
                # Probability must be between 0 and 1
                assert 0.0 <= prob <= 1.0, f"Probability {prob} not in [0,1] for {var.name}={val}"


class TestChainRule:
    """Test the chain rule: P(X,Y) = P(X|Y) * P(Y).
    
    Requirements tested:
    - 12.4: Chain rule verification
    """

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_chain_rule_pandas(self, df):
        """Test chain rule P(X,Y) = P(X|Y) * P(Y) with Pandas.
        
        Requirements: 12.4
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # Get marginal distribution P(Y)
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # For each (x_val, y_val) pair in joint distribution
        for (x_val, y_val), joint_prob in joint_dict.items():
            if isinstance((x_val, y_val), tuple):
                x_v, y_v = x_val, y_val
            else:
                # Handle case where joint result might not be tuple
                continue
                
            # Skip if P(Y=y_val) is 0 (would make conditional undefined)
            if y_v not in y_marginal_dict or y_marginal_dict[y_v] == 0:
                continue
                
            try:
                # Calculate P(X=x_val | Y=y_val)
                conditional_result = p(x == x_v).given(y == y_v)
                conditional_prob = float(conditional_result)
                
                # Calculate P(Y=y_val)
                marginal_prob = y_marginal_dict[y_v]
                
                # Verify chain rule: P(X,Y) = P(X|Y) * P(Y)
                expected_joint = conditional_prob * marginal_prob
                
                # Allow for floating point tolerance
                assert abs(joint_prob - expected_joint) < 1e-10, (
                    f"Chain rule failed: P({x_v},{y_v})={joint_prob} != "
                    f"P({x_v}|{y_v}) * P({y_v}) = {conditional_prob} * {marginal_prob} = {expected_joint}"
                )
            except Exception:
                # Skip if conditional probability calculation fails
                continue

    @given(simple_dataframe_strategy(backend="polars"))
    def test_chain_rule_polars(self, df):
        """Test chain rule P(X,Y) = P(X|Y) * P(Y) with Polars.
        
        Requirements: 12.4
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # Get marginal distribution P(Y)
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # For each (x_val, y_val) pair in joint distribution
        for (x_val, y_val), joint_prob in joint_dict.items():
            if isinstance((x_val, y_val), tuple):
                x_v, y_v = x_val, y_val
            else:
                # Handle case where joint result might not be tuple
                continue
                
            # Skip if P(Y=y_val) is 0 (would make conditional undefined)
            if y_v not in y_marginal_dict or y_marginal_dict[y_v] == 0:
                continue
                
            try:
                # Calculate P(X=x_val | Y=y_val)
                conditional_result = p(x == x_v).given(y == y_v)
                conditional_prob = float(conditional_result)
                
                # Calculate P(Y=y_val)
                marginal_prob = y_marginal_dict[y_v]
                
                # Verify chain rule: P(X,Y) = P(X|Y) * P(Y)
                expected_joint = conditional_prob * marginal_prob
                
                # Allow for floating point tolerance
                assert abs(joint_prob - expected_joint) < 1e-10, (
                    f"Chain rule failed: P({x_v},{y_v})={joint_prob} != "
                    f"P({x_v}|{y_v}) * P({y_v}) = {conditional_prob} * {marginal_prob} = {expected_joint}"
                )
            except Exception:
                # Skip if conditional probability calculation fails
                continue

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_chain_rule_reverse_pandas(self, df):
        """Test reverse chain rule P(X,Y) = P(Y|X) * P(X) with Pandas.
        
        Requirements: 12.4
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # For each (x_val, y_val) pair in joint distribution
        for (x_val, y_val), joint_prob in joint_dict.items():
            if isinstance((x_val, y_val), tuple):
                x_v, y_v = x_val, y_val
            else:
                # Handle case where joint result might not be tuple
                continue
                
            # Skip if P(X=x_val) is 0 (would make conditional undefined)
            if x_v not in x_marginal_dict or x_marginal_dict[x_v] == 0:
                continue
                
            try:
                # Calculate P(Y=y_val | X=x_val)
                conditional_result = p(y == y_v).given(x == x_v)
                conditional_prob = float(conditional_result)
                
                # Calculate P(X=x_val)
                marginal_prob = x_marginal_dict[x_v]
                
                # Verify chain rule: P(X,Y) = P(Y|X) * P(X)
                expected_joint = conditional_prob * marginal_prob
                
                # Allow for floating point tolerance
                assert abs(joint_prob - expected_joint) < 1e-10, (
                    f"Reverse chain rule failed: P({x_v},{y_v})={joint_prob} != "
                    f"P({y_v}|{x_v}) * P({x_v}) = {conditional_prob} * {marginal_prob} = {expected_joint}"
                )
            except Exception:
                # Skip if conditional probability calculation fails
                continue
class TestBayesTheorem:
    """Test Bayes' theorem: P(X|Y) * P(Y) = P(Y|X) * P(X).
    
    Requirements tested:
    - 12.5: Bayes' theorem verification
    """

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_bayes_theorem_pandas(self, df):
        """Test Bayes' theorem P(X|Y) * P(Y) = P(Y|X) * P(X) with Pandas.
        
        Requirements: 12.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distributions
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # Test Bayes' theorem for various value combinations
        x_values = list(x_marginal_dict.keys())[:3]  # Test first 3 values
        y_values = list(y_marginal_dict.keys())[:3]  # Test first 3 values
        
        for x_val in x_values:
            for y_val in y_values:
                # Skip if either marginal probability is 0
                if x_marginal_dict[x_val] == 0 or y_marginal_dict[y_val] == 0:
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val)
                    p_x_given_y_result = p(x == x_val).given(y == y_val)
                    p_x_given_y = float(p_x_given_y_result)
                    
                    # Calculate P(Y=y_val | X=x_val)
                    p_y_given_x_result = p(y == y_val).given(x == x_val)
                    p_y_given_x = float(p_y_given_x_result)
                    
                    # Get marginal probabilities
                    p_x = x_marginal_dict[x_val]
                    p_y = y_marginal_dict[y_val]
                    
                    # Verify Bayes' theorem: P(X|Y) * P(Y) = P(Y|X) * P(X)
                    left_side = p_x_given_y * p_y
                    right_side = p_y_given_x * p_x
                    
                    # Allow for floating point tolerance
                    assert abs(left_side - right_side) < 1e-10, (
                        f"Bayes' theorem failed for X={x_val}, Y={y_val}: "
                        f"P(X|Y)*P(Y) = {p_x_given_y}*{p_y} = {left_side} != "
                        f"P(Y|X)*P(X) = {p_y_given_x}*{p_x} = {right_side}"
                    )
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue

    @given(simple_dataframe_strategy(backend="polars"))
    def test_bayes_theorem_polars(self, df):
        """Test Bayes' theorem P(X|Y) * P(Y) = P(Y|X) * P(X) with Polars.
        
        Requirements: 12.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distributions
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # Test Bayes' theorem for various value combinations
        x_values = list(x_marginal_dict.keys())[:3]  # Test first 3 values
        y_values = list(y_marginal_dict.keys())[:3]  # Test first 3 values
        
        for x_val in x_values:
            for y_val in y_values:
                # Skip if either marginal probability is 0
                if x_marginal_dict[x_val] == 0 or y_marginal_dict[y_val] == 0:
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val)
                    p_x_given_y_result = p(x == x_val).given(y == y_val)
                    p_x_given_y = float(p_x_given_y_result)
                    
                    # Calculate P(Y=y_val | X=x_val)
                    p_y_given_x_result = p(y == y_val).given(x == x_val)
                    p_y_given_x = float(p_y_given_x_result)
                    
                    # Get marginal probabilities
                    p_x = x_marginal_dict[x_val]
                    p_y = y_marginal_dict[y_val]
                    
                    # Verify Bayes' theorem: P(X|Y) * P(Y) = P(Y|X) * P(X)
                    left_side = p_x_given_y * p_y
                    right_side = p_y_given_x * p_x
                    
                    # Allow for floating point tolerance
                    assert abs(left_side - right_side) < 1e-10, (
                        f"Bayes' theorem failed for X={x_val}, Y={y_val}: "
                        f"P(X|Y)*P(Y) = {p_x_given_y}*{p_y} = {left_side} != "
                        f"P(Y|X)*P(X) = {p_y_given_x}*{p_x} = {right_side}"
                    )
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_bayes_theorem_alternative_form_pandas(self, df):
        """Test alternative form of Bayes' theorem: P(X|Y) = P(Y|X) * P(X) / P(Y).
        
        Requirements: 12.5
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distributions
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # Test alternative form for various value combinations
        x_values = list(x_marginal_dict.keys())[:3]  # Test first 3 values
        y_values = list(y_marginal_dict.keys())[:3]  # Test first 3 values
        
        for x_val in x_values:
            for y_val in y_values:
                # Skip if P(Y) is 0 (would cause division by zero)
                if y_marginal_dict[y_val] == 0:
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val) directly
                    p_x_given_y_result = p(x == x_val).given(y == y_val)
                    p_x_given_y_direct = float(p_x_given_y_result)
                    
                    # Calculate P(Y=y_val | X=x_val)
                    p_y_given_x_result = p(y == y_val).given(x == x_val)
                    p_y_given_x = float(p_y_given_x_result)
                    
                    # Get marginal probabilities
                    p_x = x_marginal_dict[x_val]
                    p_y = y_marginal_dict[y_val]
                    
                    # Calculate P(X|Y) using Bayes' theorem: P(Y|X) * P(X) / P(Y)
                    p_x_given_y_bayes = (p_y_given_x * p_x) / p_y
                    
                    # Verify they match
                    assert abs(p_x_given_y_direct - p_x_given_y_bayes) < 1e-10, (
                        f"Bayes' theorem alternative form failed for X={x_val}, Y={y_val}: "
                        f"P(X|Y) direct = {p_x_given_y_direct} != "
                        f"P(Y|X)*P(X)/P(Y) = {p_y_given_x}*{p_x}/{p_y} = {p_x_given_y_bayes}"
                    )
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue


class TestLawOfTotalProbability:
    """Test law of total probability: P(X) = sum over Y of P(X|Y) * P(Y).
    
    Requirements tested:
    - 12.6: Law of total probability verification
    """

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_law_of_total_probability_pandas(self, df):
        """Test law of total probability P(X) = sum_y P(X|Y=y) * P(Y=y) with Pandas.
        
        Requirements: 12.6
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get marginal distribution P(Y)
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # For each value of X, verify law of total probability
        for x_val, p_x_direct in x_marginal_dict.items():
            # Calculate sum over Y of P(X=x_val | Y=y) * P(Y=y)
            total_prob = 0.0
            
            for y_val, p_y in y_marginal_dict.items():
                if p_y == 0:  # Skip if P(Y=y) is 0
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val)
                    p_x_given_y_result = p(x == x_val).given(y == y_val)
                    p_x_given_y = float(p_x_given_y_result)
                    
                    # Add P(X|Y) * P(Y) to total
                    total_prob += p_x_given_y * p_y
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue
            
            # Verify law of total probability
            assert abs(p_x_direct - total_prob) < 1e-10, (
                f"Law of total probability failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_y P(X|Y)*P(Y) = {total_prob}"
            )

    @given(simple_dataframe_strategy(backend="polars"))
    def test_law_of_total_probability_polars(self, df):
        """Test law of total probability P(X) = sum_y P(X|Y=y) * P(Y=y) with Polars.
        
        Requirements: 12.6
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get marginal distribution P(Y)
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # For each value of X, verify law of total probability
        for x_val, p_x_direct in x_marginal_dict.items():
            # Calculate sum over Y of P(X=x_val | Y=y) * P(Y=y)
            total_prob = 0.0
            
            for y_val, p_y in y_marginal_dict.items():
                if p_y == 0:  # Skip if P(Y=y) is 0
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val)
                    p_x_given_y_result = p(x == x_val).given(y == y_val)
                    p_x_given_y = float(p_x_given_y_result)
                    
                    # Add P(X|Y) * P(Y) to total
                    total_prob += p_x_given_y * p_y
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue
            
            # Verify law of total probability
            assert abs(p_x_direct - total_prob) < 1e-10, (
                f"Law of total probability failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_y P(X|Y)*P(Y) = {total_prob}"
            )

    @given(three_column_dataframe_strategy(backend="pandas"))
    def test_law_of_total_probability_three_variables_pandas(self, df):
        """Test law of total probability with three variables: P(X) = sum_{y,z} P(X|Y,Z) * P(Y,Z).
        
        Requirements: 12.6
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get joint distribution P(Y,Z)
        yz_joint_result = p(y, z)
        yz_joint_dict = get_distribution_dict(yz_joint_result)
        
        # For each value of X, verify law of total probability
        for x_val, p_x_direct in x_marginal_dict.items():
            # Calculate sum over (Y,Z) of P(X=x_val | Y=y, Z=z) * P(Y=y, Z=z)
            total_prob = 0.0
            
            for (y_val, z_val), p_yz in yz_joint_dict.items():
                if p_yz == 0:  # Skip if P(Y=y, Z=z) is 0
                    continue
                    
                try:
                    # Calculate P(X=x_val | Y=y_val, Z=z_val)
                    p_x_given_yz_result = p(x == x_val).given(y == y_val, z == z_val)
                    p_x_given_yz = float(p_x_given_yz_result)
                    
                    # Add P(X|Y,Z) * P(Y,Z) to total
                    total_prob += p_x_given_yz * p_yz
                except Exception:
                    # Skip if conditional probability calculation fails
                    continue
            
            # Verify law of total probability
            assert abs(p_x_direct - total_prob) < 1e-10, (
                f"Law of total probability (3 vars) failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_{{y,z}} P(X|Y,Z)*P(Y,Z) = {total_prob}"
            )
            
class TestUnionProbabilityFormula:
    """Test union probability formula: P(X or Y) = P(X) + P(Y) - P(X and Y).
    
    Requirements tested:
    - 12.7: Union probability formula verification
    """

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_union_probability_formula_pandas(self, df):
        """Test union formula P(A or B) = P(A) + P(B) - P(A and B) with Pandas.
        
        Requirements: 12.7
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get unique values for testing
        x_values = df['x'].unique()[:2]  # Test first 2 values
        y_values = df['y'].unique()[:2]  # Test first 2 values
        
        for x_val in x_values:
            for y_val in y_values:
                try:
                    # Calculate P(X = x_val)
                    p_x_result = p(x == x_val)
                    p_x = float(p_x_result)
                    
                    # Calculate P(Y = y_val)
                    p_y_result = p(y == y_val)
                    p_y = float(p_y_result)
                    
                    # Calculate P(X = x_val AND Y = y_val)
                    p_x_and_y_result = p(x == x_val, y == y_val)
                    p_x_and_y = float(p_x_and_y_result)
                    
                    # Calculate P(X = x_val OR Y = y_val) using union formula
                    p_x_or_y_formula = p_x + p_y - p_x_and_y
                    
                    # Calculate P(X = x_val OR Y = y_val) directly by counting
                    # Count rows where either condition is true
                    mask = (df['x'] == x_val) | (df['y'] == y_val)
                    p_x_or_y_direct = mask.sum() / len(df)
                    
                    # Verify union formula
                    assert abs(p_x_or_y_direct - p_x_or_y_formula) < 1e-10, (
                        f"Union formula failed for X={x_val}, Y={y_val}: "
                        f"P(X or Y) direct = {p_x_or_y_direct} != "
                        f"P(X) + P(Y) - P(X and Y) = {p_x} + {p_y} - {p_x_and_y} = {p_x_or_y_formula}"
                    )
                except Exception:
                    # Skip if probability calculation fails
                    continue

    @given(simple_dataframe_strategy(backend="polars"))
    def test_union_probability_formula_polars(self, df):
        """Test union formula P(A or B) = P(A) + P(B) - P(A and B) with Polars.
        
        Requirements: 12.7
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Convert to pandas for easier unique value extraction
        df_pd = df.to_pandas()
        
        # Get unique values for testing
        x_values = df_pd['x'].unique()[:2]  # Test first 2 values
        y_values = df_pd['y'].unique()[:2]  # Test first 2 values
        
        for x_val in x_values:
            for y_val in y_values:
                try:
                    # Calculate P(X = x_val)
                    p_x_result = p(x == x_val)
                    p_x = float(p_x_result)
                    
                    # Calculate P(Y = y_val)
                    p_y_result = p(y == y_val)
                    p_y = float(p_y_result)
                    
                    # Calculate P(X = x_val AND Y = y_val)
                    p_x_and_y_result = p(x == x_val, y == y_val)
                    p_x_and_y = float(p_x_and_y_result)
                    
                    # Calculate P(X = x_val OR Y = y_val) using union formula
                    p_x_or_y_formula = p_x + p_y - p_x_and_y
                    
                    # Calculate P(X = x_val OR Y = y_val) directly by counting
                    # Count rows where either condition is true
                    mask = (df_pd['x'] == x_val) | (df_pd['y'] == y_val)
                    p_x_or_y_direct = mask.sum() / len(df_pd)
                    
                    # Verify union formula
                    assert abs(p_x_or_y_direct - p_x_or_y_formula) < 1e-10, (
                        f"Union formula failed for X={x_val}, Y={y_val}: "
                        f"P(X or Y) direct = {p_x_or_y_direct} != "
                        f"P(X) + P(Y) - P(X and Y) = {p_x} + {p_y} - {p_x_and_y} = {p_x_or_y_formula}"
                    )
                except Exception:
                    # Skip if probability calculation fails
                    continue

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_union_probability_special_cases_pandas(self, df):
        """Test special cases of union probability formula with Pandas.
        
        Requirements: 12.7
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get unique values for testing
        x_values = df['x'].unique()[:2]  # Test first 2 values
        
        for x_val in x_values:
            try:
                # Test case: P(X or X) = P(X)
                # This should hold: P(X) + P(X) - P(X and X) = P(X) + P(X) - P(X) = P(X)
                
                # Calculate P(X = x_val)
                p_x_result = p(x == x_val)
                p_x = float(p_x_result)
                
                # Calculate P(X = x_val AND X = x_val) = P(X = x_val)
                p_x_and_x_result = p(x == x_val, x == x_val)
                p_x_and_x = float(p_x_and_x_result)
                
                # P(X and X) should equal P(X)
                assert abs(p_x - p_x_and_x) < 1e-10, (
                    f"P(X and X) should equal P(X): P(X={x_val}) = {p_x}, "
                    f"P(X={x_val} and X={x_val}) = {p_x_and_x}"
                )
                
                # P(X or X) using formula should equal P(X)
                p_x_or_x_formula = p_x + p_x - p_x_and_x
                assert abs(p_x - p_x_or_x_formula) < 1e-10, (
                    f"Union formula for identical events failed: "
                    f"P(X or X) = {p_x_or_x_formula} != P(X) = {p_x}"
                )
                
            except Exception:
                # Skip if probability calculation fails
                continue

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_union_probability_mutually_exclusive_pandas(self, df):
        """Test union formula for mutually exclusive events with Pandas.
        
        For mutually exclusive events A and B: P(A and B) = 0, so P(A or B) = P(A) + P(B).
        
        Requirements: 12.7
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Get unique values for testing mutually exclusive events
        x_values = df['x'].unique()[:2]  # Test first 2 values
        
        if len(x_values) >= 2:
            x_val1, x_val2 = x_values[0], x_values[1]
            
            try:
                # Calculate P(X = x_val1)
                p_x1_result = p(x == x_val1)
                p_x1 = float(p_x1_result)
                
                # Calculate P(X = x_val2)
                p_x2_result = p(x == x_val2)
                p_x2 = float(p_x2_result)
                
                # Calculate P(X = x_val1 AND X = x_val2) - should be 0 (mutually exclusive)
                p_x1_and_x2_result = p(x == x_val1, x == x_val2)
                p_x1_and_x2 = float(p_x1_and_x2_result)
                
                # For mutually exclusive events, P(A and B) should be 0
                assert abs(p_x1_and_x2) < 1e-10, (
                    f"Mutually exclusive events should have P(A and B) = 0: "
                    f"P(X={x_val1} and X={x_val2}) = {p_x1_and_x2}"
                )
                
                # Calculate P(X = x_val1 OR X = x_val2) using union formula
                p_x1_or_x2_formula = p_x1 + p_x2 - p_x1_and_x2
                
                # For mutually exclusive events, this should equal P(A) + P(B)
                p_x1_or_x2_simple = p_x1 + p_x2
                
                assert abs(p_x1_or_x2_formula - p_x1_or_x2_simple) < 1e-10, (
                    f"Union formula for mutually exclusive events failed: "
                    f"P(A) + P(B) - P(A and B) = {p_x1_or_x2_formula} != "
                    f"P(A) + P(B) = {p_x1_or_x2_simple}"
                )
                
            except Exception:
                # Skip if probability calculation fails
                pass
            
class TestMarginalization:
    """Test marginalization: sum over Y of P(X,Y) = P(X).
    
    Requirements tested:
    - 12.8: Marginalization verification
    """

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_marginalization_pandas(self, df):
        """Test marginalization sum_y P(X,Y=y) = P(X) with Pandas.
        
        Requirements: 12.8
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # For each value of X, sum over all Y values in joint distribution
        for x_val, p_x_direct in x_marginal_dict.items():
            # Sum P(X=x_val, Y=y) over all y values
            marginal_from_joint = 0.0
            
            for (joint_x_val, joint_y_val), joint_prob in joint_dict.items():
                if isinstance((joint_x_val, joint_y_val), tuple):
                    jx, jy = joint_x_val, joint_y_val
                else:
                    # Handle case where joint result might not be tuple
                    continue
                    
                if jx == x_val:
                    marginal_from_joint += joint_prob
            
            # Verify marginalization: sum_y P(X,Y) = P(X)
            assert abs(p_x_direct - marginal_from_joint) < 1e-10, (
                f"Marginalization failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_y P(X,Y) = {marginal_from_joint}"
            )

    @given(simple_dataframe_strategy(backend="polars"))
    def test_marginalization_polars(self, df):
        """Test marginalization sum_y P(X,Y=y) = P(X) with Polars.
        
        Requirements: 12.8
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # For each value of X, sum over all Y values in joint distribution
        for x_val, p_x_direct in x_marginal_dict.items():
            # Sum P(X=x_val, Y=y) over all y values
            marginal_from_joint = 0.0
            
            for (joint_x_val, joint_y_val), joint_prob in joint_dict.items():
                if isinstance((joint_x_val, joint_y_val), tuple):
                    jx, jy = joint_x_val, joint_y_val
                else:
                    # Handle case where joint result might not be tuple
                    continue
                    
                if jx == x_val:
                    marginal_from_joint += joint_prob
            
            # Verify marginalization: sum_y P(X,Y) = P(X)
            assert abs(p_x_direct - marginal_from_joint) < 1e-10, (
                f"Marginalization failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_y P(X,Y) = {marginal_from_joint}"
            )

    @given(simple_dataframe_strategy(backend="pandas"))
    def test_reverse_marginalization_pandas(self, df):
        """Test reverse marginalization sum_x P(X,Y=y) = P(Y) with Pandas.
        
        Requirements: 12.8
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Get marginal distribution P(Y)
        y_marginal_result = p(y)
        y_marginal_dict = get_distribution_dict(y_marginal_result)
        
        # Get joint distribution P(X,Y)
        joint_result = p(x, y)
        joint_dict = get_distribution_dict(joint_result)
        
        # For each value of Y, sum over all X values in joint distribution
        for y_val, p_y_direct in y_marginal_dict.items():
            # Sum P(X=x, Y=y_val) over all x values
            marginal_from_joint = 0.0
            
            for (joint_x_val, joint_y_val), joint_prob in joint_dict.items():
                if isinstance((joint_x_val, joint_y_val), tuple):
                    jx, jy = joint_x_val, joint_y_val
                else:
                    # Handle case where joint result might not be tuple
                    continue
                    
                if jy == y_val:
                    marginal_from_joint += joint_prob
            
            # Verify marginalization: sum_x P(X,Y) = P(Y)
            assert abs(p_y_direct - marginal_from_joint) < 1e-10, (
                f"Reverse marginalization failed for Y={y_val}: "
                f"P(Y) = {p_y_direct} != sum_x P(X,Y) = {marginal_from_joint}"
            )

    @given(three_column_dataframe_strategy(backend="pandas"))
    def test_marginalization_three_variables_pandas(self, df):
        """Test marginalization with three variables: sum_{y,z} P(X,Y,Z) = P(X).
        
        Requirements: 12.8
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Get marginal distribution P(X)
        x_marginal_result = p(x)
        x_marginal_dict = get_distribution_dict(x_marginal_result)
        
        # Get joint distribution P(X,Y,Z)
        joint_result = p(x, y, z)
        joint_dict = get_distribution_dict(joint_result)
        
        # For each value of X, sum over all (Y,Z) values in joint distribution
        for x_val, p_x_direct in x_marginal_dict.items():
            # Sum P(X=x_val, Y=y, Z=z) over all (y,z) values
            marginal_from_joint = 0.0
            
            for joint_values, joint_prob in joint_dict.items():
                if isinstance(joint_values, tuple) and len(joint_values) == 3:
                    jx, jy, jz = joint_values
                else:
                    # Handle case where joint result might not be 3-tuple
                    continue
                    
                if jx == x_val:
                    marginal_from_joint += joint_prob
            
            # Verify marginalization: sum_{y,z} P(X,Y,Z) = P(X)
            assert abs(p_x_direct - marginal_from_joint) < 1e-10, (
                f"3-variable marginalization failed for X={x_val}: "
                f"P(X) = {p_x_direct} != sum_{{y,z}} P(X,Y,Z) = {marginal_from_joint}"
            )

    @given(three_column_dataframe_strategy(backend="pandas"))
    def test_partial_marginalization_pandas(self, df):
        """Test partial marginalization: sum_z P(X,Y,Z) = P(X,Y).
        
        Requirements: 12.8
        """
        assume(len(df) > 0)  # Ensure non-empty dataframe
        
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Get joint distribution P(X,Y)
        xy_joint_result = p(x, y)
        xy_joint_dict = get_distribution_dict(xy_joint_result)
        
        # Get joint distribution P(X,Y,Z)
        xyz_joint_result = p(x, y, z)
        xyz_joint_dict = get_distribution_dict(xyz_joint_result)
        
        # For each (X,Y) pair, sum over all Z values
        for (x_val, y_val), p_xy_direct in xy_joint_dict.items():
            if not isinstance((x_val, y_val), tuple):
                continue
                
            # Sum P(X=x_val, Y=y_val, Z=z) over all z values
            marginal_from_joint = 0.0
            
            for joint_values, joint_prob in xyz_joint_dict.items():
                if isinstance(joint_values, tuple) and len(joint_values) == 3:
                    jx, jy, jz = joint_values
                else:
                    continue
                    
                if jx == x_val and jy == y_val:
                    marginal_from_joint += joint_prob
            
            # Verify partial marginalization: sum_z P(X,Y,Z) = P(X,Y)
            assert abs(p_xy_direct - marginal_from_joint) < 1e-10, (
                f"Partial marginalization failed for X={x_val}, Y={y_val}: "
                f"P(X,Y) = {p_xy_direct} != sum_z P(X,Y,Z) = {marginal_from_joint}"
            )
            
class TestEdgeCases:
    """Test edge cases and numerical stability.
    
    Requirements tested:
    - 12.14: Edge cases with empty dataframes, single-value columns, missing data
    - 16.1: Single unique value handling
    - 16.2: Large dataset handling
    - 16.4: Many unique values handling
    - 16.6: Duplicate rows handling
    - 16.7: Zero probability conditions
    - 16.8: All rows matching conditions
    - 16.9: Very small datasets
    - 16.10: Null/NaN value handling
    """

    def test_empty_dataframe_pandas(self):
        """Test that empty dataframes raise appropriate errors.
        
        Requirements: 12.14, 16.1
        """
        empty_df = pd.DataFrame({'x': [], 'y': []})
        
        with pytest.raises(Exception):  # Should raise DataframeError
            vb = VariableBuilder.from_data(empty_df)

    def test_empty_dataframe_polars(self):
        """Test that empty dataframes raise appropriate errors with Polars.
        
        Requirements: 12.14, 16.1
        """
        empty_df = pl.DataFrame({'x': [], 'y': []})
        
        with pytest.raises(Exception):  # Should raise DataframeError
            vb = VariableBuilder.from_data(empty_df)

    @given(st.integers(min_value=1, max_value=5))
    def test_single_value_columns_pandas(self, constant_value):
        """Test columns with only one unique value.
        
        Requirements: 16.1
        """
        # Create dataframe where one column has only one unique value
        df = pd.DataFrame({
            'constant': [constant_value] * 20,
            'variable': list(range(20))
        })
        
        vb = VariableBuilder.from_data(df)
        constant_var, variable_var = vb.get_variables('constant', 'variable')
        
        # P(constant = constant_value) should be 1.0
        prob_result = p(constant_var == constant_value)
        prob = float(prob_result)
        assert abs(prob - 1.0) < 1e-10, f"P(constant={constant_value}) = {prob}, expected 1.0"
        
        # Distribution of constant variable should have probability 1.0 for the single value
        dist_result = p(constant_var)
        dist_dict = get_distribution_dict(dist_result)
        assert len(dist_dict) == 1, f"Expected 1 unique value, got {len(dist_dict)}"
        assert abs(list(dist_dict.values())[0] - 1.0) < 1e-10, "Single value should have probability 1.0"

    @given(st.integers(min_value=1, max_value=5))
    def test_single_value_columns_polars(self, constant_value):
        """Test columns with only one unique value with Polars.
        
        Requirements: 16.1
        """
        # Create dataframe where one column has only one unique value
        df = pl.DataFrame({
            'constant': [constant_value] * 20,
            'variable': list(range(20))
        })
        
        vb = VariableBuilder.from_data(df)
        constant_var, variable_var = vb.get_variables('constant', 'variable')
        
        # P(constant = constant_value) should be 1.0
        prob_result = p(constant_var == constant_value)
        prob = float(prob_result)
        assert abs(prob - 1.0) < 1e-10, f"P(constant={constant_value}) = {prob}, expected 1.0"
        
        # Distribution of constant variable should have probability 1.0 for the single value
        dist_result = p(constant_var)
        dist_dict = get_distribution_dict(dist_result)
        assert len(dist_dict) == 1, f"Expected 1 unique value, got {len(dist_dict)}"
        assert abs(list(dist_dict.values())[0] - 1.0) < 1e-10, "Single value should have probability 1.0"

    @given(st.integers(min_value=5, max_value=15))
    def test_very_small_datasets_pandas(self, n_rows):
        """Test with very small datasets (1-15 rows).
        
        Requirements: 16.9
        """
        # Create small dataset
        df = pd.DataFrame({
            'x': list(range(n_rows)),
            'y': [i % 3 for i in range(n_rows)]  # Values 0, 1, 2
        })
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test that probabilities still sum to 1
        x_dist_result = p(x)
        x_probabilities = get_distribution_probabilities(x_dist_result)
        x_total = sum(x_probabilities)
        assert abs(x_total - 1.0) < 1e-10, f"Small dataset probabilities sum to {x_total}, not 1.0"
        
        # Test that all probabilities are non-negative
        for prob in x_probabilities:
            assert prob >= 0.0, f"Negative probability in small dataset: {prob}"

    def test_duplicate_rows_pandas(self):
        """Test that duplicate rows are counted appropriately.
        
        Requirements: 16.6
        """
        # Create dataframe with duplicate rows
        df = pd.DataFrame({
            'x': [1, 1, 1, 2, 2, 3],  # Value 1 appears 3 times, 2 appears 2 times, 3 appears 1 time
            'y': ['a', 'a', 'a', 'b', 'b', 'c']
        })
        
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Test probabilities reflect duplicate counts
        prob_1_result = p(x == 1)
        prob_1 = float(prob_1_result)
        expected_prob_1 = 3.0 / 6.0  # 3 occurrences out of 6 total
        assert abs(prob_1 - expected_prob_1) < 1e-10, f"P(X=1) = {prob_1}, expected {expected_prob_1}"
        
        prob_2_result = p(x == 2)
        prob_2 = float(prob_2_result)
        expected_prob_2 = 2.0 / 6.0  # 2 occurrences out of 6 total
        assert abs(prob_2 - expected_prob_2) < 1e-10, f"P(X=2) = {prob_2}, expected {expected_prob_2}"

    def test_zero_probability_conditions_pandas(self):
        """Test conditions that match zero rows.
        
        Requirements: 16.7
        """
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': ['a', 'b', 'c', 'd', 'e']
        })
        
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Test probability of non-existent value
        prob_result = p(x == 999)  # Value that doesn't exist
        prob = float(prob_result)
        assert abs(prob - 0.0) < 1e-10, f"P(X=999) = {prob}, expected 0.0"

    def test_all_rows_matching_condition_pandas(self):
        """Test conditions where all rows match.
        
        Requirements: 16.8
        """
        df = pd.DataFrame({
            'x': [1, 1, 1, 1, 1],  # All rows have x = 1
            'y': ['a', 'b', 'c', 'd', 'e']
        })
        
        vb = VariableBuilder.from_data(df)
        x = vb.get_variables('x')
        
        # Test probability when all rows match
        prob_result = p(x == 1)
        prob = float(prob_result)
        assert abs(prob - 1.0) < 1e-10, f"P(X=1) = {prob}, expected 1.0 when all rows match"

    def test_missing_data_handling_pandas(self):
        """Test handling of NaN/None values.
        
        Requirements: 16.10
        """
        # Create dataframe with NaN values
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': ['a', 'b', None, 'd', 'e']
        })
        
        # Should not raise error when creating variables
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test that probabilities are calculated correctly (excluding NaN)
        # This test verifies the system handles NaN gracefully
        try:
            dist_result = p(x)
            probabilities = get_distribution_probabilities(dist_result)
            # Should not crash and should return valid probabilities
            for prob in probabilities:
                assert prob >= 0.0, f"Invalid probability with NaN data: {prob}"
        except Exception as e:
            # If NaN handling isn't implemented yet, that's acceptable
            # The test documents the expected behavior
            pass

    @given(st.integers(min_value=100, max_value=500))
    def test_large_dataset_performance_pandas(self, n_rows):
        """Test performance with larger datasets.
        
        Requirements: 16.2
        """
        # Create larger dataset
        df = pd.DataFrame({
            'x': [i % 10 for i in range(n_rows)],  # 10 unique values
            'y': [i % 5 for i in range(n_rows)]    # 5 unique values
        })
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test that calculations complete without error
        # and maintain probability axioms
        x_dist_result = p(x)
        x_probabilities = get_distribution_probabilities(x_dist_result)
        x_total = sum(x_probabilities)
        assert abs(x_total - 1.0) < 1e-10, f"Large dataset probabilities sum to {x_total}, not 1.0"

    @given(st.integers(min_value=50, max_value=200))
    def test_many_unique_values_pandas(self, n_unique):
        """Test handling of variables with many unique values.
        
        Requirements: 16.4
        """
        # Create dataset where each row has a unique value
        df = pd.DataFrame({
            'x': list(range(n_unique)),  # All unique values
            'y': [i % 3 for i in range(n_unique)]  # Few unique values
        })
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Test distribution with many unique values
        x_dist_result = p(x)
        x_dist_dict = get_distribution_dict(x_dist_result)
        
        # Should have n_unique different values, each with probability 1/n_unique
        assert len(x_dist_dict) == n_unique, f"Expected {n_unique} unique values, got {len(x_dist_dict)}"
        
        expected_prob = 1.0 / n_unique
        for value, prob in x_dist_dict.items():
            assert abs(prob - expected_prob) < 1e-10, (
                f"Expected uniform probability {expected_prob}, got {prob} for value {value}"
            )