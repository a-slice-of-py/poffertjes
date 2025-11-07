"""Unit tests for ProbabilityCalculator."""

import pytest
import pandas as pd
import narwhals as nw
from poffertjes.calculator import ProbabilityCalculator

# Try to import polars, skip tests if not available
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class TestProbabilityCalculatorStructure:
    """Test the basic structure and initialization of ProbabilityCalculator."""
    
    def test_init_with_pandas_dataframe(self):
        """Test initialization with a Pandas dataframe."""
        # Create test data
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': ['a', 'b', 'a', 'b', 'a']
        })
        
        # Convert to Narwhals frame
        nw_df = nw.from_native(df)
        
        # Initialize calculator
        calc = ProbabilityCalculator(nw_df)
        
        # Verify initialization
        assert calc.df is nw_df
        assert calc.total_count == 5
    
    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
    def test_init_with_polars_dataframe(self):
        """Test initialization with a Polars dataframe."""
        # Create test data
        df = pl.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': ['a', 'b', 'a', 'b', 'a']
        })
        
        # Convert to Narwhals frame
        nw_df = nw.from_native(df)
        
        # Initialize calculator
        calc = ProbabilityCalculator(nw_df)
        
        # Verify initialization
        assert calc.df is nw_df
        assert calc.total_count == 5
    
    def test_init_with_empty_dataframe(self):
        """Test initialization with an empty dataframe."""
        # Create empty test data
        df = pd.DataFrame({'x': [], 'y': []})
        nw_df = nw.from_native(df)
        
        # Initialize calculator
        calc = ProbabilityCalculator(nw_df)
        
        # Verify initialization
        assert calc.df is nw_df
        assert calc.total_count == 0
    
    def test_init_with_single_row_dataframe(self):
        """Test initialization with a single-row dataframe."""
        # Create single-row test data
        df = pd.DataFrame({'x': [1], 'y': ['a']})
        nw_df = nw.from_native(df)
        
        # Initialize calculator
        calc = ProbabilityCalculator(nw_df)
        
        # Verify initialization
        assert calc.df is nw_df
        assert calc.total_count == 1
    
    def test_total_count_calculation(self):
        """Test that total_count is calculated correctly for different dataframe sizes."""
        # Test with various sizes
        sizes = [0, 1, 5, 10, 100]
        
        for size in sizes:
            df = pd.DataFrame({'x': list(range(size))})
            nw_df = nw.from_native(df)
            calc = ProbabilityCalculator(nw_df)
            
            assert calc.total_count == size, f"Failed for size {size}"


class TestCalculateDistribution:
    """Test the calculate_distribution method for marginal probability distributions."""
    
    def setup_method(self):
        """Set up test data for distribution tests."""
        # Create test dataframe with known distribution
        self.df = pd.DataFrame({
            'x': [1, 1, 2, 2, 2, 3],  # 1: 2/6, 2: 3/6, 3: 1/6
            'y': ['a', 'b', 'a', 'a', 'b', 'a'],  # a: 4/6, b: 2/6
            'z': [10, 20, 10, 20, 10, 30]  # 10: 3/6, 20: 2/6, 30: 1/6
        })
        self.nw_df = nw.from_native(self.df)
        self.calc = ProbabilityCalculator(self.nw_df)
        
        # Create mock variables (we'll need to import Variable for real tests)
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(self.df)
        self.x, self.y, self.z = vb.get_variables('x', 'y', 'z')
    
    def test_single_variable_distribution(self):
        """Test marginal distribution P(X) for a single variable."""
        # Calculate P(X)
        result = self.calc.calculate_distribution([self.x])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'x' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # Verify values - should have 3 unique values of x
        assert len(result_native) == 3
        
        # Verify probabilities sum to 1.0 (requirement 4.4)
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10, f"Probabilities sum to {prob_sum}, not 1.0"
        
        # Verify specific probabilities
        result_dict = dict(zip(result_native['x'], result_native['probability']))
        assert abs(result_dict[1] - 2/6) < 1e-10  # P(X=1) = 2/6
        assert abs(result_dict[2] - 3/6) < 1e-10  # P(X=2) = 3/6
        assert abs(result_dict[3] - 1/6) < 1e-10  # P(X=3) = 1/6
    
    def test_single_variable_distribution_categorical(self):
        """Test marginal distribution P(Y) for a categorical variable."""
        # Calculate P(Y)
        result = self.calc.calculate_distribution([self.y])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'y' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # Verify values - should have 2 unique values of y
        assert len(result_native) == 2
        
        # Verify probabilities sum to 1.0
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify specific probabilities
        result_dict = dict(zip(result_native['y'], result_native['probability']))
        assert abs(result_dict['a'] - 4/6) < 1e-10  # P(Y='a') = 4/6
        assert abs(result_dict['b'] - 2/6) < 1e-10  # P(Y='b') = 2/6
    
    def test_joint_distribution_two_variables(self):
        """Test joint distribution P(X,Y) for two variables."""
        # Calculate P(X,Y)
        result = self.calc.calculate_distribution([self.x, self.y])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'x' in result_native.columns
        assert 'y' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # Verify probabilities sum to 1.0 (requirement 4.4)
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify we have the expected combinations
        # From our data: (1,'a'):1, (1,'b'):1, (2,'a'):2, (2,'b'):1, (3,'a'):1
        expected_combinations = {(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a')}
        actual_combinations = set(zip(result_native['x'], result_native['y']))
        assert actual_combinations == expected_combinations
        
        # Verify specific joint probabilities
        result_dict = {}
        for _, row in result_native.iterrows():
            result_dict[(row['x'], row['y'])] = row['probability']
        
        assert abs(result_dict[(1, 'a')] - 1/6) < 1e-10  # P(X=1,Y='a') = 1/6
        assert abs(result_dict[(1, 'b')] - 1/6) < 1e-10  # P(X=1,Y='b') = 1/6
        assert abs(result_dict[(2, 'a')] - 2/6) < 1e-10  # P(X=2,Y='a') = 2/6
        assert abs(result_dict[(2, 'b')] - 1/6) < 1e-10  # P(X=2,Y='b') = 1/6
        assert abs(result_dict[(3, 'a')] - 1/6) < 1e-10  # P(X=3,Y='a') = 1/6
    
    def test_joint_distribution_three_variables(self):
        """Test joint distribution P(X,Y,Z) for three variables."""
        # Calculate P(X,Y,Z)
        result = self.calc.calculate_distribution([self.x, self.y, self.z])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'x' in result_native.columns
        assert 'y' in result_native.columns
        assert 'z' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # Verify probabilities sum to 1.0
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Each row in original data should be a unique combination
        assert len(result_native) == 6  # All rows are unique combinations
        
        # Each combination should have probability 1/6
        for _, row in result_native.iterrows():
            assert abs(row['probability'] - 1/6) < 1e-10
    
    def test_distribution_with_conditions(self):
        """Test conditional distribution P(X|Y='a')."""
        from poffertjes.expression import Expression
        
        # Create condition Y = 'a'
        condition = Expression(self.y, "==", "a")
        
        # Calculate P(X|Y='a')
        result = self.calc.calculate_distribution([self.x], conditions=[condition])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'x' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # Verify probabilities sum to 1.0 (conditional probabilities should normalize)
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # From our data, when Y='a': X values are [1, 2, 2, 3] (4 total)
        # So P(X=1|Y='a') = 1/4, P(X=2|Y='a') = 2/4, P(X=3|Y='a') = 1/4
        result_dict = dict(zip(result_native['x'], result_native['probability']))
        assert abs(result_dict[1] - 1/4) < 1e-10
        assert abs(result_dict[2] - 2/4) < 1e-10
        assert abs(result_dict[3] - 1/4) < 1e-10
    
    def test_distribution_with_multiple_conditions(self):
        """Test conditional distribution with multiple conditions P(Z|X=2, Y='a')."""
        from poffertjes.expression import Expression
        
        # Create conditions X = 2 AND Y = 'a'
        condition1 = Expression(self.x, "==", 2)
        condition2 = Expression(self.y, "==", "a")
        
        # Calculate P(Z|X=2, Y='a')
        result = self.calc.calculate_distribution([self.z], conditions=[condition1, condition2])
        
        # Convert to native for easier testing
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # From our data, when X=2 AND Y='a': Z values are [10, 20] (2 total)
        # So P(Z=10|X=2,Y='a') = 1/2, P(Z=20|X=2,Y='a') = 1/2
        assert len(result_native) == 2
        
        # Verify probabilities sum to 1.0
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify specific probabilities
        result_dict = dict(zip(result_native['z'], result_native['probability']))
        assert abs(result_dict[10] - 1/2) < 1e-10
        assert abs(result_dict[20] - 1/2) < 1e-10
    
    def test_distribution_zero_probability_condition(self):
        """Test that zero probability conditioning raises appropriate error."""
        from poffertjes.expression import Expression
        
        # Create condition that matches no rows
        condition = Expression(self.x, "==", 999)  # No x values are 999
        
        # Should raise ValueError for zero probability conditioning
        with pytest.raises(ValueError, match="Conditioning event has zero probability"):
            self.calc.calculate_distribution([self.y], conditions=[condition])
    
    def test_distribution_empty_dataframe(self):
        """Test distribution calculation with empty dataframe."""
        # Create empty dataframe
        empty_df = pd.DataFrame({'x': [], 'y': []})
        nw_empty = nw.from_native(empty_df)
        calc = ProbabilityCalculator(nw_empty)
        
        # Create mock variable
        from poffertjes.variable import Variable
        x_var = Variable('x', nw_empty)
        
        # Calculate distribution - should return empty result
        result = calc.calculate_distribution([x_var])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Should have correct structure but no rows
        assert 'x' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        assert len(result_native) == 0
    
    def test_distribution_single_value_column(self):
        """Test distribution with column that has only one unique value."""
        # Create dataframe where x has only one value
        single_val_df = pd.DataFrame({
            'x': [5, 5, 5, 5],
            'y': ['a', 'b', 'a', 'b']
        })
        nw_single = nw.from_native(single_val_df)
        calc = ProbabilityCalculator(nw_single)
        
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(single_val_df)
        x_var = vb.get_variables('x')
        
        # Calculate P(X) - should be P(X=5) = 1.0
        result = calc.calculate_distribution([x_var])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        assert len(result_native) == 1
        assert result_native.iloc[0]['x'] == 5
        assert abs(result_native.iloc[0]['probability'] - 1.0) < 1e-10
    
    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
    def test_distribution_with_polars(self):
        """Test that distribution calculation works with Polars dataframes."""
        # Create Polars dataframe
        pl_df = pl.DataFrame({
            'x': [1, 1, 2, 2, 3],
            'y': ['a', 'b', 'a', 'b', 'a']
        })
        nw_df = nw.from_native(pl_df)
        calc = ProbabilityCalculator(nw_df)
        
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(pl_df)
        x_var = vb.get_variables('x')
        
        # Calculate P(X)
        result = calc.calculate_distribution([x_var])
        
        # Should work the same as with Pandas
        # Convert to pandas for easier testing
        result_pd = result.to_pandas()
        
        assert 'x' in result_pd.columns
        assert 'count' in result_pd.columns
        assert 'probability' in result_pd.columns
        
        # Verify probabilities sum to 1.0
        prob_sum = result_pd['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10


class TestCalculateScalar:
    """Test the calculate_scalar method for scalar probability calculations."""
    
    def setup_method(self):
        """Set up test data for scalar probability tests."""
        # Create test dataframe with known distribution
        self.df = pd.DataFrame({
            'x': [1, 1, 2, 2, 2, 3],  # 1: 2/6, 2: 3/6, 3: 1/6
            'y': ['a', 'b', 'a', 'a', 'b', 'a'],  # a: 4/6, b: 2/6
            'z': [10, 20, 10, 20, 10, 30]  # 10: 3/6, 20: 2/6, 30: 1/6
        })
        self.nw_df = nw.from_native(self.df)
        self.calc = ProbabilityCalculator(self.nw_df)
        
        # Create variables
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(self.df)
        self.x, self.y, self.z = vb.get_variables('x', 'y', 'z')
    
    def test_single_equality_expression(self):
        """Test P(X=value) for single equality expression."""
        from poffertjes.expression import Expression
        
        # Test P(X=1) = 2/6
        expr = Expression(self.x, "==", 1)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 2/6) < 1e-10
        
        # Test P(X=2) = 3/6
        expr = Expression(self.x, "==", 2)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 3/6) < 1e-10
        
        # Test P(X=3) = 1/6
        expr = Expression(self.x, "==", 3)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 1/6) < 1e-10
        
        # Test P(X=999) = 0 (value not in data)
        expr = Expression(self.x, "==", 999)
        prob = self.calc.calculate_scalar([expr])
        assert prob == 0.0
    
    def test_single_inequality_expressions(self):
        """Test P(X op value) for inequality operators."""
        from poffertjes.expression import Expression
        
        # Test P(X > 1) = 4/6 (values 2,2,2,3)
        expr = Expression(self.x, ">", 1)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 4/6) < 1e-10
        
        # Test P(X >= 2) = 4/6 (values 2,2,2,3)
        expr = Expression(self.x, ">=", 2)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 4/6) < 1e-10
        
        # Test P(X < 3) = 5/6 (values 1,1,2,2,2)
        expr = Expression(self.x, "<", 3)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 5/6) < 1e-10
        
        # Test P(X <= 2) = 5/6 (values 1,1,2,2,2)
        expr = Expression(self.x, "<=", 2)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 5/6) < 1e-10
        
        # Test P(X != 2) = 3/6 (values 1,1,3)
        expr = Expression(self.x, "!=", 2)
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 3/6) < 1e-10
    
    def test_categorical_expressions(self):
        """Test scalar probabilities with categorical variables."""
        from poffertjes.expression import Expression
        
        # Test P(Y='a') = 4/6
        expr = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 4/6) < 1e-10
        
        # Test P(Y='b') = 2/6
        expr = Expression(self.y, "==", "b")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 2/6) < 1e-10
        
        # Test P(Y != 'a') = 2/6
        expr = Expression(self.y, "!=", "a")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 2/6) < 1e-10
    
    def test_isin_expressions(self):
        """Test scalar probabilities with isin expressions."""
        from poffertjes.expression import Expression
        
        # Test P(X in [1,3]) = 3/6 (values 1,1,3)
        expr = Expression(self.x, "in", [1, 3])
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 3/6) < 1e-10
        
        # Test P(Y in ['a']) = 4/6
        expr = Expression(self.y, "in", ["a"])
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 4/6) < 1e-10
        
        # Test P(Z in [10, 30]) = 4/6 (values 10,10,10,30)
        expr = Expression(self.z, "in", [10, 30])
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 4/6) < 1e-10
    
    def test_ternary_expressions(self):
        """Test scalar probabilities with ternary (between) expressions."""
        from poffertjes.expression import TernaryExpression
        
        # Test P(0 < X < 3) = 5/6 (values 1,1,2,2,2) - exclusive bounds
        expr = TernaryExpression(self.x, 0, 3, closed="none")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 5/6) < 1e-10
        
        # Test P(1 <= X <= 2) = 5/6 (values 1,1,2,2,2) - inclusive bounds
        expr = TernaryExpression(self.x, 1, 2, closed="both")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 5/6) < 1e-10
        
        # Test P(1 < X < 3) = 3/6 (values 2,2,2) - exclusive bounds
        expr = TernaryExpression(self.x, 1, 3, closed="none")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 3/6) < 1e-10
        
        # Test P(15 < Z < 25) = 2/6 (values 20, 20) - exclusive bounds
        expr = TernaryExpression(self.z, 15, 25, closed="none")
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 2/6) < 1e-10
    
    def test_multiple_expressions_and_logic(self):
        """Test scalar probabilities with multiple expressions (AND logic)."""
        from poffertjes.expression import Expression
        
        # Test P(X=2 AND Y='a') = 2/6 (two rows match both conditions)
        expr1 = Expression(self.x, "==", 2)
        expr2 = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr1, expr2])
        assert abs(prob - 2/6) < 1e-10
        
        # Test P(X>1 AND Y='b') = 1/6 (one row: x=2, y='b')
        expr1 = Expression(self.x, ">", 1)
        expr2 = Expression(self.y, "==", "b")
        prob = self.calc.calculate_scalar([expr1, expr2])
        assert abs(prob - 1/6) < 1e-10
        
        # Test P(X=1 AND Y='a' AND Z=10) = 1/6 (one row matches all)
        expr1 = Expression(self.x, "==", 1)
        expr2 = Expression(self.y, "==", "a")
        expr3 = Expression(self.z, "==", 10)
        prob = self.calc.calculate_scalar([expr1, expr2, expr3])
        assert abs(prob - 1/6) < 1e-10
        
        # Test P(X=999 AND Y='a') = 0 (no rows match)
        expr1 = Expression(self.x, "==", 999)
        expr2 = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr1, expr2])
        assert prob == 0.0
    
    def test_composite_expressions(self):
        """Test scalar probabilities with CompositeExpression objects."""
        from poffertjes.expression import Expression, CompositeExpression
        
        # Test P((X=1) OR (X=3)) = 3/6 using CompositeExpression
        expr1 = Expression(self.x, "==", 1)
        expr2 = Expression(self.x, "==", 3)
        composite = CompositeExpression([expr1, expr2], "OR")
        prob = self.calc.calculate_scalar([composite])
        assert abs(prob - 3/6) < 1e-10
        
        # Test P((X=2) AND (Y='a')) = 2/6 using CompositeExpression
        expr1 = Expression(self.x, "==", 2)
        expr2 = Expression(self.y, "==", "a")
        composite = CompositeExpression([expr1, expr2], "AND")
        prob = self.calc.calculate_scalar([composite])
        assert abs(prob - 2/6) < 1e-10
        
        # Test complex composite: P((X=1 OR X=2) AND Y='a') = 3/6
        expr1 = Expression(self.x, "==", 1)
        expr2 = Expression(self.x, "==", 2)
        or_composite = CompositeExpression([expr1, expr2], "OR")
        expr3 = Expression(self.y, "==", "a")
        and_composite = CompositeExpression([or_composite, expr3], "AND")
        prob = self.calc.calculate_scalar([and_composite])
        assert abs(prob - 3/6) < 1e-10
    
    def test_conditional_scalar_probabilities(self):
        """Test conditional scalar probabilities P(expressions|conditions)."""
        from poffertjes.expression import Expression
        
        # Test P(X=2|Y='a') = 2/4 = 0.5
        # When Y='a', we have 4 rows: X values [1,2,2,3]
        # Of these, 2 have X=2, so P(X=2|Y='a') = 2/4
        expr = Expression(self.x, "==", 2)
        condition = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        assert abs(prob - 2/4) < 1e-10
        
        # Test P(Z=10|X=2) = 1/3
        # When X=2, we have 3 rows: Z values [10,20,10]
        # Of these, 2 have Z=10, so P(Z=10|X=2) = 2/3
        expr = Expression(self.z, "==", 10)
        condition = Expression(self.x, "==", 2)
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        assert abs(prob - 2/3) < 1e-10
        
        # Test P(Y='b'|X>1) = 1/4
        # When X>1, we have 4 rows: Y values ['a','a','b','a']
        # Of these, 1 has Y='b', so P(Y='b'|X>1) = 1/4
        expr = Expression(self.y, "==", "b")
        condition = Expression(self.x, ">", 1)
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        assert abs(prob - 1/4) < 1e-10
    
    def test_conditional_with_multiple_conditions(self):
        """Test conditional probabilities with multiple conditioning expressions."""
        from poffertjes.expression import Expression
        
        # Test P(Z=10|X=2, Y='a') = 1/2
        # When X=2 AND Y='a', we have 2 rows: Z values [10,20]
        # Of these, 1 has Z=10, so P(Z=10|X=2,Y='a') = 1/2
        expr = Expression(self.z, "==", 10)
        condition1 = Expression(self.x, "==", 2)
        condition2 = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr], conditions=[condition1, condition2])
        assert abs(prob - 1/2) < 1e-10
        
        # Test P(X=1|Y='a', Z=10) = 1/2
        # When Y='a' AND Z=10, we have 2 rows: X values [1,2]
        # Of these, 1 has X=1, so P(X=1|Y='a',Z=10) = 1/2
        expr = Expression(self.x, "==", 1)
        condition1 = Expression(self.y, "==", "a")
        condition2 = Expression(self.z, "==", 10)
        prob = self.calc.calculate_scalar([expr], conditions=[condition1, condition2])
        assert abs(prob - 1/2) < 1e-10
    
    def test_zero_probability_conditioning(self):
        """Test that zero probability conditioning raises appropriate error."""
        from poffertjes.expression import Expression
        
        # Create condition that matches no rows
        expr = Expression(self.x, "==", 1)
        condition = Expression(self.x, "==", 999)  # No x values are 999
        
        # Should raise ValueError for zero probability conditioning
        with pytest.raises(ValueError, match="Conditioning event has zero probability"):
            self.calc.calculate_scalar([expr], conditions=[condition])
    
    def test_scalar_with_empty_dataframe(self):
        """Test scalar calculation with empty dataframe."""
        # Create empty dataframe
        empty_df = pd.DataFrame({'x': [], 'y': []})
        nw_empty = nw.from_native(empty_df)
        calc = ProbabilityCalculator(nw_empty)
        
        # Create expression directly (VariableBuilder raises error for empty dataframes)
        from poffertjes.variable import Variable
        from poffertjes.expression import Expression
        x_var = Variable('x', nw_empty)
        expr = Expression(x_var, "==", 1)
        
        # Should return 0.0 for empty dataframe
        prob = calc.calculate_scalar([expr])
        assert prob == 0.0
    
    def test_scalar_edge_cases(self):
        """Test scalar calculation edge cases."""
        from poffertjes.expression import Expression
        
        # Test when all rows match the condition
        expr = Expression(self.x, ">=", 1)  # All x values are >= 1
        prob = self.calc.calculate_scalar([expr])
        assert abs(prob - 1.0) < 1e-10
        
        # Test when no rows match the condition
        expr = Expression(self.x, ">", 100)  # No x values are > 100
        prob = self.calc.calculate_scalar([expr])
        assert prob == 0.0
        
        # Test with single-value column
        single_val_df = pd.DataFrame({'x': [5, 5, 5, 5]})
        nw_single = nw.from_native(single_val_df)
        calc = ProbabilityCalculator(nw_single)
        
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(single_val_df)
        x_var = vb.get_variables('x')
        
        # P(X=5) should be 1.0
        expr = Expression(x_var, "==", 5)
        prob = calc.calculate_scalar([expr])
        assert abs(prob - 1.0) < 1e-10
        
        # P(X=999) should be 0.0
        expr = Expression(x_var, "==", 999)
        prob = calc.calculate_scalar([expr])
        assert prob == 0.0
    
    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
    def test_scalar_with_polars(self):
        """Test that scalar calculation works with Polars dataframes."""
        # Create Polars dataframe
        pl_df = pl.DataFrame({
            'x': [1, 1, 2, 2, 3],
            'y': ['a', 'b', 'a', 'b', 'a']
        })
        nw_df = nw.from_native(pl_df)
        calc = ProbabilityCalculator(nw_df)
        
        from poffertjes.variable import VariableBuilder
        from poffertjes.expression import Expression
        vb = VariableBuilder.from_data(pl_df)
        x_var = vb.get_variables('x')
        
        # Test P(X=2) = 2/5
        expr = Expression(x_var, "==", 2)
        prob = calc.calculate_scalar([expr])
        assert abs(prob - 2/5) < 1e-10
        
        # Should work the same as with Pandas
        y_var = vb.get_variables('y')
        expr1 = Expression(x_var, "==", 2)
        expr2 = Expression(y_var, "==", "a")
        prob = calc.calculate_scalar([expr1, expr2])
        assert abs(prob - 1/5) < 1e-10  # One row matches both conditions


class TestConditionalProbabilities:
    """Comprehensive tests for conditional probability support (Task 6.4)."""
    
    def setup_method(self):
        """Set up test data for conditional probability tests."""
        # Create test dataframe with known conditional relationships
        self.df = pd.DataFrame({
            'x': [1, 1, 2, 2, 2, 3, 3, 4],  # 1: 2/8, 2: 3/8, 3: 2/8, 4: 1/8
            'y': ['a', 'b', 'a', 'a', 'b', 'a', 'b', 'a'],  # a: 5/8, b: 3/8
            'z': [10, 20, 10, 20, 10, 30, 30, 40]  # Various values
        })
        self.nw_df = nw.from_native(self.df)
        self.calc = ProbabilityCalculator(self.nw_df)
        
        # Create variables
        from poffertjes.variable import VariableBuilder
        vb = VariableBuilder.from_data(self.df)
        self.x, self.y, self.z = vb.get_variables('x', 'y', 'z')
    
    def test_conditional_distribution_single_condition(self):
        """Test P(X|Y='a') - Requirement 5.1, 5.3."""
        from poffertjes.expression import Expression
        
        # Create condition Y = 'a'
        condition = Expression(self.y, "==", "a")
        
        # Calculate P(X|Y='a')
        result = self.calc.calculate_distribution([self.x], conditions=[condition])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # Verify structure
        assert 'x' in result_native.columns
        assert 'count' in result_native.columns
        assert 'probability' in result_native.columns
        
        # When Y='a', we have rows: x=[1,2,2,3,4] (5 total)
        # P(X=1|Y='a') = 1/5, P(X=2|Y='a') = 2/5, P(X=3|Y='a') = 1/5, P(X=4|Y='a') = 1/5
        
        # Verify probabilities sum to 1.0 (Requirement 5.8)
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify specific conditional probabilities
        result_dict = dict(zip(result_native['x'], result_native['probability']))
        assert abs(result_dict[1] - 1/5) < 1e-10
        assert abs(result_dict[2] - 2/5) < 1e-10
        assert abs(result_dict[3] - 1/5) < 1e-10
        assert abs(result_dict[4] - 1/5) < 1e-10
    
    def test_conditional_distribution_multiple_conditions(self):
        """Test P(Z|X=2, Y='a') - Requirement 5.4, 5.5."""
        from poffertjes.expression import Expression
        
        # Create conditions X = 2 AND Y = 'a'
        condition1 = Expression(self.x, "==", 2)
        condition2 = Expression(self.y, "==", "a")
        
        # Calculate P(Z|X=2, Y='a')
        result = self.calc.calculate_distribution([self.z], conditions=[condition1, condition2])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # When X=2 AND Y='a', we have rows: z=[10,20] (2 total)
        # P(Z=10|X=2,Y='a') = 1/2, P(Z=20|X=2,Y='a') = 1/2
        
        assert len(result_native) == 2
        
        # Verify probabilities sum to 1.0 (Requirement 5.8)
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify specific probabilities
        result_dict = dict(zip(result_native['z'], result_native['probability']))
        assert abs(result_dict[10] - 1/2) < 1e-10
        assert abs(result_dict[20] - 1/2) < 1e-10
    
    def test_conditional_joint_distribution(self):
        """Test P(X,Z|Y='b') - joint conditional distribution."""
        from poffertjes.expression import Expression
        
        # Create condition Y = 'b'
        condition = Expression(self.y, "==", "b")
        
        # Calculate P(X,Z|Y='b')
        result = self.calc.calculate_distribution([self.x, self.z], conditions=[condition])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        # When Y='b', we have rows: (x,z)=[(1,20), (2,10), (3,30)] (3 total)
        # Each combination should have probability 1/3
        
        assert len(result_native) == 3
        
        # Verify probabilities sum to 1.0
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Verify each combination has probability 1/3
        for _, row in result_native.iterrows():
            assert abs(row['probability'] - 1/3) < 1e-10
    
    def test_conditional_scalar_probability(self):
        """Test P(X=2|Y='a') - Requirement 5.2."""
        from poffertjes.expression import Expression
        
        # Create expression and condition
        expr = Expression(self.x, "==", 2)
        condition = Expression(self.y, "==", "a")
        
        # Calculate P(X=2|Y='a')
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        
        # When Y='a', we have 5 rows with x=[1,2,2,3,4]
        # Of these, 2 have X=2, so P(X=2|Y='a') = 2/5
        assert abs(prob - 2/5) < 1e-10
    
    def test_conditional_scalar_multiple_conditions(self):
        """Test P(Z=10|X=2, Y='a') - Requirement 5.5."""
        from poffertjes.expression import Expression
        
        # Create expression and conditions
        expr = Expression(self.z, "==", 10)
        condition1 = Expression(self.x, "==", 2)
        condition2 = Expression(self.y, "==", "a")
        
        # Calculate P(Z=10|X=2, Y='a')
        prob = self.calc.calculate_scalar([expr], conditions=[condition1, condition2])
        
        # When X=2 AND Y='a', we have 2 rows with z=[10,20]
        # Of these, 1 has Z=10, so P(Z=10|X=2,Y='a') = 1/2
        assert abs(prob - 1/2) < 1e-10
    
    def test_conditional_with_comparison_operators(self):
        """Test conditional probabilities with comparison operators."""
        from poffertjes.expression import Expression
        
        # Test P(X>2|Y='a')
        expr = Expression(self.x, ">", 2)
        condition = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        
        # When Y='a', we have x=[1,2,2,3,4] (5 total)
        # Of these, 2 have X>2 (values 3,4), so P(X>2|Y='a') = 2/5
        assert abs(prob - 2/5) < 1e-10
        
        # Test P(Z<=20|X>=2)
        expr = Expression(self.z, "<=", 20)
        condition = Expression(self.x, ">=", 2)
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        
        # When X>=2, we have 6 rows with z=[10,20,10,30,30,40]
        # Of these, 3 have Z<=20 (values 10,20,10), so P(Z<=20|X>=2) = 3/6 = 1/2
        assert abs(prob - 1/2) < 1e-10
    
    def test_conditional_with_isin_expressions(self):
        """Test conditional probabilities with isin expressions."""
        from poffertjes.expression import Expression
        
        # Test P(X in [1,3]|Y='a')
        expr = Expression(self.x, "in", [1, 3])
        condition = Expression(self.y, "==", "a")
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        
        # When Y='a', we have x=[1,2,2,3,4] (5 total)
        # Of these, 2 have X in [1,3] (values 1,3), so P(X in [1,3]|Y='a') = 2/5
        assert abs(prob - 2/5) < 1e-10
    
    def test_zero_probability_conditioning_distribution(self):
        """Test zero probability conditioning for distributions - Requirement 5.7."""
        from poffertjes.expression import Expression
        
        # Create condition that matches no rows
        condition = Expression(self.x, "==", 999)  # No x values are 999
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="Conditioning event has zero probability"):
            self.calc.calculate_distribution([self.y], conditions=[condition])
    
    def test_zero_probability_conditioning_scalar(self):
        """Test zero probability conditioning for scalar probabilities - Requirement 5.7."""
        from poffertjes.expression import Expression
        
        # Create expression and impossible condition
        expr = Expression(self.x, "==", 1)
        condition = Expression(self.x, "==", 999)  # No x values are 999
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="Conditioning event has zero probability"):
            self.calc.calculate_scalar([expr], conditions=[condition])
    
    def test_conditional_probability_axioms(self):
        """Test that conditional probabilities satisfy basic axioms."""
        from poffertjes.expression import Expression
        
        # Test that P(X=x_i|Y=y) sums to 1 for all x_i (Requirement 5.8)
        condition = Expression(self.y, "==", "a")
        result = self.calc.calculate_distribution([self.x], conditions=[condition])
        result_native = result.to_pandas() if hasattr(result, 'to_pandas') else result.to_native()
        
        prob_sum = result_native['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Test that P(X=x, Y=y) = P(X=x|Y=y) * P(Y=y)
        # Choose specific values
        x_val, y_val = 2, 'a'
        
        # Calculate P(X=2, Y='a')
        expr1 = Expression(self.x, "==", x_val)
        expr2 = Expression(self.y, "==", y_val)
        joint_prob = self.calc.calculate_scalar([expr1, expr2])
        
        # Calculate P(X=2|Y='a')
        conditional_prob = self.calc.calculate_scalar([expr1], conditions=[expr2])
        
        # Calculate P(Y='a')
        marginal_prob = self.calc.calculate_scalar([expr2])
        
        # Verify P(X=2, Y='a') = P(X=2|Y='a') * P(Y='a')
        expected_joint = conditional_prob * marginal_prob
        assert abs(joint_prob - expected_joint) < 1e-10
    
    def test_conditional_with_complex_expressions(self):
        """Test conditional probabilities with complex composite expressions."""
        from poffertjes.expression import Expression, CompositeExpression
        
        # Test P((X=1 OR X=3)|Y='a')
        expr1 = Expression(self.x, "==", 1)
        expr2 = Expression(self.x, "==", 3)
        composite = CompositeExpression([expr1, expr2], "OR")
        condition = Expression(self.y, "==", "a")
        
        prob = self.calc.calculate_scalar([composite], conditions=[condition])
        
        # When Y='a', we have x=[1,2,2,3,4] (5 total)
        # Of these, 2 have X=1 OR X=3 (values 1,3), so probability = 2/5
        assert abs(prob - 2/5) < 1e-10
    
    def test_nested_conditional_probabilities(self):
        """Test multiple levels of conditioning."""
        from poffertjes.expression import Expression
        
        # Test P(Z=10|X=2, Y='a', Z<=20)
        expr = Expression(self.z, "==", 10)
        condition1 = Expression(self.x, "==", 2)
        condition2 = Expression(self.y, "==", "a")
        condition3 = Expression(self.z, "<=", 20)
        
        prob = self.calc.calculate_scalar([expr], conditions=[condition1, condition2, condition3])
        
        # When X=2 AND Y='a' AND Z<=20, we have z=[10,20] (2 total)
        # Of these, 1 has Z=10, so probability = 1/2
        assert abs(prob - 1/2) < 1e-10
    
    def test_conditional_edge_cases(self):
        """Test edge cases for conditional probabilities."""
        from poffertjes.expression import Expression
        
        # Test conditioning on always-true condition
        condition = Expression(self.x, ">=", 0)  # All x values are >= 0
        expr = Expression(self.y, "==", "a")
        
        # P(Y='a'|X>=0) should equal P(Y='a') since condition is always true
        conditional_prob = self.calc.calculate_scalar([expr], conditions=[condition])
        marginal_prob = self.calc.calculate_scalar([expr])
        
        assert abs(conditional_prob - marginal_prob) < 1e-10
        
        # Test conditioning on the same variable
        condition = Expression(self.x, "==", 2)
        expr = Expression(self.x, "==", 2)
        
        # P(X=2|X=2) should be 1.0
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        assert abs(prob - 1.0) < 1e-10
        
        # Test conditioning on different value of same variable
        condition = Expression(self.x, "==", 2)
        expr = Expression(self.x, "==", 1)
        
        # P(X=1|X=2) should be 0.0
        prob = self.calc.calculate_scalar([expr], conditions=[condition])
        assert prob == 0.0
    
    def test_conditional_with_missing_values(self):
        """Test conditional probabilities with missing values."""
        # Create dataframe with NaN values
        df_with_nan = pd.DataFrame({
            'x': [1, 1, 2, None, 2, 3],
            'y': ['a', 'b', 'a', 'a', None, 'a']
        })
        nw_df_nan = nw.from_native(df_with_nan)
        calc_nan = ProbabilityCalculator(nw_df_nan)
        
        from poffertjes.variable import VariableBuilder
        from poffertjes.expression import Expression
        
        vb = VariableBuilder.from_data(df_with_nan)
        x_var, y_var = vb.get_variables('x', 'y')
        
        # Test P(X=2|Y='a') - should exclude rows with NaN
        expr = Expression(x_var, "==", 2)
        condition = Expression(y_var, "==", "a")
        
        # When Y='a', we have x=[1,2,NaN,3] (4 total)
        # Of these, 1 has X=2, so P(X=2|Y='a') = 1/4
        prob = calc_nan.calculate_scalar([expr], conditions=[condition])
        assert abs(prob - 1/4) < 1e-10
    
    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
    def test_conditional_probabilities_with_polars(self):
        """Test that conditional probabilities work with Polars dataframes."""
        # Create Polars dataframe
        pl_df = pl.DataFrame({
            'x': [1, 1, 2, 2, 3],
            'y': ['a', 'b', 'a', 'b', 'a']
        })
        nw_df = nw.from_native(pl_df)
        calc = ProbabilityCalculator(nw_df)
        
        from poffertjes.variable import VariableBuilder
        from poffertjes.expression import Expression
        
        vb = VariableBuilder.from_data(pl_df)
        x_var, y_var = vb.get_variables('x', 'y')
        
        # Test P(X=2|Y='a')
        expr = Expression(x_var, "==", 2)
        condition = Expression(y_var, "==", "a")
        prob = calc.calculate_scalar([expr], conditions=[condition])
        
        # When Y='a', we have x=[1,2,3] (3 total)
        # Of these, 1 has X=2, so P(X=2|Y='a') = 1/3
        assert abs(prob - 1/3) < 1e-10
        
        # Test conditional distribution
        result = calc.calculate_distribution([x_var], conditions=[condition])
        result_pd = result.to_pandas()
        
        # Should work the same as with Pandas
        prob_sum = result_pd['probability'].sum()
        assert abs(prob_sum - 1.0) < 1e-10
