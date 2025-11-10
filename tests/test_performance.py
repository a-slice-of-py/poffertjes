"""Performance tests for poffertjes library.

This module contains tests that benchmark performance with large datasets
and verify memory efficiency of the probability calculations.

Requirements addressed:
- 7.14: Benchmark with large datasets
- 16.2: Verify memory efficiency
- 16.4: Handle very large datasets efficiently
"""

import time
import psutil
import os
import pytest
import pandas as pd
import polars as pl
import numpy as np

from poffertjes import p
from poffertjes.variable import VariableBuilder


class MemoryMonitor:
    """Helper class to monitor memory usage during tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_mb()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_increase(self) -> float:
        """Get memory increase since initialization in MB."""
        return self.get_memory_mb() - self.initial_memory


def generate_large_dataset(n_rows: int, n_categorical: int = 3) -> pd.DataFrame:
    """Generate a large dataset for performance testing.
    
    Args:
        n_rows: Number of rows to generate
        n_categorical: Number of categories for categorical variables
        
    Returns:
        DataFrame with mixed data types for testing
    """
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'x': np.random.randint(0, 10, n_rows),
        'y': np.random.choice(['A', 'B', 'C'], n_rows),
        'z': np.random.normal(0, 1, n_rows),
        'category': np.random.choice([f'cat_{i}' for i in range(n_categorical)], n_rows),
        'boolean': np.random.choice([True, False], n_rows),
    }
    
    return pd.DataFrame(data)


class TestPerformanceBenchmarks:
    """Test performance with large datasets."""
    
    @pytest.mark.parametrize("n_rows", [10_000, 100_000])
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_large_dataset_marginal_performance(self, n_rows: int, backend: str):
        """Test performance of marginal probability calculations on large datasets.
        
        Requirements addressed:
        - 7.14: Benchmark with large datasets
        - 16.2: Verify memory efficiency
        """
        # Generate test data
        df_pandas = generate_large_dataset(n_rows)
        
        if backend == "polars":
            df = pl.from_pandas(df_pandas)
        else:
            df = df_pandas
        
        # Create variables
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # Monitor memory and time
        memory_monitor = MemoryMonitor()
        start_time = time.time()
        
        # Perform calculations
        dist_x = p(x)
        dist_y = p(y)
        dist_z = p(z)
        
        # Convert to dict to force evaluation
        dict_x = dist_x.to_dict()
        dict_y = dist_y.to_dict()
        dict_z = dist_z.to_dict()
        
        end_time = time.time()
        memory_increase = memory_monitor.get_memory_increase()
        
        # Performance assertions
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust thresholds as needed)
        if n_rows == 10_000:
            assert execution_time < 5.0, f"Execution took {execution_time:.2f}s for {n_rows} rows"
        else:  # 100_000 rows
            assert execution_time < 30.0, f"Execution took {execution_time:.2f}s for {n_rows} rows"
        
        # Memory usage should be reasonable (less than 500MB increase for 100k rows)
        max_memory_mb = 500 if n_rows == 100_000 else 100
        assert memory_increase < max_memory_mb, f"Memory increased by {memory_increase:.1f}MB"
        
        # Verify results are correct
        assert len(dict_x) > 0
        assert len(dict_y) > 0
        assert len(dict_z) > 0
        
        # Probabilities should sum to 1.0
        assert abs(sum(dict_x.values()) - 1.0) < 1e-10
        assert abs(sum(dict_y.values()) - 1.0) < 1e-10
        assert abs(sum(dict_z.values()) - 1.0) < 1e-10

    @pytest.mark.parametrize("n_rows", [10_000, 50_000])
    def test_large_dataset_conditional_performance(self, n_rows: int):
        """Test performance of conditional probability calculations on large datasets.
        
        Requirements addressed:
        - 7.14: Benchmark with large datasets
        - 16.4: Handle very large datasets efficiently
        """
        # Generate test data
        df = generate_large_dataset(n_rows)
        
        # Create variables
        vb = VariableBuilder.from_data(df)
        x, y, category = vb.get_variables('x', 'y', 'category')
        
        # Monitor memory and time
        memory_monitor = MemoryMonitor()
        start_time = time.time()
        
        # Perform conditional calculations
        cond_dist = p(x).given(y == 'A')
        cond_scalar = p(x == 5).given(y == 'A')
        joint_cond = p(x, category).given(y == 'B')
        
        # Force evaluation
        cond_dict = cond_dist.to_dict()
        scalar_val = float(cond_scalar)
        joint_dict = joint_cond.to_dict()
        
        end_time = time.time()
        memory_increase = memory_monitor.get_memory_increase()
        
        # Performance assertions
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        if n_rows == 10_000:
            assert execution_time < 10.0, f"Execution took {execution_time:.2f}s for {n_rows} rows"
        else:  # 50_000 rows
            assert execution_time < 60.0, f"Execution took {execution_time:.2f}s for {n_rows} rows"
        
        # Memory usage should be reasonable
        max_memory_mb = 300 if n_rows == 50_000 else 150
        assert memory_increase < max_memory_mb, f"Memory increased by {memory_increase:.1f}MB"
        
        # Verify results are correct
        assert len(cond_dict) > 0
        assert 0.0 <= scalar_val <= 1.0
        assert len(joint_dict) > 0
        
        # Conditional probabilities should sum to 1.0
        assert abs(sum(cond_dict.values()) - 1.0) < 1e-10
        assert abs(sum(joint_dict.values()) - 1.0) < 1e-10

    def test_caching_performance_improvement(self):
        """Test that caching improves performance for repeated calculations.
        
        Requirements addressed:
        - 7.13: Reuse computations where possible
        """
        # Generate test data
        df = generate_large_dataset(20_000)
        
        # Create calculator
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # First calculation (cold cache)
        start_time = time.time()
        dist1 = p(x)
        dict1 = dist1.to_dict()
        first_time = time.time() - start_time
        
        # Second calculation (should use cache)
        start_time = time.time()
        dist2 = p(x)
        dict2 = dist2.to_dict()
        second_time = time.time() - start_time
        
        # Third calculation with same variables (should use cache)
        start_time = time.time()
        dist3 = p(x, y)
        dict3 = dist3.to_dict()
        third_time = time.time() - start_time
        
        # Fourth calculation of same joint distribution (should use cache)
        start_time = time.time()
        dist4 = p(x, y)
        dict4 = dist4.to_dict()
        fourth_time = time.time() - start_time
        
        # Verify results are identical
        assert dict1 == dict2
        assert dict3 == dict4
        
        # Second calculation should be faster (cached) or at least not slower
        # For small datasets, the improvement might be minimal due to overhead
        # but it should not be significantly slower
        print(f"First calculation: {first_time:.4f}s, Second (cached): {second_time:.4f}s")
        print(f"Third calculation: {third_time:.4f}s, Fourth (cached): {fourth_time:.4f}s")
        
        # Allow for some variance in timing, but cached should not be much slower
        # Use a more lenient threshold since caching overhead might be significant for small operations
        assert second_time <= first_time * 1.5, f"Cached calculation much slower: {second_time:.3f}s vs {first_time:.3f}s"
        assert fourth_time <= third_time * 1.5, f"Cached joint calculation much slower: {fourth_time:.3f}s vs {third_time:.3f}s"
        
        # For larger datasets, we should see some improvement
        if first_time > 0.01:  # Only check improvement for operations that take meaningful time
            assert second_time < first_time * 0.8, f"No caching improvement for substantial operation: {second_time:.3f}s vs {first_time:.3f}s"

    def test_caching_with_large_dataset(self):
        """Test caching performance improvement with larger datasets where the effect is more visible.
        
        Requirements addressed:
        - 7.13: Reuse computations where possible
        """
        # Use a larger dataset where caching effects are more pronounced
        df = generate_large_dataset(100_000)
        
        vb = VariableBuilder.from_data(df)
        x, y, z = vb.get_variables('x', 'y', 'z')
        
        # First calculation (cold cache)
        start_time = time.time()
        dist1 = p(x, y)
        dict1 = dist1.to_dict()
        first_time = time.time() - start_time
        
        # Second calculation (should use cache)
        start_time = time.time()
        dist2 = p(x, y)
        dict2 = dist2.to_dict()
        second_time = time.time() - start_time
        
        # Verify results are identical
        assert dict1 == dict2
        
        print(f"Large dataset - First: {first_time:.4f}s, Cached: {second_time:.4f}s")
        
        # With larger datasets, caching should show clear improvement
        if first_time > 0.1:  # Only test if the operation takes substantial time
            improvement_ratio = second_time / first_time
            assert improvement_ratio < 0.7, f"Insufficient caching improvement: {improvement_ratio:.2f} (cached: {second_time:.3f}s vs original: {first_time:.3f}s)"

    def test_memory_efficiency_with_many_variables(self):
        """Test memory efficiency when working with many variables.
        
        Requirements addressed:
        - 16.2: Verify memory efficiency
        - 16.4: Handle very large datasets efficiently
        """
        # Generate dataset with many columns
        n_rows = 10_000
        n_cols = 20
        
        np.random.seed(42)
        data = {}
        for i in range(n_cols):
            data[f'var_{i}'] = np.random.randint(0, 5, n_rows)
        
        df = pd.DataFrame(data)
        
        # Create variables
        vb = VariableBuilder.from_data(df)
        variables = vb.get_variables(*[f'var_{i}' for i in range(n_cols)])
        
        # Monitor memory
        memory_monitor = MemoryMonitor()
        
        # Calculate marginals for all variables
        distributions = []
        for var in variables:
            dist = p(var)
            distributions.append(dist.to_dict())
        
        memory_increase = memory_monitor.get_memory_increase()
        
        # Memory increase should be reasonable (less than 200MB for this test)
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB for {n_cols} variables"
        
        # Verify all distributions are valid
        for dist_dict in distributions:
            assert len(dist_dict) > 0
            assert abs(sum(dist_dict.values()) - 1.0) < 1e-10

    def test_lazy_evaluation_memory_efficiency(self):
        """Test that lazy evaluation keeps memory usage low.
        
        Requirements addressed:
        - 7.1: Prefer Narwhals lazy operations
        - 7.11: Avoid unnecessary .collect() calls
        - 7.12: Maintain laziness until final result computation
        """
        # Create a large Polars LazyFrame to test lazy evaluation
        n_rows = 100_000
        
        # Generate data as Polars LazyFrame
        df_pandas = generate_large_dataset(n_rows)
        df_polars = pl.from_pandas(df_pandas)
        df_lazy = df_polars.lazy()
        
        # Create variables from lazy frame
        vb = VariableBuilder.from_data(df_lazy)
        x, y = vb.get_variables('x', 'y')
        
        # Monitor memory during operations
        memory_monitor = MemoryMonitor()
        
        # Perform operations that should remain lazy
        # These should not significantly increase memory until evaluation
        query_result = p(x).given(y == 'A')
        
        # Memory should not increase much yet (lazy evaluation)
        memory_after_query = memory_monitor.get_memory_increase()
        
        # Now force evaluation
        result_dict = query_result.to_dict()
        
        # Memory increase should be reasonable
        memory_after_eval = memory_monitor.get_memory_increase()
        
        # The lazy operations should not have consumed much memory
        assert memory_after_query < 50, f"Lazy operations used {memory_after_query:.1f}MB"
        
        # Even after evaluation, memory should be reasonable
        assert memory_after_eval < 300, f"Total memory usage {memory_after_eval:.1f}MB"
        
        # Verify result is correct
        assert len(result_dict) > 0
        assert abs(sum(result_dict.values()) - 1.0) < 1e-10


class TestScalabilityEdgeCases:
    """Test edge cases related to scalability and performance."""
    
    def test_very_small_dataset_performance(self):
        """Test performance with very small datasets.
        
        Requirements addressed:
        - 16.9: Handle very small datasets correctly
        """
        # Create minimal dataset
        df = pd.DataFrame({
            'x': [1, 2],
            'y': ['A', 'B']
        })
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('x', 'y')
        
        # Should work quickly even with tiny data
        start_time = time.time()
        
        dist = p(x)
        scalar = p(x == 1)
        cond = p(x).given(y == 'A')
        
        # Force evaluation
        dist_dict = dist.to_dict()
        scalar_val = float(scalar)
        cond_dict = cond.to_dict()
        
        execution_time = time.time() - start_time
        
        # Should be very fast
        assert execution_time < 1.0, f"Small dataset took {execution_time:.3f}s"
        
        # Results should be correct
        assert len(dist_dict) == 2
        assert scalar_val == 0.5  # 1 out of 2 rows
        assert len(cond_dict) == 1

    def test_single_value_column_performance(self):
        """Test performance with columns that have only one unique value.
        
        Requirements addressed:
        - 16.1: Handle single value columns correctly
        """
        # Create dataset with single-value column
        df = pd.DataFrame({
            'constant': ['A'] * 10_000,
            'variable': np.random.randint(0, 10, 10_000)
        })
        
        vb = VariableBuilder.from_data(df)
        constant, variable = vb.get_variables('constant', 'variable')
        
        start_time = time.time()
        
        # These should handle single-value columns efficiently
        const_dist = p(constant)
        var_dist = p(variable)
        joint_dist = p(constant, variable)
        
        # Force evaluation
        const_dict = const_dist.to_dict()
        var_dict = var_dist.to_dict()
        joint_dict = joint_dist.to_dict()
        
        execution_time = time.time() - start_time
        
        # Should complete quickly
        assert execution_time < 5.0, f"Single-value column test took {execution_time:.3f}s"
        
        # Verify results
        assert len(const_dict) == 1
        assert const_dict['A'] == 1.0
        assert len(var_dict) > 1
        assert len(joint_dict) == len(var_dict)  # Same as variable distribution

    @pytest.mark.parametrize("n_unique", [2, 10, 100, 1000])
    def test_high_cardinality_performance(self, n_unique: int):
        """Test performance with high cardinality categorical variables.
        
        Requirements addressed:
        - 16.4: Handle variables with many unique values efficiently
        """
        n_rows = 50_000
        
        # Create high cardinality data
        df = pd.DataFrame({
            'high_card': np.random.choice([f'val_{i}' for i in range(n_unique)], n_rows),
            'low_card': np.random.choice(['A', 'B', 'C'], n_rows)
        })
        
        vb = VariableBuilder.from_data(df)
        high_card, low_card = vb.get_variables('high_card', 'low_card')
        
        memory_monitor = MemoryMonitor()
        start_time = time.time()
        
        # Calculate distributions
        high_dist = p(high_card)
        joint_dist = p(high_card, low_card)
        
        # Force evaluation
        high_dict = high_dist.to_dict()
        joint_dict = joint_dist.to_dict()
        
        execution_time = time.time() - start_time
        memory_increase = memory_monitor.get_memory_increase()
        
        # Performance should scale reasonably with cardinality
        max_time = 10.0 + (n_unique / 100)  # Allow more time for higher cardinality
        assert execution_time < max_time, f"High cardinality test took {execution_time:.3f}s for {n_unique} unique values"
        
        # Memory should be reasonable
        max_memory = 100 + (n_unique / 10)  # Allow more memory for higher cardinality
        assert memory_increase < max_memory, f"Memory increased by {memory_increase:.1f}MB for {n_unique} unique values"
        
        # Verify results
        assert len(high_dict) <= n_unique
        assert len(joint_dict) <= n_unique * 3  # 3 low_card values
        assert abs(sum(high_dict.values()) - 1.0) < 1e-10
        assert abs(sum(joint_dict.values()) - 1.0) < 1e-10


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])