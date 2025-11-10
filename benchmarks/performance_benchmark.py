#!/usr/bin/env python3
"""Performance benchmark script for poffertjes.

This script provides comprehensive benchmarks for the poffertjes library,
measuring performance across different dataset sizes, operations, and backends.

Usage:
    python benchmarks/performance_benchmark.py
    python benchmarks/performance_benchmark.py --quick  # Run quick benchmarks only
    python benchmarks/performance_benchmark.py --backend pandas  # Test only pandas
    python benchmarks/performance_benchmark.py --output results.json  # Save results

Requirements addressed:
- 7.14: Benchmark with large datasets
- 16.2: Verify memory efficiency  
- 16.4: Handle very large datasets efficiently
"""

import argparse
import json
import time
import gc
import psutil
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import polars as pl
import numpy as np

from poffertjes import p
from poffertjes.variable import VariableBuilder


class BenchmarkRunner:
    """Runs performance benchmarks for poffertjes."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results: Dict[str, Any] = {}
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def generate_dataset(self, n_rows: int, n_categories: int = 5) -> pd.DataFrame:
        """Generate a test dataset."""
        np.random.seed(42)  # Reproducible results
        
        return pd.DataFrame({
            'int_col': np.random.randint(0, 20, n_rows),
            'float_col': np.random.normal(0, 1, n_rows),
            'cat_col': np.random.choice([f'cat_{i}' for i in range(n_categories)], n_rows),
            'bool_col': np.random.choice([True, False], n_rows),
            'high_card': np.random.choice([f'val_{i}' for i in range(min(n_rows//10, 1000))], n_rows),
        })
    
    def benchmark_operation(self, name: str, operation_func, *args, **kwargs) -> Dict[str, float]:
        """Benchmark a single operation."""
        # Force garbage collection before benchmark
        gc.collect()
        
        # Measure initial memory
        initial_memory = self.get_memory_mb()
        
        # Run operation and measure time
        start_time = time.time()
        result = operation_func(*args, **kwargs)
        
        # Force evaluation if it's a lazy result
        if hasattr(result, 'to_dict'):
            result.to_dict()
        elif hasattr(result, '__float__'):
            float(result)
        
        end_time = time.time()
        
        # Measure final memory
        final_memory = self.get_memory_mb()
        
        execution_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        print(f"  {name}: {execution_time:.4f}s, {memory_increase:.1f}MB")
        
        return {
            'execution_time': execution_time,
            'memory_increase': memory_increase,
            'initial_memory': initial_memory,
            'final_memory': final_memory
        }
    
    def benchmark_dataset_size(self, backend: str, sizes: List[int]) -> Dict[str, Any]:
        """Benchmark performance across different dataset sizes."""
        print(f"\n=== Dataset Size Benchmarks ({backend}) ===")
        
        results = {}
        
        for size in sizes:
            print(f"\nTesting {size:,} rows:")
            
            # Generate data
            df_pandas = self.generate_dataset(size)
            if backend == 'polars':
                df = pl.from_pandas(df_pandas)
            else:
                df = df_pandas
            
            # Create variables
            vb = VariableBuilder.from_data(df)
            int_col, cat_col, float_col = vb.get_variables('int_col', 'cat_col', 'float_col')
            
            size_results = {}
            
            # Marginal distribution
            size_results['marginal'] = self.benchmark_operation(
                "Marginal P(X)",
                lambda: p(int_col)
            )
            
            # Joint distribution
            size_results['joint'] = self.benchmark_operation(
                "Joint P(X,Y)",
                lambda: p(int_col, cat_col)
            )
            
            # Conditional distribution
            size_results['conditional'] = self.benchmark_operation(
                "Conditional P(X|Y=val)",
                lambda: p(int_col).given(cat_col == 'cat_0')
            )
            
            # Scalar probability
            size_results['scalar'] = self.benchmark_operation(
                "Scalar P(X=val)",
                lambda: p(int_col == 5)
            )
            
            # Conditional scalar
            size_results['conditional_scalar'] = self.benchmark_operation(
                "Conditional Scalar P(X=val|Y=val)",
                lambda: p(int_col == 5).given(cat_col == 'cat_0')
            )
            
            results[f'{size}_rows'] = size_results
        
        return results
    
    def benchmark_cardinality(self, backend: str) -> Dict[str, Any]:
        """Benchmark performance with different cardinalities."""
        print(f"\n=== Cardinality Benchmarks ({backend}) ===")
        
        results = {}
        n_rows = 50_000
        cardinalities = [2, 10, 50, 200, 1000]
        
        for cardinality in cardinalities:
            print(f"\nTesting cardinality {cardinality}:")
            
            # Generate data with specific cardinality
            df_pandas = pd.DataFrame({
                'high_card': np.random.choice([f'val_{i}' for i in range(cardinality)], n_rows),
                'low_card': np.random.choice(['A', 'B', 'C'], n_rows)
            })
            
            if backend == 'polars':
                df = pl.from_pandas(df_pandas)
            else:
                df = df_pandas
            
            vb = VariableBuilder.from_data(df)
            high_card, low_card = vb.get_variables('high_card', 'low_card')
            
            card_results = {}
            
            # High cardinality marginal
            card_results['marginal'] = self.benchmark_operation(
                f"Marginal (card={cardinality})",
                lambda: p(high_card)
            )
            
            # High cardinality joint
            card_results['joint'] = self.benchmark_operation(
                f"Joint (card={cardinality})",
                lambda: p(high_card, low_card)
            )
            
            results[f'cardinality_{cardinality}'] = card_results
        
        return results
    
    def benchmark_caching(self, backend: str) -> Dict[str, Any]:
        """Benchmark caching performance."""
        print(f"\n=== Caching Benchmarks ({backend}) ===")
        
        # Use medium-sized dataset for visible caching effects
        df_pandas = self.generate_dataset(100_000)
        if backend == 'polars':
            df = pl.from_pandas(df_pandas)
        else:
            df = df_pandas
        
        vb = VariableBuilder.from_data(df)
        x, y = vb.get_variables('int_col', 'cat_col')
        
        results = {}
        
        # First calculation (cold cache)
        results['cold_cache'] = self.benchmark_operation(
            "Cold cache P(X,Y)",
            lambda: p(x, y)
        )
        
        # Second calculation (warm cache)
        results['warm_cache'] = self.benchmark_operation(
            "Warm cache P(X,Y)",
            lambda: p(x, y)
        )
        
        # Calculate improvement
        cold_time = results['cold_cache']['execution_time']
        warm_time = results['warm_cache']['execution_time']
        
        if cold_time > 0:
            improvement = (cold_time - warm_time) / cold_time * 100
            results['cache_improvement_percent'] = improvement
            print(f"  Cache improvement: {improvement:.1f}%")
        
        return results
    
    def run_benchmarks(self, backends: List[str], quick: bool = False) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("Starting poffertjes performance benchmarks...")
        print(f"Python version: {os.sys.version}")
        print(f"Memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        all_results = {
            'metadata': {
                'python_version': os.sys.version,
                'available_memory_gb': psutil.virtual_memory().available / 1024**3,
                'timestamp': time.time()
            }
        }
        
        for backend in backends:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING {backend.upper()}")
            print(f"{'='*60}")
            
            backend_results = {}
            
            # Dataset size benchmarks
            if quick:
                sizes = [1_000, 10_000]
            else:
                sizes = [1_000, 10_000, 100_000, 500_000]
            
            backend_results['dataset_sizes'] = self.benchmark_dataset_size(backend, sizes)
            
            # Cardinality benchmarks
            backend_results['cardinality'] = self.benchmark_cardinality(backend)
            
            # Caching benchmarks
            backend_results['caching'] = self.benchmark_caching(backend)
            
            all_results[backend] = backend_results
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        for backend in ['pandas', 'polars']:
            if backend not in results:
                continue
                
            print(f"\n{backend.upper()} Performance:")
            
            # Dataset size summary
            dataset_results = results[backend]['dataset_sizes']
            print("  Dataset Size Performance (marginal P(X)):")
            
            for size_key in sorted(dataset_results.keys()):
                if 'rows' in size_key:
                    size = size_key.replace('_rows', '')
                    time_ms = dataset_results[size_key]['marginal']['execution_time'] * 1000
                    memory_mb = dataset_results[size_key]['marginal']['memory_increase']
                    print(f"    {size:>7} rows: {time_ms:6.1f}ms, {memory_mb:5.1f}MB")
            
            # Caching summary
            caching_results = results[backend]['caching']
            if 'cache_improvement_percent' in caching_results:
                improvement = caching_results['cache_improvement_percent']
                print(f"  Caching improvement: {improvement:.1f}%")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Run poffertjes performance benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks only')
    parser.add_argument('--backend', choices=['pandas', 'polars'], help='Test specific backend only')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Determine backends to test
    if args.backend:
        backends = [args.backend]
    else:
        backends = ['pandas', 'polars']
    
    # Run benchmarks
    runner = BenchmarkRunner()
    results = runner.run_benchmarks(backends, quick=args.quick)
    
    # Print summary
    runner.print_summary(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()