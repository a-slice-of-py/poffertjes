# Implementation Plan

## Overview

This implementation plan breaks down the probabilistic query interface into incremental, testable steps. Each task builds on previous ones and includes specific requirements references. The plan follows test-driven development principles where appropriate.

## Tasks

- [x] 1. Set up project dependencies and structure

  - Add narwhals to pyproject.toml dependencies
  - Add pytest and hypothesis to dev dependencies
  - Create src/poffertjes module structure
  - Create tests/ directory structure
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Implement Variable and VariableBuilder classes
  - [x] 2.1 Create Variable class with basic structure

    - Implement `__init__(name, nw_frame, frame_id)`
    - Implement `__repr__` and `__str__` methods
    - Implement `dataframe_id` property
    - Write unit tests for Variable creation and representation
    - _Requirements: 3.1, 3.4, 3.5, 14.3_
  
  - [x] 2.2 Create VariableBuilder class

    - Implement `__init__(data)` with Narwhals conversion
    - Implement `get_variables(*args)` method
    - Implement `from_data(data)` static method
    - Handle column validation and error cases
    - Write unit tests for VariableBuilder
    - _Requirements: 3.1, 3.2, 3.3, 3.6, 17.1_
  
  - [x] 2.3 Add operator overloading to Variable

    - Implement `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` methods
    - Implement `isin()` method for categorical variables
    - Write unit tests for operator overloading
    - _Requirements: 9.1, 9.2, 9.3, 14.3_

- [ ] 3. Implement Expression system
  - [x] 3.1 Create ExpressionOp enum

    - Define all operators (EQ, NE, LT, LE, GT, GE, BETWEEN, IN)
    - Write unit tests for enum
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 3.2 Create Expression class

    - Implement `__init__(variable, operator, value, upper_bound)`
    - Implement `to_narwhals_expr()` method using native Narwhals expressions
    - Implement `__and__` and `__or__` for combining expressions
    - Write unit tests for Expression creation and conversion
    - _Requirements: 7.7, 7.8, 7.9, 9.1, 9.2, 9.3_
  
  - [x] 3.3 Create CompositeExpression class

    - Implement `__init__(expressions, logic)`
    - Implement `to_narwhals_expr()` for AND/OR logic
    - Write unit tests for composite expressions
    - _Requirements: 9.6_
  
  - [x] 3.4 Create TernaryExpression class

    - Implement ternary condition support using `is_between`
    - Write unit tests for ternary expressions
    - _Requirements: 7.8, 9.4_

- [ ] 4. Implement P singleton interface
  - [x] 4.1 Create P class with singleton pattern

    - Implement `__new__` for singleton behavior
    - Implement `__call__` method
    - Write unit tests for singleton behavior
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 4.2 Implement variable extraction and validation

    - Implement `_extract_variables()` method
    - Implement `_validate_same_dataframe()` method
    - Write unit tests for validation
    - _Requirements: 13.2, 13.3, 13.6, 17.6_
  
  - [x] 4.3 Create and export p singleton instance

    - Create p instance in __init__.py
    - Write integration tests for `from poffertjes import p`
    - _Requirements: 2.1, 14.6_

- [ ] 5. Implement QueryBuilder
  - [x] 5.1 Create QueryBuilder class

    - Implement `__init__(args)` and `_parse_args()`
    - Parse arguments into variables and expressions
    - Write unit tests for argument parsing
    - _Requirements: 4.1, 4.2, 9.1_
  
  - [x] 5.2 Implement query execution logic

    - Implement `execute()` method
    - Determine scalar vs distribution queries
    - Write unit tests for query execution routing
    - _Requirements: 4.1, 4.2, 9.1_

- [ ] 6. Implement ProbabilityCalculator
  - [x] 6.1 Create ProbabilityCalculator class structure

    - Implement `__init__(dataframe)` with total count
    - Write basic unit tests
    - _Requirements: 6.1, 6.2_
  
  - [x] 6.2 Implement marginal probability distribution calculation

    - Implement `calculate_distribution(variables, conditions=None)`
    - Use `group_by().agg(nw.len())` pattern
    - Calculate probabilities by dividing by total
    - Write unit tests for marginal distributions
    - _Requirements: 4.1, 4.3, 4.4, 6.3, 6.4, 7.2, 7.3_
  
  - [x] 6.3 Implement scalar probability calculation

    - Implement `calculate_scalar(expressions, conditions=None)`
    - Use filter operations with Narwhals expressions
    - Write unit tests for scalar probabilities
    - _Requirements: 4.2, 4.4, 6.3, 6.5, 7.9, 7.10, 9.1, 9.2, 9.3_
  
  - [x] 6.4 Implement conditional probability support

    - Add conditioning logic to both methods
    - Handle zero probability conditioning events
    - Write unit tests for conditional probabilities
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.5, 7.5_
  
  - [x] 6.5 Implement joint probability calculation

    - Implement `calculate_joint(variables, conditions=None)`
    - Use multi-column group_by
    - Write unit tests for joint probabilities
    - _Requirements: 6.4, 7.4, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 7. Implement result objects

  - [x] 7.1 Create QueryResult base class

    - Define abstract `given()` method
    - Write basic structure
    - _Requirements: 5.1_
  
  - [x] 7.2 Create ScalarResult class

    - Implement `__init__(value, expressions, dataframe)`
    - Implement `__float__` and `__repr__`
    - Implement `given()` method for conditional probabilities
    - Write unit tests for ScalarResult
    - _Requirements: 4.2, 5.2, 9.6, 15.7_
  
  - [x] 7.3 Create DistributionResult class

    - Implement `__init__(distribution, variables, dataframe, conditions)`
    - Implement `given()` method for conditional distributions
    - Write unit tests for DistributionResult
    - _Requirements: 4.1, 5.1, 5.3, 5.4, 5.5, 5.6, 9.1_
  
  - [x] 7.4 Implement Distribution class

    - Implement `__init__(data, variables)`
    - Implement `__iter__` for (value, probability) pairs
    - Implement `__repr__` with readable formatting
    - Implement `to_dict()` and `to_dataframe()` methods
    - Implement `__eq__` for comparison
    - Write unit tests for Distribution
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.8_

- [x] 8. Implement error handling

  - [x] 8.1 Create exception hierarchy

    - Create PoffertjesError base class
    - Create DataframeError, VariableError, ExpressionError, ProbabilityError
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_
  
  - [x] 8.2 Add error handling throughout codebase

    - Add descriptive error messages
    - Handle edge cases (empty dataframes, missing columns, zero probability)
    - Write unit tests for error cases
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7_

- [x] 9. Add multi-dtype support

  - [x] 9.1 Test with integer columns

    - Write integration tests with int dtype
    - _Requirements: 8.1_
  
  - [x] 9.2 Test with float columns

    - Write integration tests with float dtype
    - Handle floating point precision
    - _Requirements: 8.2, 8.7, 16.3_
  
  - [x] 9.3 Test with categorical/string columns

    - Write integration tests with string dtype
    - Test `isin()` method
    - _Requirements: 8.3_
  
  - [x] 9.4 Test with boolean columns

    - Write integration tests with bool dtype
    - _Requirements: 8.4_
  
  - [x] 9.5 Test with datetime columns

    - Write integration tests with datetime dtype
    - _Requirements: 8.5_
  
  - [x] 9.6 Test dtype-appropriate comparisons

    - Write tests for all comparison operators across dtypes
    - _Requirements: 8.6_

- [x] 10. Add comprehensive integration tests

  - [x] 10.1 Test end-to-end marginal probability workflows

    - Test `p(x)` returning distributions
    - Test `p(x == value)` returning scalars
    - Test with both Pandas and Polars
    - _Requirements: 1.1, 1.2, 4.1, 4.2, 4.3, 4.4_
  
  - [x] 10.2 Test end-to-end conditional probability workflows

    - Test `p(x).given(y)` and `p(x).given(y == value)`
    - Test multiple conditioning variables
    - Test with both Pandas and Polars
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  
  - [x] 10.3 Test end-to-end joint probability workflows

    - Test `p(x, y)` and `p(x == v1, y == v2)`
    - Test conditional joint distributions
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [x] 10.4 Test ternary conditions

    - Test `p(a < x < b)` syntax
    - Test with `given()` method
    - _Requirements: 9.4_
  
  - [x] 10.5 Test result export functionality

    - Test `.to_dict()` and `.to_dataframe()` methods
    - Verify native format conversion
    - _Requirements: 15.1, 15.2, 15.6_

- [x] 11. Implement property-based tests

  - [x] 11.1 Create Hypothesis strategies for test data

    - Create dataframe generation strategy
    - Support multiple dtypes
    - _Requirements: 12.1, 12.9_
  
  - [x] 11.2 Test probability axioms

    - Test non-negativity: all probabilities >= 0
    - Test normalization: probabilities sum to 1.0
    - _Requirements: 12.3, 16.5_
  
  - [x] 11.3 Test chain rule

    - Verify P(X,Y) = P(X|Y) * P(Y)
    - _Requirements: 12.4_
  
  - [x] 11.4 Test Bayes' theorem

    - Verify `P(X|Y) * P(Y) = P(Y|X) * P(X)`
    - _Requirements: 12.5_
  
  - [x] 11.5 Test law of total probability

    - Verify sum over Y of P(X|Y) * P(Y) = P(X)
    - _Requirements: 12.6_
  
  - [x] 11.6 Test union probability formula

    - Verify P(X or Y) = P(X) + P(Y) - P(X and Y)
    - _Requirements: 12.7_
  
  - [x] 11.7 Test marginalization

    - Verify sum over Y of P(X,Y) = P(X)
    - _Requirements: 12.8_
  
  - [x] 11.8 Test edge cases

    - Test empty dataframes, single-value columns, missing data
    - Test very small and very large datasets
    - _Requirements: 12.14, 16.1, 16.2, 16.4, 16.6, 16.7, 16.8, 16.9, 16.10_

- [x] 12. Add type hints and documentation

  - [x] 12.1 Add type hints to all public APIs

    - Use proper type annotations
    - Support IDE autocomplete
    - _Requirements: 14.2, 14.8_
  
  - [x] 12.2 Add docstrings to all public APIs

    - Follow Google or NumPy docstring style
    - Include examples
    - _Requirements: 14.7_
  
  - [x] 12.3 Create API documentation

    - Document main usage patterns
    - Include examples for all features
    - _Requirements: 14.7_

- [ ] 13. Performance optimization


  - [x] 13.1 Ensure lazy evaluation is used

    - Verify Narwhals lazy operations are preferred
    - Avoid unnecessary `.collect()` calls
    - _Requirements: 7.1, 7.11, 7.12_
  
  - [x] 13.2 Optimize group_by operations

    - Ensure efficient aggregation patterns
    - Reuse computations where possible
    - _Requirements: 7.2, 7.3, 7.4, 7.13_
  
  - [x] 13.3 Add performance tests

    - Benchmark with large datasets
    - Verify memory efficiency
    - _Requirements: 7.14, 16.2, 16.4_

- [ ] 14. Final integration and polish
  - [ ] 14.1 Verify PEP 8 compliance
    - Run linter (ruff)
    - Fix any style issues
    - _Requirements: 14.1_
  
  - [ ] 14.2 Run full test suite
    - Ensure all tests pass
    - Check test coverage
    - _Requirements: 12.15_
  
  - [ ] 14.3 Update README with examples
    - Add installation instructions
    - Add usage examples
    - Add API reference
    - _Requirements: 14.7_
  
  - [ ] 14.4 Verify both Pandas and Polars support
    - Test with both backends
    - Ensure feature parity
    - _Requirements: 1.1, 1.2, 1.3_
