# Requirements Document

## Introduction

Poffertjes is a Python library that provides a friendly, pythonic, and intuitive interface for running probabilistic queries on dataframes. The library aims to make probability calculations on data as simple as writing mathematical notation like `p(x)` or `p(x).given(y)`. It supports both Pandas and Polars dataframes through the Narwhals abstraction layer, handles variables of any dtype, and provides both scalar probability values and full probability distributions for marginal and conditional probabilities.

## Requirements

### Requirement 1: Dataframe Backend Agnosticism

**User Story:** As a data scientist, I want to use poffertjes with both Pandas and Polars dataframes, so that I can work with my preferred dataframe library without changing my probabilistic query code.

#### Acceptance Criteria

1. WHEN a user provides a Pandas dataframe THEN poffertjes SHALL process it correctly through the Narwhals backend
2. WHEN a user provides a Polars dataframe THEN poffertjes SHALL process it correctly through the Narwhals backend
3. WHEN poffertjes processes dataframes THEN it SHALL use Narwhals as the abstraction layer
4. IF a dataframe type is not supported by Narwhals THEN poffertjes SHALL raise a clear error message

### Requirement 2: Singleton P Interface

**User Story:** As a library user, I want to import and use a simple `p` instance for all probabilistic queries, so that my code is concise and readable like mathematical notation.

#### Acceptance Criteria

1. WHEN a user imports poffertjes THEN they SHALL be able to use `from poffertjes import p`
2. WHEN the P class is instantiated THEN it SHALL follow the singleton pattern
3. WHEN a user calls `p(...)` THEN the `__call__` method SHALL be invoked
4. WHEN multiple imports of `p` occur THEN they SHALL reference the same singleton instance

### Requirement 3: Variable Extraction from Dataframes

**User Story:** As a library user, I want to extract variables from dataframe columns using a VariableBuilder, so that I can reference them in probabilistic queries.

#### Acceptance Criteria

1. WHEN a user creates a VariableBuilder with `VariableBuilder.from_data(df)` THEN it SHALL store the dataframe reference
2. WHEN a user calls `get_variables()` without arguments THEN it SHALL return Variable objects for all columns
3. WHEN a user calls `get_variables('x', 'y')` with column names THEN it SHALL return Variable objects for those specific columns
4. WHEN a Variable is created THEN it SHALL store the column name
5. WHEN a Variable is printed THEN it SHALL display as `Variable(column_name)`
6. IF a requested column name does not exist in the dataframe THEN the system SHALL raise a clear error

### Requirement 4: Marginal Probability Queries

**User Story:** As a data analyst, I want to calculate marginal probabilities using `p(x)` syntax, so that I can understand the distribution of individual variables.

#### Acceptance Criteria

1. WHEN a user calls `p(x)` with a Variable THEN the system SHALL return the probability distribution of that variable
2. WHEN a user calls `p(x == value)` with a condition THEN the system SHALL return the scalar probability of that condition
3. WHEN a probability distribution is returned THEN it SHALL include all observed values and their probabilities
4. WHEN probabilities are calculated THEN they SHALL sum to 1.0 (within floating point precision)

### Requirement 5: Conditional Probability Queries

**User Story:** As a data analyst, I want to calculate conditional probabilities using `p(x).given(y)` syntax, so that I can understand relationships between variables.

#### Acceptance Criteria

1. WHEN a user calls `p(x).given(y)` THEN the system SHALL return the conditional probability distribution of x given y
2. WHEN a user calls `p(x == value1).given(y == value2)` THEN the system SHALL return the scalar conditional probability
3. WHEN a user calls `p(x).given(y == value)` THEN the system SHALL return the distribution of x conditioned on y equals value
4. WHEN a user calls `p(x).given(y, z)` with multiple conditioning variables THEN the system SHALL return P(X|Y,Z)
5. WHEN a user calls `p(x).given(y == value1, z == value2)` THEN the system SHALL return P(X|Y=value1 AND Z=value2)
6. WHEN calculating `p(x).given(y)` without conditions THEN the system SHALL return P(X=x_i|Y=y_j) for all combinations of x_i and y_j
7. WHEN the conditioning event has zero occurrences THEN the system SHALL raise a clear error or return NaN
8. WHEN conditional probabilities are calculated for a specific conditioning value THEN they SHALL sum to 1.0

### Requirement 6: Frequency-Based Probability Calculation

**User Story:** As a library user, I want probabilities calculated using empirical frequency counts from the data, so that the results accurately reflect the observed data distribution.

#### Acceptance Criteria

1. WHEN calculating any probability THEN the system SHALL use frequency counting as the estimation method
2. WHEN counting frequencies THEN the system SHALL use Narwhals operations for dataframe-agnostic counting
3. WHEN calculating P(X=x) THEN the system SHALL count rows where X equals x and divide by total rows
4. WHEN calculating P(X=x, Y=y) THEN the system SHALL count rows where both conditions hold and divide by total rows
5. WHEN calculating P(X=x|Y=y) THEN the system SHALL count rows where both hold, divided by rows where Y=y holds
6. WHEN handling comparison operators (`<`, `>`, `<=`, `>=`, `!=`) THEN the system SHALL count all rows satisfying the condition
7. WHEN multiple conditions are combined with AND THEN the system SHALL count rows satisfying all conditions
8. WHEN the dataframe has missing values (NaN/None) THEN the system SHALL exclude them from counts by default
9. WHEN a user queries a value that doesn't exist in the data THEN the system SHALL return probability 0.0
10. WHEN calculating probabilities THEN the system SHALL handle floating point precision appropriately

### Requirement 7: Efficient Computation with Narwhals

**User Story:** As a library user working with large datasets, I want poffertjes to use efficient dataframe operations, so that probability calculations are fast and memory-efficient.

#### Acceptance Criteria

1. WHEN performing dataframe operations THEN the system SHALL prefer Narwhals lazy operations over eager ones whenever possible
2. WHEN calculating probability distributions THEN the system SHALL use group_by operations followed by aggregations
3. WHEN computing P(X) THEN the system SHALL use `df.group_by('X').agg(nw.len())` pattern rather than iterative counting
4. WHEN computing P(X,Y) THEN the system SHALL use `df.group_by(['X', 'Y']).agg(nw.len())` pattern
5. WHEN computing conditional probabilities THEN the system SHALL use efficient filter + group_by operations
6. WHEN using Narwhals expressions THEN the system SHALL leverage expression API for aggregations
7. WHEN implementing conditions THEN the system SHALL use native Narwhals expression methods rather than re-implementing logic
8. WHEN evaluating ternary conditions like `a < x < b` THEN the system SHALL use `nw.col('x').is_between(a, b)` rather than combining separate comparisons
9. WHEN evaluating comparison operators THEN the system SHALL use Narwhals column expression methods (e.g., `nw.col('x') > value`)
10. WHEN filtering dataframes THEN the system SHALL use Narwhals filter expressions rather than manual row selection
11. WHEN possible THEN the system SHALL avoid materializing intermediate results unnecessarily
12. WHEN working with lazy frames THEN the system SHALL maintain laziness until final result computation
13. WHEN calculating multiple probabilities THEN the system SHALL reuse computed aggregations when possible
14. WHEN operations can be vectorized THEN the system SHALL prefer vectorized operations over row-by-row processing

### Requirement 8: Multi-dtype Variable Support

**User Story:** As a library user, I want to perform probabilistic queries on variables of any dtype (int, float, categorical, boolean), so that I can analyze all types of data in my dataframes.

#### Acceptance Criteria

1. WHEN a Variable has integer dtype THEN probabilistic queries SHALL work correctly
2. WHEN a Variable has float dtype THEN probabilistic queries SHALL work correctly
3. WHEN a Variable has categorical/string dtype THEN probabilistic queries SHALL work correctly
4. WHEN a Variable has boolean dtype THEN probabilistic queries SHALL work correctly
5. WHEN a Variable has datetime dtype THEN probabilistic queries SHALL work correctly
6. WHEN comparing values in conditions THEN the system SHALL handle dtype-appropriate comparisons
7. WHEN a Variable has float dtype THEN the system SHALL treat each unique float value as a distinct outcome
8. WHEN working with continuous variables THEN the user SHALL be responsible for binning if needed

### Requirement 9: Probability Distribution Output

**User Story:** As a data scientist, I want to retrieve full probability distributions (not just scalar values), so that I can analyze and visualize the complete distribution of variables.

#### Acceptance Criteria

1. WHEN a user calls `p(x)` without conditions THEN the system SHALL return a probability distribution object
2. WHEN a probability distribution is returned THEN it SHALL be iterable over (value, probability) pairs
3. WHEN a probability distribution is returned THEN it SHALL support conversion to common formats (dict, dataframe)
4. WHEN a probability distribution is printed THEN it SHALL display in a human-readable format
5. WHEN a conditional distribution `p(x).given(y)` is requested THEN it SHALL return distributions for each value of y

### Requirement 10: Scalar Probability Output

**User Story:** As a library user, I want to calculate specific probability values using conditions like `p(x == 0)`, so that I can answer precise probabilistic questions.

#### Acceptance Criteria

1. WHEN a user provides a condition with `==` operator THEN the system SHALL return a scalar probability
2. WHEN a user provides a condition with `!=` operator THEN the system SHALL return a scalar probability
3. WHEN a user provides a condition with comparison operators (`<`, `>`, `<=`, `>=`) THEN the system SHALL return a scalar probability
4. WHEN a user provides a ternary condition like `p(a < x < b)` THEN the system SHALL return P(a < X < b)
5. WHEN multiple conditions are combined THEN the system SHALL evaluate them correctly
6. WHEN a scalar probability is returned THEN it SHALL be a float between 0.0 and 1.0

### Requirement 11: Joint and Multiple Variable Queries

**User Story:** As a data scientist, I want to calculate joint probabilities and query multiple variables simultaneously, so that I can analyze multivariate relationships.

#### Acceptance Criteria

1. WHEN a user calls `p(x, y)` with multiple variables THEN the system SHALL return the joint probability distribution
2. WHEN a user calls `p(x == value1, y == value2)` THEN the system SHALL return P(X=value1 AND Y=value2)
3. WHEN a joint distribution is returned THEN it SHALL include all observed combinations of values
4. WHEN calculating `p(x, y).given(z)` THEN the system SHALL support conditional joint distributions
5. WHEN joint probabilities are calculated THEN they SHALL sum to 1.0 across all combinations

### Requirement 12: Comprehensive Test Suite

**User Story:** As a library maintainer, I want a comprehensive test suite with unit tests and property-based tests, so that I can ensure correctness across all combinations of features.

#### Acceptance Criteria

1. WHEN tests are run THEN there SHALL be pytest unit tests for all core functionality
2. WHEN tests are run THEN there SHALL be hypothesis property-based tests for probabilistic properties
3. WHEN property-based tests run THEN they SHALL verify probability axioms (non-negativity, sum to 1)
4. WHEN property-based tests run THEN they SHALL verify the chain rule: P(X,Y) = P(X|Y) * P(Y)
5. WHEN property-based tests run THEN they SHALL verify Bayes' theorem: P(X|Y) = P(Y|X) * P(X) / P(Y)
6. WHEN property-based tests run THEN they SHALL verify the law of total probability
7. WHEN property-based tests run THEN they SHALL verify the union probability formula: P(X or Y) = P(X) + P(Y) - P(X and Y)
8. WHEN property-based tests run THEN they SHALL verify marginalization: sum over Y of P(X,Y) = P(X)
9. WHEN tests cover scenarios THEN they SHALL include all dtype combinations
10. WHEN tests cover scenarios THEN they SHALL include both marginal and conditional probabilities
11. WHEN tests cover scenarios THEN they SHALL include both distribution and scalar outputs
12. WHEN tests cover scenarios THEN they SHALL include multiple conditioning variables
13. WHEN tests cover scenarios THEN they SHALL include ternary conditions
14. WHEN tests cover edge cases THEN they SHALL include empty dataframes, single-value columns, and missing data
15. WHEN all tests pass THEN the library SHALL be considered ready for release

### Requirement 13: Data Binding and Context Management

**User Story:** As a library user, I want to bind dataframes to the probability calculator and manage context, so that I can work with different datasets without confusion.

#### Acceptance Criteria

1. WHEN a user creates variables with `VariableBuilder.from_data(df)` THEN the variables SHALL be bound to that specific dataframe
2. WHEN a user calls `p(x)` THEN the system SHALL know which dataframe x belongs to
3. WHEN variables from different dataframes are mixed THEN the system SHALL raise a clear error
4. WHEN a user wants to switch datasets THEN they SHALL create new variables from the new dataframe
5. WHEN the P singleton is used THEN it SHALL maintain context about which dataframe is being queried
6. WHEN a probability query is made THEN the system SHALL validate that all variables come from the same dataframe
7. WHEN variables are created THEN they SHALL maintain a reference to their source dataframe

### Requirement 14: API Usability and Pythonic Interface

**User Story:** As a Python developer, I want the library to follow Python conventions and provide a natural API, so that it feels intuitive and integrates well with my workflow.

#### Acceptance Criteria

1. WHEN using the library THEN it SHALL follow PEP 8 style guidelines
2. WHEN using the library THEN it SHALL provide type hints for all public APIs
3. WHEN a user inspects objects THEN they SHALL have helpful `__repr__` and `__str__` methods
4. WHEN a probability distribution is returned THEN it SHALL be iterable and support common Python protocols
5. WHEN errors occur THEN they SHALL use appropriate Python exception types
6. WHEN the library is imported THEN it SHALL not perform expensive operations at import time
7. WHEN documentation is needed THEN all public APIs SHALL have docstrings
8. WHEN a user wants IDE support THEN type hints SHALL enable autocomplete and type checking

### Requirement 15: Result Representation and Export

**User Story:** As a data analyst, I want to easily view, export, and work with probability results, so that I can integrate them into my analysis workflow.

#### Acceptance Criteria

1. WHEN a probability distribution is computed THEN it SHALL be convertible to a dictionary
2. WHEN a probability distribution is computed THEN it SHALL be convertible to a Pandas/Polars dataframe
3. WHEN a probability distribution is printed THEN it SHALL display in a readable tabular format
4. WHEN a distribution has many values THEN the display SHALL be truncated with indication of total count
5. WHEN a conditional distribution is computed THEN it SHALL clearly show the conditioning values
6. WHEN results are exported to dataframe THEN column names SHALL be descriptive (e.g., 'value', 'probability')
7. WHEN a scalar probability is returned THEN it SHALL be a standard Python float
8. WHEN distributions are compared THEN they SHALL support equality comparison

### Requirement 16: Edge Cases and Numerical Stability

**User Story:** As a library user, I want the library to handle edge cases gracefully and maintain numerical stability, so that I can trust the results even with unusual data.

#### Acceptance Criteria

1. WHEN a dataframe has only one unique value for a variable THEN P(X=value) SHALL equal 1.0
2. WHEN a dataframe is very large THEN probability calculations SHALL not overflow or underflow
3. WHEN probabilities are very small THEN they SHALL be represented accurately within float precision
4. WHEN a variable has many unique values THEN the system SHALL handle it efficiently
5. WHEN computing probabilities that should sum to 1.0 THEN floating point errors SHALL be within acceptable tolerance (e.g., 1e-10)
6. WHEN a dataframe has duplicate rows THEN they SHALL be counted appropriately
7. WHEN a condition matches zero rows THEN the probability SHALL be exactly 0.0
8. WHEN all rows match a condition THEN the probability SHALL be exactly 1.0
9. WHEN working with very small datasets (e.g., 1-10 rows) THEN calculations SHALL still be correct
10. WHEN variables have null/NaN values THEN they SHALL be handled consistently across all operations

### Requirement 17: Error Handling and User Feedback

**User Story:** As a library user, I want clear error messages when I make mistakes, so that I can quickly understand and fix issues in my code.

#### Acceptance Criteria

1. WHEN a user references a non-existent column THEN the system SHALL raise a descriptive error
2. WHEN a conditioning event has zero probability THEN the system SHALL handle it gracefully with a clear message
3. WHEN invalid syntax is used THEN the system SHALL provide helpful error messages
4. WHEN dataframe is empty THEN the system SHALL raise a clear error
5. WHEN incompatible operations are attempted THEN the system SHALL explain what went wrong
6. WHEN variables from different dataframes are mixed THEN the error SHALL indicate which variables are incompatible
7. WHEN a Variable is used without being properly initialized THEN the error SHALL guide the user to use VariableBuilder
