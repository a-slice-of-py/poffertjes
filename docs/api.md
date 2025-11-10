# Poffertjes API Documentation

Poffertjes provides a friendly, pythonic interface for running probabilistic queries on dataframes. This document covers all public APIs and their usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Main Interface](#main-interface)
4. [Variables](#variables)
5. [Expressions](#expressions)
6. [Results](#results)
7. [Exceptions](#exceptions)
8. [Examples](#examples)

## Quick Start

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 25, 35, 30, 25],
    'income': ['low', 'high', 'low', 'high', 'medium', 'low'],
    'purchased': [True, True, False, True, False, True]
})

# Extract variables
vb = VariableBuilder.from_data(df)
age, income, purchased = vb.get_variables('age', 'income', 'purchased')

# Calculate probabilities
print(p(age))  # Marginal distribution of age
print(p(purchased == True))  # P(purchased = True)
print(p(age).given(income == 'high'))  # P(age | income = 'high')
print(p(age, income))  # Joint distribution of age and income
```

## Core Components

### Importing Poffertjes

```python
from poffertjes import p  # Main probability interface
from poffertjes.variable import VariableBuilder  # For creating variables
```

### Supported Dataframes

Poffertjes works with both Pandas and Polars dataframes through the Narwhals abstraction:

```python
import pandas as pd
import polars as pl

# Works with Pandas
df_pandas = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
vb_pandas = VariableBuilder.from_data(df_pandas)

# Works with Polars
df_polars = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
vb_polars = VariableBuilder.from_data(df_polars)
```

## Main Interface

### `p` - The Probability Interface

The `p` object is the main entry point for all probability queries. It's a singleton that provides a mathematical notation for probability calculations.

#### Marginal Probabilities

```python
# Distribution queries (return DistributionResult)
p(x)        # P(X) - marginal distribution of x
p(x, y)     # P(X,Y) - joint distribution of x and y
p(x, y, z)  # P(X,Y,Z) - joint distribution of x, y, and z

# Scalar queries (return ScalarResult)
p(x == 5)           # P(X = 5)
p(x > 10)           # P(X > 10)
p(x == 1, y == 2)   # P(X = 1, Y = 2)
```

#### Conditional Probabilities

```python
# Conditional distributions
p(x).given(y == 2)        # P(X | Y = 2)
p(x).given(y)             # P(X | Y) - distribution for each value of Y
p(x, y).given(z == 3)     # P(X,Y | Z = 3)

# Conditional scalars
p(x == 1).given(y == 2)   # P(X = 1 | Y = 2)
p(x > 5).given(y == 'A')  # P(X > 5 | Y = 'A')

# Multiple conditions
p(x).given(y == 1, z == 2)  # P(X | Y = 1, Z = 2)
```

## Variables

### `VariableBuilder`

Factory class for creating Variable objects from dataframes.

#### `VariableBuilder.from_data(data)`

Create a VariableBuilder from a dataframe.

**Parameters:**
- `data`: Pandas or Polars dataframe

**Returns:**
- `VariableBuilder` instance

**Raises:**
- `DataframeError`: If dataframe is empty

```python
vb = VariableBuilder.from_data(df)
```

#### `get_variables(*args)`

Extract variables from dataframe columns.

**Parameters:**
- `*args`: Column names. If empty, returns all columns.

**Returns:**
- Single `Variable` if one column requested
- `List[Variable]` if multiple columns requested

**Raises:**
- `VariableError`: If column doesn't exist

```python
# Get specific variables
x, y = vb.get_variables('x', 'y')

# Get single variable
x = vb.get_variables('x')

# Get all variables
all_vars = vb.get_variables()
```

### `Variable`

Represents a random variable bound to a dataframe column.

#### Properties

- `name`: Column name
- `dataframe_id`: Unique identifier for the source dataframe

#### Comparison Operators

Variables support all comparison operators to create expressions:

```python
x == 5      # Equality
x != 5      # Inequality
x < 5       # Less than
x <= 5      # Less than or equal
x > 5       # Greater than
x >= 5      # Greater than or equal
```

#### `isin(values)`

Create membership expression for categorical variables.

**Parameters:**
- `values`: List of values to check membership against

**Returns:**
- `Expression` object

```python
x.isin([1, 2, 3])           # x in {1, 2, 3}
category.isin(['A', 'B'])   # category in {'A', 'B'}
```

## Expressions

### `Expression`

Represents a condition on a variable (e.g., x == 5, x > 10).

#### Combining Expressions

```python
# AND logic
(x > 5) & (x < 10)      # 5 < x < 10
(x == 1) & (y == 2)     # x = 1 AND y = 2

# OR logic
(x == 1) | (x == 2)     # x = 1 OR x = 2
(x < 5) | (x > 10)      # x < 5 OR x > 10
```

### `TernaryExpression`

Specialized expression for range conditions.

```python
from poffertjes.expression import TernaryExpression

# Create range condition: 3 < x < 7
expr = TernaryExpression(x, 3, 7, closed="none")

# Different boundary conditions
TernaryExpression(x, 3, 7, closed="left")   # 3 <= x < 7
TernaryExpression(x, 3, 7, closed="right")  # 3 < x <= 7
TernaryExpression(x, 3, 7, closed="both")   # 3 <= x <= 7
```

### `CompositeExpression`

Represents multiple expressions combined with AND/OR logic.

```python
# Created automatically when combining expressions
complex_expr = (x > 5) & (x < 10) & (y == 'A')
```

## Results

### `ScalarResult`

Represents a scalar probability value.

#### Methods

- `__float__()`: Convert to float
- `given(*conditions)`: Apply conditional probability

```python
result = p(x == 5)
print(float(result))  # 0.333333
print(result)         # 0.333333

# Conditional probability
conditional = result.given(y == 2)
```

### `DistributionResult`

Represents a probability distribution.

#### Methods

- `given(*conditions)`: Apply conditional probability
- `to_dict()`: Convert to dictionary
- `to_dataframe()`: Convert to native dataframe

```python
dist = p(x)

# Conditional distribution
conditional_dist = dist.given(y == 2)

# Export formats
dict_format = dist.to_dict()        # {1: 0.4, 2: 0.6}
df_format = dist.to_dataframe()     # Native dataframe
```

### `Distribution`

Low-level distribution object with iteration support.

#### Methods

- `__iter__()`: Iterate over (value, probability) pairs
- `to_dict()`: Convert to dictionary
- `to_dataframe()`: Convert to native dataframe
- `__eq__(other)`: Compare distributions

```python
dist = p(x).distribution

# Iterate over values
for value, prob in dist:
    print(f"P(X = {value}) = {prob}")

# Convert formats
dict_data = dist.to_dict()
df_data = dist.to_dataframe()
```

## Exceptions

### Exception Hierarchy

```
PoffertjesError
├── DataframeError
├── VariableError
├── ExpressionError
└── ProbabilityError
```

### `PoffertjesError`

Base exception for all poffertjes errors.

```python
try:
    result = p(x, y)  # Variables from different dataframes
except PoffertjesError as e:
    print(f"Poffertjes error: {e}")
```

### `DataframeError`

Errors related to dataframe operations.

**Common causes:**
- Empty dataframes
- Mixed dataframes from different sources
- Dataframe conversion issues

### `VariableError`

Errors related to variable operations.

**Common causes:**
- Non-existent column names
- Invalid variable operations
- Improper conditioning syntax

### `ExpressionError`

Errors related to expression operations.

**Common causes:**
- Invalid operators
- Empty lists for `isin()`
- Malformed expressions

### `ProbabilityError`

Errors related to probability calculations.

**Common causes:**
- Zero probability conditioning events
- Invalid probability computations
- Insufficient variables for joint calculations

## Examples

### Basic Probability Calculations

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Sample data
df = pd.DataFrame({
    'coin': ['H', 'T', 'H', 'T', 'H', 'T'],
    'die': [1, 2, 3, 4, 5, 6],
    'sum_val': [2, 3, 4, 5, 6, 7]
})

vb = VariableBuilder.from_data(df)
coin, die, sum_val = vb.get_variables('coin', 'die', 'sum_val')

# Marginal probabilities
print(p(coin))              # Distribution of coin flips
print(p(coin == 'H'))       # P(coin = 'H') = 0.5

# Joint probabilities
print(p(coin, die))         # Joint distribution
print(p(coin == 'H', die == 1))  # P(coin='H', die=1)

# Conditional probabilities
print(p(sum_val).given(coin == 'H'))  # P(sum | coin='H')
print(p(die > 3).given(coin == 'T'))  # P(die > 3 | coin='T')
```

### Working with Different Data Types

```python
# Numeric data
df_numeric = pd.DataFrame({
    'age': [25, 30, 35, 25, 30],
    'score': [85.5, 92.0, 78.5, 88.0, 95.5]
})

vb = VariableBuilder.from_data(df_numeric)
age, score = vb.get_variables('age', 'score')

print(p(age == 25))         # Integer comparison
print(p(score > 90.0))      # Float comparison
print(p(age).given(score > 85))  # Mixed conditions

# Categorical data
df_categorical = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B'],
    'status': ['active', 'inactive', 'active', 'pending', 'active']
})

vb = VariableBuilder.from_data(df_categorical)
category, status = vb.get_variables('category', 'status')

print(p(category.isin(['A', 'B'])))  # Membership test
print(p(status == 'active'))         # String comparison

# Boolean data
df_boolean = pd.DataFrame({
    'is_member': [True, False, True, True, False],
    'has_discount': [False, True, True, False, True]
})

vb = VariableBuilder.from_data(df_boolean)
is_member, has_discount = vb.get_variables('is_member', 'has_discount')

print(p(is_member == True))          # Boolean comparison
print(p(is_member).given(has_discount == True))  # Boolean conditioning
```

### Complex Expressions

```python
# Multiple conditions with AND/OR
complex_condition = (age > 25) & (age < 35) & (score > 80)
print(p(complex_condition))

# Range conditions
from poffertjes.expression import TernaryExpression
age_range = TernaryExpression(age, 25, 35, closed="both")  # 25 <= age <= 35
print(p(age_range))

# Multiple conditioning variables
print(p(score).given(age > 25, category == 'A'))
```

### Error Handling

```python
try:
    # This will raise VariableError - column doesn't exist
    bad_var = vb.get_variables('nonexistent_column')
except VariableError as e:
    print(f"Variable error: {e}")

try:
    # This will raise DataframeError - variables from different dataframes
    vb1 = VariableBuilder.from_data(df1)
    vb2 = VariableBuilder.from_data(df2)
    x = vb1.get_variables('x')
    y = vb2.get_variables('y')
    result = p(x, y)
except DataframeError as e:
    print(f"Dataframe error: {e}")

try:
    # This will raise ProbabilityError - zero probability conditioning
    result = p(x).given(y == 'impossible_value')
except ProbabilityError as e:
    print(f"Probability error: {e}")
```

### Performance Tips

```python
# Use lazy evaluation with Polars for large datasets
import polars as pl

large_df = pl.DataFrame({
    'x': range(1000000),
    'y': ['A', 'B'] * 500000
})

vb = VariableBuilder.from_data(large_df)
x, y = vb.get_variables('x', 'y')

# This uses lazy evaluation internally
result = p(x > 500000).given(y == 'A')

# Convert to native format only when needed
native_result = result.to_dataframe()
```

## Best Practices

1. **Variable Management**: Always create variables from the same dataframe for a single analysis.

2. **Error Handling**: Use specific exception types to handle different error conditions appropriately.

3. **Performance**: For large datasets, prefer Polars dataframes for better performance with lazy evaluation.

4. **Type Safety**: Use type hints and IDE support for better development experience.

5. **Readability**: Use descriptive variable names and leverage the mathematical notation for clear probability expressions.

```python
# Good: Clear and readable
age_dist = p(age)
high_income_prob = p(income == 'high')
age_given_high_income = p(age).given(income == 'high')

# Less clear: Generic names
result1 = p(x)
result2 = p(y == 1)
result3 = p(x).given(y == 1)
```