# Poffertjes Quick Reference

## Installation & Import

```python
# Installation (when available)
pip install poffertjes

# Basic imports
from poffertjes import p
from poffertjes.variable import VariableBuilder
```

## Basic Setup

```python
import pandas as pd  # or polars as pl

# Create variables from dataframe
vb = VariableBuilder.from_data(df)
x, y, z = vb.get_variables('x', 'y', 'z')

# Or get all variables
all_vars = vb.get_variables()
```

## Probability Queries

### Marginal Probabilities

| Query | Description | Returns |
|-------|-------------|---------|
| `p(x)` | Distribution of x | DistributionResult |
| `p(x == 5)` | P(X = 5) | ScalarResult |
| `p(x > 10)` | P(X > 10) | ScalarResult |
| `p(x, y)` | Joint distribution of x,y | DistributionResult |
| `p(x == 1, y == 2)` | P(X=1, Y=2) | ScalarResult |

### Conditional Probabilities

| Query | Description | Returns |
|-------|-------------|---------|
| `p(x).given(y == 2)` | P(X \| Y=2) | DistributionResult |
| `p(x == 1).given(y == 2)` | P(X=1 \| Y=2) | ScalarResult |
| `p(x).given(y)` | P(X \| Y) for all Y | DistributionResult |
| `p(x).given(y == 1, z == 2)` | P(X \| Y=1, Z=2) | DistributionResult |

## Variable Operations

### Comparison Operators

```python
x == 5      # Equality
x != 5      # Inequality  
x < 5       # Less than
x <= 5      # Less than or equal
x > 5       # Greater than
x >= 5      # Greater than or equal
x.isin([1,2,3])  # Membership test
```

### Combining Expressions

```python
(x > 5) & (x < 10)    # AND: 5 < x < 10
(x == 1) | (x == 2)   # OR: x = 1 OR x = 2
```

### Ternary Expressions

```python
from poffertjes.expression import TernaryExpression

# Range conditions
TernaryExpression(x, 3, 7, closed="none")   # 3 < x < 7
TernaryExpression(x, 3, 7, closed="both")   # 3 ≤ x ≤ 7
TernaryExpression(x, 3, 7, closed="left")   # 3 ≤ x < 7
TernaryExpression(x, 3, 7, closed="right")  # 3 < x ≤ 7
```

## Result Operations

### ScalarResult

```python
result = p(x == 5)
float(result)           # Convert to float
result.given(y == 2)    # Conditional probability
```

### DistributionResult

```python
dist = p(x)
dist.given(y == 2)      # Conditional distribution
dist.to_dict()          # Convert to dictionary
dist.to_dataframe()     # Convert to native dataframe
```

### Distribution

```python
# Iterate over (value, probability) pairs
for value, prob in dist:
    print(f"P(X={value}) = {prob}")
```

## Common Patterns

### Basic Analysis

```python
# Marginal distribution
p(x)

# Specific probability
p(x == value)

# Conditional analysis
p(x).given(y == condition)

# Joint analysis
p(x, y)
```

### Complex Conditions

```python
# Multiple conditions
p(outcome == 'success').given(
    (age > 25) & (age < 65),
    income == 'high'
)

# Range conditions
p(sales > target).given(
    temperature.isin([20, 21, 22, 23, 24])
)
```

### Data Export

```python
# Get results as dictionary
result_dict = p(x).to_dict()

# Get results as dataframe
result_df = p(x).to_dataframe()

# Convert to native format
native_df = result_df.to_native()
```

## Error Handling

```python
from poffertjes import (
    PoffertjesError,
    DataframeError, 
    VariableError,
    ExpressionError,
    ProbabilityError
)

try:
    result = p(x, y)  # Variables from different dataframes
except DataframeError as e:
    print(f"Dataframe error: {e}")
except PoffertjesError as e:
    print(f"General poffertjes error: {e}")
```

## Performance Tips

### Use Polars for Large Data

```python
import polars as pl

# Better performance with large datasets
df = pl.DataFrame(large_data)
vb = VariableBuilder.from_data(df)
```

### Efficient Queries

```python
# Prefer specific conditions over broad ranges
p(category.isin(['A', 'B']))  # Better than multiple ORs

# Use appropriate data types
# Convert strings to categories if many repeated values
```

## Common Use Cases

### A/B Testing

```python
# Conversion rates
control_rate = p(converted == True).given(group == 'control')
treatment_rate = p(converted == True).given(group == 'treatment')

# Lift calculation
lift = (float(treatment_rate) / float(control_rate) - 1) * 100
```

### Customer Segmentation

```python
# Purchase probability by segment
purchase_by_age = p(purchased == True).given(age_group)

# Product preferences
product_pref = p(product_category).given(
    income_level == 'high',
    age_group == '25-34'
)
```

### Risk Analysis

```python
# Default probability
default_rate = p(default == True).given(
    credit_score < 600,
    debt_ratio > 0.4
)

# Risk factors
high_risk = p(risk_category == 'high').given(
    (income < 30000) & (age < 25)
)
```

### Medical Analysis

```python
# Diagnostic accuracy
sensitivity = p(test_positive == True).given(has_disease == True)
specificity = p(test_positive == False).given(has_disease == False)

# Predictive values
ppv = p(has_disease == True).given(test_positive == True)
npv = p(has_disease == False).given(test_positive == False)
```

## Cheat Sheet

| Want to calculate | Use this syntax |
|-------------------|-----------------|
| P(X) | `p(x)` |
| P(X=a) | `p(x == a)` |
| P(X>a) | `p(x > a)` |
| P(X∈{a,b,c}) | `p(x.isin([a,b,c]))` |
| P(X,Y) | `p(x, y)` |
| P(X=a,Y=b) | `p(x == a, y == b)` |
| P(X\|Y=b) | `p(x).given(y == b)` |
| P(X=a\|Y=b) | `p(x == a).given(y == b)` |
| P(X\|Y=b,Z=c) | `p(x).given(y == b, z == c)` |
| P(a<X<b) | `TernaryExpression(x, a, b)` |
| P(X=a AND Y=b) | `p((x == a) & (y == b))` |
| P(X=a OR Y=b) | `p((x == a) | (y == b))` |

## Data Types Supported

- **Integers**: `x == 5`, `x > 10`
- **Floats**: `x == 3.14`, `x > 2.5`
- **Strings**: `category == 'A'`, `name.isin(['Alice', 'Bob'])`
- **Booleans**: `flag == True`, `is_active == False`
- **Datetime**: `date > '2023-01-01'` (with proper datetime columns)

## Best Practices

1. **Create variables from same dataframe** for a single analysis
2. **Use descriptive variable names** for clarity
3. **Handle exceptions appropriately** with specific exception types
4. **Use Polars for large datasets** for better performance
5. **Leverage type hints** for better IDE support
6. **Cache results** if performing repeated calculations
7. **Use `.to_dict()` or `.to_dataframe()`** for integration with other tools