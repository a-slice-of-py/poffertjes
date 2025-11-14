# 🥞 Poffertjes

Friendly, pythonic interface for running probabilistic queries on dataframes.

> [!CAUTION]
> This is a personal experiment of coding a library with a mixed approach between TDD and spec-driven development. It's **not meant for production use** since I currently do not expect to actively maintain it.

<details>
    
<summary>Why "Poffertjes"?</summary><br>
    
The inspiration for this library's "friendly interface" came from [Vincent Warmerdam's `peegeem`](https://github.com/koaning/peegeem) and I wanted to pay him tribute. When I was a kid, I visited the Netherlands and fell in love with [poffertjes](https://en.wikipedia.org/wiki/Poffertjes): since this project is filled with _syntactic sugar_, these sweet treats seemed like the perfect fit for the name!

</details>

## Features

- **Mathematical notation**: Write probability queries like `p(x)`, `p(x == 5)`, `p(x).given(y)`
- **Dataframe agnostic**: Works with both Pandas and Polars through Narwhals
- **Efficient computation**: Uses lazy evaluation and optimized operations
- **Type safe**: Full type hints and comprehensive error handling
- **Comprehensive**: Supports marginal, conditional, and joint probabilities

## Quick Start

```python
import pandas as pd
from poffertjes import p, VariableBuilder

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 25, 35, 30, 25],
    'income': ['low', 'high', 'low', 'high', 'medium', 'low'],
    'purchased': [True, True, False, True, False, True]
})

# Extract variables from dataframe
vb = VariableBuilder.from_data(df)
age, income, purchased = vb.get_variables('age', 'income', 'purchased')

# Calculate probabilities
print(p(age))                           # Marginal distribution of age
print(p(purchased == True))             # P(purchased = True) = 0.666667
print(p(age).given(income == 'high'))   # P(age | income = 'high')
print(p(age, income))                   # Joint distribution of age and income

# Works with Polars too!
import polars as pl
df_polars = pl.DataFrame(df)
vb_polars = VariableBuilder.from_data(df_polars)
age_pl, income_pl = vb_polars.get_variables('age', 'income')
print(p(age_pl > 25))                   # P(age > 25) with Polars backend
```

## Installation

### From PyPI (when available)
```bash
pip install poffertjes
```

### From Source
```bash
git clone https://github.com/your-repo/poffertjes
cd poffertjes
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/your-repo/poffertjes
cd poffertjes
pip install -e ".[dev]"  # Includes test dependencies
```

### Requirements
- Python 3.8+
- pandas or polars (at least one)
- narwhals (for dataframe abstraction)

The library automatically detects and works with your installed dataframe library.

## API Reference

### Core Interface

#### `p(*args)` - Probability Query Function
The main interface for probability calculations. Import with `from poffertjes import p`.

```python
# Marginal probabilities
p(x)                    # Returns DistributionResult
p(x == value)          # Returns ScalarResult

# Joint probabilities  
p(x, y)                # Returns DistributionResult
p(x == v1, y == v2)    # Returns ScalarResult

# All results support .given() for conditional probabilities
p(x).given(y == value)          # Conditional distribution
p(x == v1).given(y == v2)       # Conditional scalar
```

#### `VariableBuilder` - Variable Factory
Creates Variable objects from dataframes.

```python
from poffertjes import VariableBuilder

vb = VariableBuilder.from_data(df)      # Create from dataframe
variables = vb.get_variables()          # All columns
x, y = vb.get_variables('x', 'y')      # Specific columns
```

### Variable Class
Represents a random variable bound to a dataframe column.

```python
# Comparison operators (return Expression objects)
x == value, x != value                  # Equality/inequality
x < value, x <= value                   # Less than
x > value, x >= value                   # Greater than
x.isin([val1, val2, ...])              # Membership

# Properties
x.name                                  # Column name
x.dataframe_id                         # Unique dataframe identifier
```

### Expression Classes

#### `Expression` - Single Condition
Represents a condition on a variable (e.g., x == 5).

```python
expr = x == 5
expr.variable                           # The variable
expr.operator                          # ExpressionOp enum
expr.value                             # The comparison value
expr.to_narwhals_expr()                # Convert to Narwhals expression

# Combine expressions
(x == 1) & (y == 2)                   # AND combination
(x == 1) | (y == 2)                   # OR combination
```

#### `CompositeExpression` - Multiple Conditions
Represents combined expressions with AND/OR logic.

#### `TernaryExpression` - Range Conditions
Represents range conditions like a < x < b.

```python
from poffertjes import TernaryExpression

# Create ternary expressions
TernaryExpression(x, 1, 10)            # 1 < x < 10 (exclusive)
TernaryExpression(x, 1, 10, "both")    # 1 ≤ x ≤ 10 (inclusive)
TernaryExpression(x, 1, 10, "left")    # 1 ≤ x < 10
TernaryExpression(x, 1, 10, "right")   # 1 < x ≤ 10
```

### Result Classes

#### `ScalarResult` - Single Probability Value
Returned for queries like `p(x == 5)`.

```python
result = p(x == 5)
float(result)                          # Convert to float
result.value                           # Access probability value
result.given(y == 2)                   # Conditional probability
```

#### `DistributionResult` - Probability Distribution
Returned for queries like `p(x)`.

```python
result = p(x)
result.given(y == 2)                   # Conditional distribution
result.to_dict()                       # Convert to dictionary
result.to_dataframe()                  # Convert to dataframe
```

#### `Distribution` - Distribution Data
Contains the actual probability distribution data.

```python
dist = p(x).distribution
for value, prob in dist:               # Iterate over (value, probability)
    print(f"P(X={value}) = {prob}")

dict_result = dist.to_dict()           # {value: probability, ...}
df_result = dist.to_dataframe()        # Native dataframe format
```

### Exception Classes
All exceptions inherit from `PoffertjesError`.

```python
from poffertjes import (
    PoffertjesError,      # Base exception
    DataframeError,       # Dataframe-related errors
    VariableError,        # Variable-related errors  
    ExpressionError,      # Expression-related errors
    ProbabilityError      # Probability calculation errors
)
```

## Documentation

- **[API Documentation](docs/api.md)** - Complete API reference with all classes and methods
- **[Usage Examples](docs/examples.md)** - Comprehensive examples for real-world scenarios
- **[Quick Reference](docs/quick_reference.md)** - Cheat sheet for common operations

## Key Concepts

### Variables
Extract variables from dataframe columns using the VariableBuilder:
```python
from poffertjes import VariableBuilder

# Create variables from all columns
vb = VariableBuilder.from_data(df)
variables = vb.get_variables()  # All columns

# Create variables from specific columns
x, y, z = vb.get_variables('x', 'y', 'z')

# Variables are bound to their source dataframe
print(x)  # Variable(x)
```

### Probability Queries
Use mathematical notation for intuitive probability calculations:
```python
# Marginal probabilities
p(x)                    # Distribution: P(X=x) for all x
p(x == 5)              # Scalar: P(X=5)
p(x > 10)              # Scalar: P(X>10)
p(x.isin([1,2,3]))     # Scalar: P(X ∈ {1,2,3})

# Joint probabilities  
p(x, y)                # Distribution: P(X=x, Y=y) for all x,y
p(x == 1, y == 2)      # Scalar: P(X=1, Y=2)

# Conditional probabilities
p(x).given(y == 2)     # Distribution: P(X=x | Y=2) for all x
p(x == 1).given(y == 2) # Scalar: P(X=1 | Y=2)
p(x, y).given(z == 3)  # Joint conditional: P(X,Y | Z=3)
```

### Expressions and Operators
Create complex conditions using comparison and logical operators:
```python
# Comparison operators
x == 5                  # Equality
x != 5                  # Inequality  
x > 5, x >= 5          # Greater than (or equal)
x < 5, x <= 5          # Less than (or equal)
x.isin([1, 2, 3])      # Membership test

# Logical combinations
(x > 5) & (x < 10)     # AND: 5 < x < 10
(x == 1) | (x == 2)    # OR: x = 1 or x = 2

# Ternary expressions (between)
from poffertjes import TernaryExpression
TernaryExpression(x, 5, 10)  # 5 < x < 10 (exclusive)
```

## Supported Data Types

- **Numeric**: integers, floats
- **Categorical**: strings, categories
- **Boolean**: True/False values
- **Datetime**: date and time columns

## Performance

Poffertjes is built on Narwhals for efficient, dataframe-agnostic operations:

- **Lazy evaluation**: Computations are optimized and deferred when possible
- **Vectorized operations**: Uses efficient group-by and aggregation patterns
- **Memory efficient**: Shares dataframe references, avoids unnecessary copying
- **Scalable**: Works well with large datasets, especially with Polars backend

## Examples

### A/B Testing Analysis
```python
import pandas as pd
from poffertjes import p, VariableBuilder

# A/B test data
df = pd.DataFrame({
    'user_id': range(1000),
    'group': ['control'] * 500 + ['treatment'] * 500,
    'converted': [True, False] * 250 + [True] * 300 + [False] * 200,
    'revenue': [0, 10, 0, 15] * 250  # Revenue per conversion
})

vb = VariableBuilder.from_data(df)
group, converted, revenue = vb.get_variables('group', 'converted', 'revenue')

# Calculate conversion rates
control_rate = p(converted == True).given(group == 'control')
treatment_rate = p(converted == True).given(group == 'treatment')

print(f"Control conversion rate: {float(control_rate):.2%}")
print(f"Treatment conversion rate: {float(treatment_rate):.2%}")

# Calculate lift
lift = (float(treatment_rate) / float(control_rate) - 1) * 100
print(f"Conversion lift: {lift:.1f}%")

# Revenue analysis
avg_revenue_control = p(revenue).given(group == 'control', converted == True)
avg_revenue_treatment = p(revenue).given(group == 'treatment', converted == True)
```

### Customer Segmentation
```python
# Customer data
df = pd.DataFrame({
    'age': [25, 35, 45, 30, 50, 28, 40, 33],
    'income': ['low', 'high', 'high', 'medium', 'high', 'low', 'medium', 'high'],
    'purchased': [False, True, True, False, True, False, True, True],
    'amount': [0, 150, 300, 0, 500, 0, 200, 250]
})

vb = VariableBuilder.from_data(df)
age, income, purchased, amount = vb.get_variables('age', 'income', 'purchased', 'amount')

# Purchase probability by demographics
purchase_by_income = p(purchased == True).given(income)
print("Purchase rates by income:")
for income_level, prob in purchase_by_income.to_dict().items():
    print(f"  {income_level}: {prob:.2%}")

# High-value customer probability
high_value = p(amount > 200).given(purchased == True)
print(f"High-value customer rate: {float(high_value):.2%}")

# Age distribution of purchasers
age_dist_purchasers = p(age).given(purchased == True)
print("Age distribution of purchasers:")
print(age_dist_purchasers)
```

### Risk Analysis
```python
# Credit risk data
df = pd.DataFrame({
    'credit_score': [720, 650, 580, 700, 550, 680, 620, 750],
    'debt_ratio': [0.2, 0.4, 0.6, 0.3, 0.7, 0.25, 0.45, 0.15],
    'income': [50000, 35000, 25000, 60000, 20000, 45000, 30000, 80000],
    'default': [False, False, True, False, True, False, True, False]
})

vb = VariableBuilder.from_data(df)
credit_score, debt_ratio, income, default = vb.get_variables(
    'credit_score', 'debt_ratio', 'income', 'default'
)

# Default probability with multiple risk factors
high_risk_default = p(default == True).given(
    (credit_score < 600) & (debt_ratio > 0.4) & (income < 30000)
)
print(f"High-risk default probability: {float(high_risk_default):.2%}")

# Risk segmentation
low_credit_default = p(default == True).given(credit_score < 600)
high_debt_default = p(default == True).given(debt_ratio > 0.5)
low_income_default = p(default == True).given(income < 30000)

print(f"Low credit score default rate: {float(low_credit_default):.2%}")
print(f"High debt ratio default rate: {float(high_debt_default):.2%}")
print(f"Low income default rate: {float(low_income_default):.2%}")
```

### Time Series Analysis
```python
import pandas as pd
from datetime import datetime, timedelta

# Generate time series data
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
df = pd.DataFrame({
    'date': dates,
    'day_of_week': [d.strftime('%A') for d in dates],
    'sales': [100 + i*2 + (i%7)*10 for i in range(100)],  # Trend + weekly pattern
    'weather': ['sunny', 'rainy', 'cloudy'] * 33 + ['sunny']
})

vb = VariableBuilder.from_data(df)
day_of_week, sales, weather = vb.get_variables('day_of_week', 'sales', 'weather')

# Sales patterns by day of week
sales_by_day = p(sales > 150).given(day_of_week)
print("High sales probability by day:")
for day, prob in sales_by_day.to_dict().items():
    print(f"  {day}: {prob:.2%}")

# Weather impact on sales
sales_by_weather = p(sales > 150).given(weather)
print("High sales probability by weather:")
for weather_type, prob in sales_by_weather.to_dict().items():
    print(f"  {weather_type}: {prob:.2%}")
```

## Error Handling

Poffertjes provides clear, specific error messages:

```python
from poffertjes import PoffertjesError, VariableError, DataframeError

try:
    result = p(x, y)  # Variables from different dataframes
except DataframeError as e:
    print(f"Dataframe error: {e}")
except PoffertjesError as e:
    print(f"General error: {e}")
```

## Resources

- [ProbPy](https://github.com/petermlm/ProbPy) - Probabilistic reasoning in Python
- [distfit](https://github.com/erdogant/distfit) - Probability density fitting
- https://www.perplexity.ai/search/how-to-compute-conditional-pro-J1F8xdG4SL2FbQrGk3k5Hw
- https://stackoverflow.com/questions/33468976/pandas-conditional-probability-of-a-given-specific-b
- https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.density.html
- https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
