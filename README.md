# Poffertjes

Friendly, pythonic interface for running probabilistic queries on dataframes using mathematical notation.

## Features

- **Mathematical notation**: Write probability queries like `p(x)`, `p(x == 5)`, `p(x).given(y)`
- **Dataframe agnostic**: Works with both Pandas and Polars through Narwhals
- **Efficient computation**: Uses lazy evaluation and optimized operations
- **Type safe**: Full type hints and comprehensive error handling
- **Comprehensive**: Supports marginal, conditional, and joint probabilities

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

## Installation

```bash
# When available on PyPI
pip install poffertjes

# For development
git clone https://github.com/your-repo/poffertjes
cd poffertjes
pip install -e .
```

## Documentation

- **[API Documentation](docs/api.md)** - Complete API reference with all classes and methods
- **[Usage Examples](docs/examples.md)** - Comprehensive examples for real-world scenarios
- **[Quick Reference](docs/quick_reference.md)** - Cheat sheet for common operations

## Key Concepts

### Variables
Extract variables from dataframe columns:
```python
vb = VariableBuilder.from_data(df)
x, y = vb.get_variables('x', 'y')
```

### Probability Queries
Use mathematical notation for intuitive probability calculations:
```python
p(x)                    # Marginal distribution
p(x == 5)              # Scalar probability
p(x, y)                # Joint distribution
p(x).given(y == 2)     # Conditional probability
```

### Expressions
Create complex conditions using comparison operators:
```python
x > 5                   # Greater than
x.isin([1, 2, 3])      # Membership test
(x > 5) & (x < 10)     # Combined conditions
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

### A/B Testing
```python
# Calculate conversion rates
control_rate = p(converted == True).given(group == 'control')
treatment_rate = p(converted == True).given(group == 'treatment')
lift = (float(treatment_rate) / float(control_rate) - 1) * 100
```

### Customer Segmentation
```python
# Purchase probability by demographics
purchase_by_age = p(purchased == True).given(age_group)
high_value_customers = p(revenue > 1000).given(
    (age > 30) & (income == 'high')
)
```

### Risk Analysis
```python
# Default probability with multiple risk factors
default_risk = p(default == True).given(
    credit_score < 600,
    debt_ratio > 0.4,
    income < 30000
)
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

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Setting up the development environment
- Running tests
- Submitting pull requests
- Code style and conventions

## Alternatives

- [ProbPy](https://github.com/petermlm/ProbPy) - Probabilistic reasoning in Python
- [distfit](https://github.com/erdogant/distfit) - Probability density fitting

## Resources

- https://www.perplexity.ai/search/how-to-compute-conditional-pro-J1F8xdG4SL2FbQrGk3k5Hw
- https://stackoverflow.com/questions/33468976/pandas-conditional-probability-of-a-given-specific-b
- https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.density.html
- https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html

## Why "Poffertjes"?

The inspiration for this library's "friendly interface" came from [Vincent Warmerdam's `peegeem`](https://github.com/koaning/peegeem) and I wanted to pay him tribute. When I was a kid, I visited the Netherlands and fell in love with [poffertjes](https://en.wikipedia.org/wiki/Poffertjes): since this project is filled with _syntactic sugar_, these sweet treats seemed like the perfect fit for the name!
