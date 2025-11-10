# Poffertjes Usage Examples

This document provides comprehensive examples of using Poffertjes for probabilistic queries on dataframes.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Real-World Scenarios](#real-world-scenarios)
3. [Advanced Usage](#advanced-usage)
4. [Performance Examples](#performance-examples)
5. [Integration Examples](#integration-examples)

## Basic Examples

### Simple Coin Flip Analysis

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Create coin flip data
df = pd.DataFrame({
    'flip': ['H', 'T', 'H', 'H', 'T', 'T', 'H', 'T', 'H', 'T'],
    'trial': list(range(1, 11))
})

vb = VariableBuilder.from_data(df)
flip, trial = vb.get_variables('flip', 'trial')

# Basic probabilities
print("Marginal distribution of flips:")
print(p(flip))

print(f"\nP(Heads) = {p(flip == 'H')}")
print(f"P(Tails) = {p(flip == 'T')}")

# Conditional on trial number
print(f"\nP(Heads | trial > 5) = {p(flip == 'H').given(trial > 5)}")
```

### Dice Rolling Analysis

```python
import pandas as pd
import numpy as np
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Simulate dice rolls
np.random.seed(42)
n_rolls = 1000

df = pd.DataFrame({
    'die1': np.random.randint(1, 7, n_rolls),
    'die2': np.random.randint(1, 7, n_rolls)
})
df['sum'] = df['die1'] + df['die2']
df['is_double'] = df['die1'] == df['die2']

vb = VariableBuilder.from_data(df)
die1, die2, sum_dice, is_double = vb.get_variables('die1', 'die2', 'sum', 'is_double')

# Analyze dice probabilities
print("Distribution of first die:")
print(p(die1))

print(f"\nP(sum = 7) = {p(sum_dice == 7)}")
print(f"P(sum > 10) = {p(sum_dice > 10)}")
print(f"P(double) = {p(is_double == True)}")

# Joint probabilities
print(f"\nP(die1=6, die2=6) = {p(die1 == 6, die2 == 6)}")

# Conditional probabilities
print(f"P(sum=12 | double) = {p(sum_dice == 12).given(is_double == True)}")
print(f"P(die1=6 | sum>10) = {p(die1 == 6).given(sum_dice > 10)}")
```

## Real-World Scenarios

### Customer Purchase Analysis

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Customer data
df = pd.DataFrame({
    'age_group': ['18-25', '26-35', '36-45', '46-55', '55+'] * 200,
    'income_level': ['low', 'medium', 'high'] * 333 + ['low'],
    'purchased': [True, False] * 500,
    'product_category': ['electronics', 'clothing', 'books', 'home'] * 250,
    'season': ['spring', 'summer', 'fall', 'winter'] * 250
})

vb = VariableBuilder.from_data(df)
age_group, income_level, purchased, product_category, season = vb.get_variables(
    'age_group', 'income_level', 'purchased', 'product_category', 'season'
)

# Purchase probability analysis
print("Overall purchase probability:")
print(f"P(purchased) = {p(purchased == True)}")

print("\nPurchase probability by age group:")
for age in df['age_group'].unique():
    prob = p(purchased == True).given(age_group == age)
    print(f"P(purchased | age={age}) = {prob}")

print("\nPurchase probability by income level:")
purchase_by_income = p(purchased == True).given(income_level)
print(purchase_by_income)

# Product preferences by demographics
print("\nProduct category preferences by age group:")
product_given_age = p(product_category).given(age_group == '26-35')
print(product_given_age)

# Complex conditional analysis
print(f"\nP(electronics | high income, purchased) = {p(product_category == 'electronics').given(income_level == 'high', purchased == True)}")

# Seasonal analysis
print("\nSeasonal purchase patterns:")
seasonal_purchases = p(purchased == True).given(season)
print(seasonal_purchases)
```

### Medical Diagnosis Example

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Medical test data (synthetic)
df = pd.DataFrame({
    'age': [25, 45, 65, 30, 55, 70, 35, 60, 40, 50] * 100,
    'test_positive': [True, False] * 500,
    'has_condition': [True, False, False, True, True, False, False, True, False, True] * 100,
    'severity': ['mild', 'moderate', 'severe', 'none'] * 250,
    'treatment_response': ['good', 'fair', 'poor'] * 333 + ['good']
})

vb = VariableBuilder.from_data(df)
age, test_positive, has_condition, severity, treatment_response = vb.get_variables(
    'age', 'test_positive', 'has_condition', 'severity', 'treatment_response'
)

# Diagnostic accuracy
print("Test accuracy analysis:")
print(f"P(test positive | has condition) = {p(test_positive == True).given(has_condition == True)}")  # Sensitivity
print(f"P(test negative | no condition) = {p(test_positive == False).given(has_condition == False)}")  # Specificity

# Bayes' theorem in action
print(f"\nP(has condition | test positive) = {p(has_condition == True).given(test_positive == True)}")  # Positive predictive value
print(f"P(no condition | test negative) = {p(has_condition == False).given(test_positive == False)}")  # Negative predictive value

# Age-related analysis
print(f"\nP(has condition | age > 60) = {p(has_condition == True).given(age > 60)}")
print(f"P(severe | has condition, age > 50) = {p(severity == 'severe').given(has_condition == True, age > 50)}")

# Treatment effectiveness
print(f"\nP(good response | severe condition) = {p(treatment_response == 'good').given(severity == 'severe')}")
```

### Financial Risk Analysis

```python
import pandas as pd
import numpy as np
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Financial data
np.random.seed(123)
n_customers = 1000

df = pd.DataFrame({
    'credit_score': np.random.normal(650, 100, n_customers).astype(int),
    'income': np.random.lognormal(10, 0.5, n_customers).astype(int),
    'debt_ratio': np.random.beta(2, 5, n_customers),
    'default': np.random.choice([True, False], n_customers, p=[0.1, 0.9]),
    'loan_amount': np.random.uniform(5000, 50000, n_customers).astype(int)
})

# Create risk categories
df['credit_category'] = pd.cut(df['credit_score'], 
                              bins=[0, 600, 700, 800, 1000], 
                              labels=['poor', 'fair', 'good', 'excellent'])
df['income_category'] = pd.cut(df['income'], 
                              bins=[0, 30000, 60000, 100000, np.inf], 
                              labels=['low', 'medium', 'high', 'very_high'])

vb = VariableBuilder.from_data(df)
credit_score, income, debt_ratio, default, loan_amount, credit_category, income_category = vb.get_variables(
    'credit_score', 'income', 'debt_ratio', 'default', 'loan_amount', 'credit_category', 'income_category'
)

# Risk analysis
print("Default probability analysis:")
print(f"Overall default rate: {p(default == True)}")

print("\nDefault rate by credit category:")
default_by_credit = p(default == True).given(credit_category)
print(default_by_credit)

print("\nDefault rate by income category:")
default_by_income = p(default == True).given(income_category)
print(default_by_income)

# High-risk combinations
print(f"\nP(default | poor credit, high debt) = {p(default == True).given(credit_category == 'poor', debt_ratio > 0.5)}")
print(f"P(default | low income, large loan) = {p(default == True).given(income_category == 'low', loan_amount > 40000)}")

# Joint risk factors
print("\nJoint distribution of credit and income categories:")
joint_dist = p(credit_category, income_category)
print(joint_dist)
```

## Advanced Usage

### Complex Conditional Queries

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder
from poffertjes.expression import TernaryExpression

# Complex dataset
df = pd.DataFrame({
    'temperature': np.random.normal(20, 10, 1000),
    'humidity': np.random.uniform(30, 90, 1000),
    'pressure': np.random.normal(1013, 20, 1000),
    'weather': np.random.choice(['sunny', 'cloudy', 'rainy'], 1000),
    'sales': np.random.poisson(50, 1000)
})

vb = VariableBuilder.from_data(df)
temperature, humidity, pressure, weather, sales = vb.get_variables(
    'temperature', 'humidity', 'pressure', 'weather', 'sales'
)

# Complex range conditions
temp_range = TernaryExpression(temperature, 15, 25, closed="both")  # 15 <= temp <= 25
print(f"P(comfortable temperature) = {p(temp_range)}")

# Multiple range conditions
comfortable_conditions = (
    (temperature > 18) & (temperature < 26) & 
    (humidity > 40) & (humidity < 70)
)
print(f"P(comfortable conditions) = {p(comfortable_conditions)}")

# Complex conditional analysis
print(f"P(high sales | comfortable, sunny) = {p(sales > 60).given(comfortable_conditions, weather == 'sunny')}")

# Chained conditionals
weather_dist = p(weather)
weather_given_temp = p(weather).given(temperature > 25)
sales_given_weather_temp = p(sales > 50).given(weather == 'sunny', temperature > 20)

print("Weather distribution:", weather_dist)
print("Weather given high temp:", weather_given_temp)
print("High sales probability:", sales_given_weather_temp)
```

### Working with Time Series Data

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Generate time series data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

df = pd.DataFrame({
    'date': dates,
    'day_of_week': [d.strftime('%A') for d in dates],
    'month': [d.strftime('%B') for d in dates],
    'is_weekend': [d.weekday() >= 5 for d in dates],
    'sales': np.random.poisson(100, 365) + np.random.normal(0, 20, 365).astype(int),
    'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365)
})

# Add categorical variables
df['sales_category'] = pd.cut(df['sales'], bins=[0, 80, 120, np.inf], labels=['low', 'medium', 'high'])
df['temp_category'] = pd.cut(df['temperature'], bins=[-np.inf, 10, 25, np.inf], labels=['cold', 'mild', 'hot'])

vb = VariableBuilder.from_data(df)
day_of_week, month, is_weekend, sales, temperature, sales_category, temp_category = vb.get_variables(
    'day_of_week', 'month', 'is_weekend', 'sales', 'temperature', 'sales_category', 'temp_category'
)

# Temporal analysis
print("Sales patterns by day of week:")
sales_by_day = p(sales_category).given(day_of_week)
print(sales_by_day)

print(f"\nP(high sales | weekend) = {p(sales_category == 'high').given(is_weekend == True)}")
print(f"P(high sales | weekday) = {p(sales_category == 'high').given(is_weekend == False)}")

# Seasonal patterns
print("\nSeasonal sales patterns:")
seasonal_sales = p(sales_category).given(month.isin(['December', 'January', 'February']))
print(seasonal_sales)

# Weather impact
print(f"\nP(high sales | hot weather) = {p(sales_category == 'high').given(temp_category == 'hot')}")
print(f"P(low sales | cold weather) = {p(sales_category == 'low').given(temp_category == 'cold')}")
```

## Performance Examples

### Large Dataset Handling with Polars

```python
import polars as pl
import numpy as np
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Create large dataset with Polars for better performance
n_rows = 1_000_000

large_df = pl.DataFrame({
    'user_id': range(n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
    'value': np.random.exponential(10, n_rows),
    'flag': np.random.choice([True, False], n_rows, p=[0.3, 0.7]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows)
})

vb = VariableBuilder.from_data(large_df)
user_id, category, value, flag, region = vb.get_variables('user_id', 'category', 'value', 'flag', 'region')

# Efficient probability calculations using lazy evaluation
print("Working with 1M rows using Polars...")

# These operations use lazy evaluation internally
category_dist = p(category)
high_value_prob = p(value > 20)
regional_patterns = p(flag == True).given(region)

print("Category distribution:", category_dist)
print("High value probability:", high_value_prob)
print("Regional flag patterns:", regional_patterns)

# Complex query on large dataset
complex_prob = p(flag == True).given(
    (category == 'A') & (value > 15), 
    region.isin(['North', 'East'])
)
print("Complex conditional probability:", complex_prob)
```

### Batch Processing Example

```python
import pandas as pd
from poffertjes import p
from poffertjes.variable import VariableBuilder

def analyze_batch(df_batch, batch_id):
    """Analyze a batch of data and return key metrics."""
    vb = VariableBuilder.from_data(df_batch)
    
    # Assume standard columns
    if 'outcome' in df_batch.columns and 'feature1' in df_batch.columns:
        outcome, feature1 = vb.get_variables('outcome', 'feature1')
        
        metrics = {
            'batch_id': batch_id,
            'success_rate': float(p(outcome == 'success')),
            'feature1_high_prob': float(p(feature1 > df_batch['feature1'].median())),
            'conditional_success': float(p(outcome == 'success').given(feature1 > df_batch['feature1'].median()))
        }
        return metrics
    return None

# Simulate batch processing
batches = []
for i in range(5):
    batch_df = pd.DataFrame({
        'outcome': np.random.choice(['success', 'failure'], 1000, p=[0.6, 0.4]),
        'feature1': np.random.normal(50, 15, 1000),
        'feature2': np.random.uniform(0, 100, 1000)
    })
    
    metrics = analyze_batch(batch_df, i)
    if metrics:
        batches.append(metrics)

# Analyze batch results
batch_results = pd.DataFrame(batches)
print("Batch analysis results:")
print(batch_results)
```

## Integration Examples

### Integration with Scikit-learn

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(5)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Add some categorical features
df['category'] = np.random.choice(['A', 'B', 'C'], 1000)
df['segment'] = pd.cut(df['feature_0'], bins=3, labels=['low', 'medium', 'high'])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test['target'] = y_test
df_test['predicted'] = y_pred
df_test['correct'] = df_test['target'] == df_test['predicted']

# Analyze model performance with poffertjes
vb = VariableBuilder.from_data(df_test)
target, predicted, correct = vb.get_variables('target', 'predicted', 'correct')

print("Model Performance Analysis:")
print(f"Overall accuracy: {p(correct == True)}")
print(f"Precision (class 1): {p(target == 1).given(predicted == 1)}")
print(f"Recall (class 1): {p(predicted == 1).given(target == 1)}")

# Feature-based analysis
for i, feature in enumerate(feature_names):
    feature_var = vb.get_variables(feature)
    high_feature = feature_var > df_test[feature].median()
    accuracy_high = p(correct == True).given(high_feature)
    print(f"Accuracy when {feature} > median: {accuracy_high}")
```

### Integration with Matplotlib for Visualization

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Generate data
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

# Add derived variables
df['quadrant'] = np.where(df['x'] > 0, 
                         np.where(df['y'] > 0, 'Q1', 'Q4'),
                         np.where(df['y'] > 0, 'Q2', 'Q3'))

vb = VariableBuilder.from_data(df)
x, y, category, quadrant = vb.get_variables('x', 'y', 'category', 'quadrant')

# Calculate probabilities for visualization
quadrant_probs = p(quadrant).to_dict()
category_probs = p(category).to_dict()

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Quadrant distribution
ax1.bar(quadrant_probs.keys(), quadrant_probs.values())
ax1.set_title('Probability Distribution by Quadrant')
ax1.set_ylabel('Probability')

# Category distribution
ax2.pie(category_probs.values(), labels=category_probs.keys(), autopct='%1.1f%%')
ax2.set_title('Category Distribution')

# Conditional probabilities
categories = df['category'].unique()
quadrants = df['quadrant'].unique()
cond_probs = np.zeros((len(categories), len(quadrants)))

for i, cat in enumerate(categories):
    for j, quad in enumerate(quadrants):
        prob = float(p(quadrant == quad).given(category == cat))
        cond_probs[i, j] = prob

im = ax3.imshow(cond_probs, cmap='Blues')
ax3.set_xticks(range(len(quadrants)))
ax3.set_yticks(range(len(categories)))
ax3.set_xticklabels(quadrants)
ax3.set_yticklabels(categories)
ax3.set_title('P(Quadrant | Category)')

# Add text annotations
for i in range(len(categories)):
    for j in range(len(quadrants)):
        text = ax3.text(j, i, f'{cond_probs[i, j]:.3f}',
                       ha="center", va="center", color="black")

# Scatter plot with probability-based coloring
scatter_data = []
for _, row in df.iterrows():
    prob = float(p(category == row['category']).given(
        (x > row['x'] - 0.1) & (x < row['x'] + 0.1)
    ))
    scatter_data.append(prob)

scatter = ax4.scatter(df['x'], df['y'], c=scatter_data, cmap='viridis', alpha=0.6)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Scatter Plot Colored by Local Category Probability')
plt.colorbar(scatter, ax=ax4)

plt.tight_layout()
plt.show()

# Print some interesting conditional probabilities
print("\nInteresting Conditional Probabilities:")
print(f"P(Q1 | Category A) = {p(quadrant == 'Q1').given(category == 'A')}")
print(f"P(Category A | Q1) = {p(category == 'A').given(quadrant == 'Q1')}")
print(f"P(x > 0 | Category B) = {p(x > 0).given(category == 'B')}")
```

### Integration with Statistical Tests

```python
import pandas as pd
import numpy as np
from scipy import stats
from poffertjes import p
from poffertjes.variable import VariableBuilder

# Generate data for A/B test
np.random.seed(42)
n_control, n_treatment = 1000, 1000

df = pd.DataFrame({
    'group': ['control'] * n_control + ['treatment'] * n_treatment,
    'converted': (
        np.random.binomial(1, 0.1, n_control).tolist() +  # 10% conversion in control
        np.random.binomial(1, 0.12, n_treatment).tolist()  # 12% conversion in treatment
    ),
    'revenue': np.concatenate([
        np.random.exponential(50, n_control),  # Lower revenue in control
        np.random.exponential(55, n_treatment)  # Higher revenue in treatment
    ])
})

vb = VariableBuilder.from_data(df)
group, converted, revenue = vb.get_variables('group', 'converted', 'revenue')

# Calculate conversion rates using poffertjes
control_rate = p(converted == 1).given(group == 'control')
treatment_rate = p(converted == 1).given(group == 'treatment')

print("A/B Test Analysis:")
print(f"Control conversion rate: {control_rate}")
print(f"Treatment conversion rate: {treatment_rate}")
print(f"Lift: {(float(treatment_rate) / float(control_rate) - 1) * 100:.2f}%")

# Statistical significance test
control_conversions = df[df['group'] == 'control']['converted'].sum()
treatment_conversions = df[df['group'] == 'treatment']['converted'].sum()

# Chi-square test
contingency_table = np.array([
    [control_conversions, n_control - control_conversions],
    [treatment_conversions, n_treatment - treatment_conversions]
])

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nStatistical significance (chi-square test):")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Revenue analysis
high_revenue_threshold = df['revenue'].quantile(0.75)
high_revenue_control = p(revenue > high_revenue_threshold).given(group == 'control')
high_revenue_treatment = p(revenue > high_revenue_threshold).given(group == 'treatment')

print(f"\nRevenue Analysis:")
print(f"P(high revenue | control): {high_revenue_control}")
print(f"P(high revenue | treatment): {high_revenue_treatment}")

# Conditional conversion analysis
print(f"\nConditional Analysis:")
print(f"P(converted | high revenue, control): {p(converted == 1).given(revenue > high_revenue_threshold, group == 'control')}")
print(f"P(converted | high revenue, treatment): {p(converted == 1).given(revenue > high_revenue_threshold, group == 'treatment')}")
```

These examples demonstrate the versatility and power of Poffertjes for probabilistic analysis across various domains and use cases. The library's intuitive syntax makes complex probability calculations accessible while maintaining computational efficiency through the Narwhals backend.