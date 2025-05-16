import pandas as pd
import numpy as np

from poffertjes import p, Variable
from typing import Any


def test_marginal_probability_single_variable(df: pd.DataFrame, x: Variable) -> None:
    pd.testing.assert_series_equal(df[x.name].value_counts(normalize=True), p(x))


def test_conditional_probability_single_variable_single_condition(
    df: pd.DataFrame, x: Variable, y: Variable, y0: Any
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[y.name] == y0, x.name].value_counts(normalize=True), p(x | (y == y0))
    )


def test_conditional_probability_single_variable_condition_tuple(
    df: pd.DataFrame, x: Variable, y: Variable, z: Variable, y0: Any, z0: Any
) -> None:
    pd.testing.assert_series_equal(
        df.loc[(df[y.name] == y0) & (df[z.name] == z0), x.name].value_counts(
            normalize=True
        ),
        p(x | (y == y0, z == z0)),
    )


def test_simmetry_conditional_probability_single_variable_condition_tuple(
    x: Variable, y: Variable, z: Variable, y0: Any, z0: Any
) -> None:
    pd.testing.assert_series_equal(
        p(x | (z == z0, y == y0)),
        p(x | (y == y0, z == z0)),
    )


def test_pointwise_marginal_probability(df: pd.DataFrame, x: Variable, x0: Any) -> None:
    assert (len(df.loc[df[x.name] == x0]) / len(df)) == p(x == x0)


def test_pointwise_conditional_probability_single_variable_single_condition(
    df: pd.DataFrame, x: Variable, y: Variable, x0: Any, y0: Any
) -> None:
    assert (
        len(df.loc[(df[x.name] == x0) & (df[y.name] == y0)])
        / len(df.loc[df[x.name] == x0])
        if not df.loc[df[x.name] == x0].empty
        else 0
    ) == p((y == y0) | (x == x0))


def test_simmetry_joint_probability_two_variables(
    x: Variable, y: Variable, x0: Any, y0: Any
) -> None:
    assert np.isclose(p(x == x0, y == y0), p(y == y0, x == x0))


# New tests for range conditions
def test_range_condition_less_than(
    df: pd.DataFrame, x: Variable, upper: float
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[x.name] < upper, x.name].value_counts(normalize=True),
        p(x < upper)
    )


def test_range_condition_greater_than(
    df: pd.DataFrame, x: Variable, lower: float
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[x.name] > lower, x.name].value_counts(normalize=True),
        p(x > lower)
    )


def test_range_condition_less_than_equal(
    df: pd.DataFrame, x: Variable, upper: float
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[x.name] <= upper, x.name].value_counts(normalize=True),
        p(x <= upper)
    )


def test_range_condition_greater_than_equal(
    df: pd.DataFrame, x: Variable, lower: float
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[x.name] >= lower, x.name].value_counts(normalize=True),
        p(x >= lower)
    )


def test_range_condition_not_equal(
    df: pd.DataFrame, x: Variable, value: Any
) -> None:
    pd.testing.assert_series_equal(
        df.loc[df[x.name] != value, x.name].value_counts(normalize=True),
        p(x != value)
    )


# Tests for different data types
def test_categorical_variable(df_cat: pd.DataFrame, cat_var: Variable, category: str) -> None:
    pd.testing.assert_series_equal(
        df_cat.loc[df_cat[cat_var.name] == category, cat_var.name].value_counts(normalize=True),
        p(cat_var | (cat_var == category))
    )


def test_boolean_variable(df_bool: pd.DataFrame, bool_var: Variable) -> None:
    pd.testing.assert_series_equal(
        df_bool.loc[df_bool[bool_var.name] == True, bool_var.name].value_counts(normalize=True),
        p(bool_var | (bool_var == True))
    )


def test_float_variable_distribution(df: pd.DataFrame, float_var: Variable) -> None:
    pd.testing.assert_series_equal(
        df[float_var.name].value_counts(normalize=True),
        p(float_var)
    )


def test_integer_variable_distribution(df: pd.DataFrame, int_var: Variable) -> None:
    pd.testing.assert_series_equal(
        df[int_var.name].value_counts(normalize=True),
        p(int_var)
    )


# Test for variable data type detection
def test_variable_dtype_detection(
    df_mixed: pd.DataFrame, 
    cat_var: Variable, 
    bool_var: Variable,
    float_var: Variable,
    int_var: Variable
) -> None:
    assert cat_var.dtype == "categorical"
    assert bool_var.dtype == "boolean"
    assert float_var.dtype == "float"
    assert int_var.dtype == "integer"