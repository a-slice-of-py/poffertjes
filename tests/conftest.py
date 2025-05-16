import pytest
import pandas as pd
import numpy as np
from poffertjes import VariableBuilder


@pytest.fixture
def df():
    """Create a sample dataframe with numeric data."""
    N_SAMPLES = 100
    columns = ["x", "y", "z", "u"]
    
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame(
        dict(
            zip(
                columns,
                [np.random.randn(N_SAMPLES).transpose() for _ in range(len(columns))],
            )
        )
    ).map(lambda x: 10 * round(abs(x), 1))
    
    return df


@pytest.fixture
def vb(df):
    """Create a variable builder from the sample dataframe."""
    return VariableBuilder.from_data(df)


@pytest.fixture
def x(vb):
    """Create a variable for column 'x'."""
    return vb.Variable("x")


@pytest.fixture
def y(vb):
    """Create a variable for column 'y'."""
    return vb.Variable("y")


@pytest.fixture
def z(vb):
    """Create a variable for column 'z'."""
    return vb.Variable("z")


@pytest.fixture
def u(vb):
    """Create a variable for column 'u'."""
    return vb.Variable("u")


@pytest.fixture
def x0(df):
    """Get a sample value from column 'x'."""
    return df.iloc[0]["x"]


@pytest.fixture
def y0(df):
    """Get a sample value from column 'y'."""
    return df.iloc[0]["y"]


@pytest.fixture
def z0(df):
    """Get a sample value from column 'z'."""
    return df.iloc[0]["z"]


@pytest.fixture
def u0(df):
    """Get a sample value from column 'u'."""
    return df.iloc[0]["u"]


@pytest.fixture
def lower():
    """Get a lower bound for range tests."""
    return 5.0


@pytest.fixture
def upper():
    """Get an upper bound for range tests."""
    return 15.0


@pytest.fixture
def df_cat():
    """Create a sample dataframe with categorical data."""
    N_SAMPLES = 100
    categories = ["A", "B", "C", "D"]
    
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        "category": np.random.choice(categories, size=N_SAMPLES),
        "value": np.random.randn(N_SAMPLES)
    })
    df["category"] = df["category"].astype("category")
    
    return df


@pytest.fixture
def df_bool():
    """Create a sample dataframe with boolean data."""
    N_SAMPLES = 100
    
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        "flag": np.random.choice([True, False], size=N_SAMPLES),
        "value": np.random.randn(N_SAMPLES)
    })
    
    return df


@pytest.fixture
def df_mixed():
    """Create a sample dataframe with mixed data types."""
    N_SAMPLES = 100
    categories = ["A", "B", "C", "D"]
    
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        "category": np.random.choice(categories, size=N_SAMPLES),
        "flag": np.random.choice([True, False], size=N_SAMPLES),
        "float_val": np.random.randn(N_SAMPLES),
        "int_val": np.random.randint(1, 100, size=N_SAMPLES)
    })
    df["category"] = df["category"].astype("category")
    
    return df


@pytest.fixture
def vb_cat(df_cat):
    """Create a variable builder from the categorical dataframe."""
    return VariableBuilder.from_data(df_cat)


@pytest.fixture
def vb_bool(df_bool):
    """Create a variable builder from the boolean dataframe."""
    return VariableBuilder.from_data(df_bool)


@pytest.fixture
def vb_mixed(df_mixed):
    """Create a variable builder from the mixed dataframe."""
    return VariableBuilder.from_data(df_mixed)


@pytest.fixture
def cat_var(vb_cat):
    """Create a variable for the categorical column."""
    return vb_cat.Variable("category")


@pytest.fixture
def bool_var(vb_bool):
    """Create a variable for the boolean column."""
    return vb_bool.Variable("flag")


@pytest.fixture
def float_var(vb_mixed):
    """Create a variable for the float column."""
    return vb_mixed.Variable("float_val")


@pytest.fixture
def int_var(vb_mixed):
    """Create a variable for the integer column."""
    return vb_mixed.Variable("int_val")


@pytest.fixture
def category():
    """Get a sample category value."""
    return "A"