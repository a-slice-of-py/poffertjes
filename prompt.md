# Prompt

The current workspace is a repo where I initialized an empty Python packaged library called "poffertjes", managed via uv and published to PyPI, and I want to bootstrap the core functionalities of this library together with you.

The aim of the library is to provide a friendly, pythonic and intuitive interface to run probabilistic queries on dataframes.

Here are some specs that I want you to take into account regarding the library development:

- I want poffertjes to support both Pandas and Polars, so I want it to be dataframe agnostic and implement its backend via Narwhals https://github.com/narwhals-dev/narwhals;
- I want the main interface of poffertjes to be a singleton class "P", to instantiate via `p = P()`, and a `__call__` method where the probabilistic query logic is specified: the end-user of poffertjes is then supposed to use it via `from poffertjes import p`;
- I want p to support both marginal and conditional probabilities: both `p(x)` and `p(x).given(y)` must be supported, so the method call should perform operations (for marginal probabilities) but also return an object with a `given` method to perform conditional probabilities computations;
- I want poffertjes p instance to operate on _variables_ extracted from dataframes and corresponding to dataframe columns and random variables in the probabilistic framework;
- to extract variables from dataframe, I think to adopt and interface such the following:

```python
import narwhals as nw
from narwhals.typing import IntoFrameT

class Variable:
    """Class representing a random variable in the dataset."""
    
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"Variable({self.name})"

    ...

class VariableBuilder:
    def __init__(self, data: IntoFrameT):
        self.data = data

    def get_variables(self, *args) -> List[Variable]:
        if not args:
            return [Variable(name) for name in self.data.columns]
        return [Variable(name) for name in args]

    @staticmethod
    def from_data(data: IntoFrameT) -> "VariableBuilder":
        return VariableBuilder(data)
```

- the previous approach to extract variables from dataframes must then support usage like the following

```python
N_SAMPLES = 100
columns = ["x", "y"]

df = pd.DataFrame(
    dict(
        zip(
            columns,
            [np.random.randn(N_SAMPLES).transpose() for _ in range(len(columns))],
        )
    )
).map(lambda x: 10 * round(abs(x), 1))

vb = VariableBuilder.from_data(df)

x, y = vb.get_variables()
```

- I want poffertjes to support variables of any possible dtypes: int, categorical, float, boolean, ...
- I want poffertjes to support both probability calculation, for example `p(x==0)` as well as returning probability distributions, for example `p(x)`, and both the situation must works for marginal and conditional probabilities
- I want a comprehensive test suite of both unit tests (pytest) and property based test as well (via hypothesis library), to ensure that every combination of marginal and conditional probabilities, distribution vs scalar output, variabile dtypes are well supported by poffertjes
