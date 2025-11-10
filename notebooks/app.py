import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from poffertjes.variable import VariableBuilder
    from poffertjes import p
    return VariableBuilder, np, p, pd


@app.cell
def _(np, pd):
    N_SAMPLES = 10
    columns = ["x", "y", "z", "u"]

    df = pd.DataFrame(
        dict(
            zip(
                columns,
                [np.random.randn(N_SAMPLES).transpose() for _ in range(len(columns))],
            )
        )
    ).map(lambda x: 10 * round(abs(x), 1))
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(VariableBuilder, df):
    builder = VariableBuilder.from_data(df)
    return (builder,)


@app.cell
def _(builder):
    x, y = builder.get_variables("x", "y")
    return x, y


@app.cell
def _(p, x, y):
    p(x).given(y)
    return


if __name__ == "__main__":
    app.run()
