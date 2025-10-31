import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from poffertjes.variable import Variable, VariableBuilder
    return VariableBuilder, np, pd


@app.cell
def _(np, pd):
    N_SAMPLES = 100
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
    builder.get_variables("x")
    return


if __name__ == "__main__":
    app.run()
