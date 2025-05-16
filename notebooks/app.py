import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    from poffertjes import VariableBuilder, p

    return VariableBuilder, np, pd


@app.cell
def _(VariableBuilder, np, pd):
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

    vb = VariableBuilder.from_data(df)

    x, y, z, u = vb.get_variables()
    x0, y0, z0, u0 = df.iloc[0]
    return (df,)


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
