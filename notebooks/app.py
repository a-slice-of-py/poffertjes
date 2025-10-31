import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    pd.__version__


if __name__ == "__main__":
    app.run()
