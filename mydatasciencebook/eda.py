# mydatasciencebook/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def basic_summary(df: pd.DataFrame) -> None:
    """
    Print/display a quick summary of the dataframe:
      - head
      - shape
      - dtypes
      - missing values
      - describe()
    """
    display(df.head())
    print("\nShape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDescribe (include='all'):")
    display(df.describe(include="all"))


def correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> None:
    """
    Show a correlation heatmap for numeric columns, if there aren't too many.

    Parameters
    ----------
    max_cols : maximum number of numeric columns allowed for plotting.
    """
    num_df = df.select_dtypes("number")
    if num_df.shape[1] == 0:
        print("No numeric columns; skipping correlation heatmap.")
        return
    if num_df.shape[1] > max_cols:
        print(
            f"Numeric columns: {num_df.shape[1]} > max_cols={max_cols}. "
            "Skipping heatmap for readability."
        )
        return

    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap")
    plt.show()
