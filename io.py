# mydatasciencebook/io.py

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.

    Example
    -------
    >>> df = load_csv("data/chapter2/LinearRegression1.csv")
    """
    path = Path(path)
    return pd.read_csv(path, **read_csv_kwargs)


def train_val_test_split(
    df: pd.DataFrame,
    target: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split a dataframe into train/validation/test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    y = df[target]
    X = df.drop(columns=[target])

    # First split off combined val+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state
    )
    # Split temp into val and test
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
