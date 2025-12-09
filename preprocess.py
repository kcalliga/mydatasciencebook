# mydatasciencebook/preprocess.py

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_default_preprocessor(
    df: pd.DataFrame,
    numeric: Sequence[str] | None = None,
    categoric: Sequence[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes numeric with median + scales them
      - imputes categorical with most_frequent + one-hot encodes.

    If numeric/categoric are None, they are inferred from dtypes.
    """
    if numeric is None:
        numeric = list(df.select_dtypes(include=["number"]).columns)
    if categoric is None:
        categoric = list(df.select_dtypes(include=["object", "category"]).columns)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categoric),
        ],
        remainder="drop",
    )
    return pre
