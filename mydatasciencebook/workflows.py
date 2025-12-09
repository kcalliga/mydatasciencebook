# mydatasciencebook/workflows.py

"""
High-level workflows that mirror the 'Samples Datasets and Workflows' page:

A. What type of problem?
B. Loading and viewing data.
C. Visualization and modeling.

Each function here is meant to be called from a notebook or example in the Wiki.
"""

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import io as io_utils
from . import eda as eda_utils
from . import preprocess as prep_utils
from . import models as model_utils
from . import metrics as metric_utils


# ---------- Regression workflow ----------

def simple_regression_workflow(
    path: str | Path,
    target: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    End-to-end linear regression workflow:

    1. Load CSV.
    2. Train/val/test split.
    3. Build default preprocessor.
    4. LinearRegression model in a Pipeline.
    5. Fit on train, evaluate on validation with RMSE / MAE / R2.

    Returns
    -------
    pipeline, metrics_dict
    """
    df = io_utils.load_csv(path)
    X_train, X_val, X_test, y_train, y_val, y_test = io_utils.train_val_test_split(
        df, target=target, val_size=val_size, test_size=test_size, random_state=random_state
    )

    pre = prep_utils.make_default_preprocessor(X_train)
    model = model_utils.make_linear_regression()

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    metrics = metric_utils.regression_report(y_val, y_val_pred)
    return pipe, metrics


# ---------- Classification workflow ----------

def simple_classification_workflow(
    path: str | Path,
    target: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "logistic",
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    End-to-end classification workflow.

    model_type: 'logistic', 'tree', or 'forest'
    """
    df = io_utils.load_csv(path)
    X_train, X_val, X_test, y_train, y_val, y_test = io_utils.train_val_test_split(
        df, target=target, val_size=val_size, test_size=test_size, random_state=random_state
    )

    pre = prep_utils.make_default_preprocessor(X_train)

    if model_type == "logistic":
        model = model_utils.make_logistic_regression()
    elif model_type == "tree":
        model = model_utils.make_decision_tree_classifier()
    elif model_type == "forest":
        model = model_utils.make_random_forest_classifier()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)

    y_val_pred = pipe.predict(X_val)
    metrics = metric_utils.classification_report_basic(y_val, y_val_pred)
    return pipe, metrics


# ---------- KMeans clustering workflow ----------

from sklearn.metrics import silhouette_score
from .models import make_kmeans


def simple_kmeans_workflow(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_clusters: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple KMeans clustering on numeric columns.

    Returns
    -------
    labels, summary
    """
    if numeric_cols is None:
        numeric_cols = list(df.select_dtypes("number").columns)
    if not numeric_cols:
        raise ValueError("No numeric columns provided for clustering.")

    subset = df[numeric_cols].copy()
    subset = subset.dropna()  # simple handling here

    km = make_kmeans(n_clusters=n_clusters)
    labels = km.fit_predict(subset)

    # silhouette needs > 1 cluster and enough samples
    if len(set(labels)) > 1 and len(subset) > len(set(labels)):
        sil = float(silhouette_score(subset, labels))
    else:
        sil = float("nan")

    summary = {
        "n_samples": len(subset),
        "n_clusters": int(len(set(labels))),
        "silhouette_score": sil,
    }
    return labels, summary
