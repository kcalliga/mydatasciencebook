# mydatasciencebook/metrics.py

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Compute a small set of regression metrics:
      - RMSE
      - MAE
      - R2
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_report_basic(y_true, y_pred) -> Dict[str, float]:
    """
    Compute basic classification metrics:
      - accuracy
      - precision (macro)
      - recall (macro)
      - f1 (macro)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
