# mydatasciencebook/tooling.py

"""
Extra utilities that correspond to the 'Toolbox' page on the website.
"""

from typing import Iterable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def run_sfs_regression(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 20,
    direction: str = "forward",
    scoring: str = "r2",
) -> Dict[str, object]:
    """
    Run Sequential Feature Selection for a regression model.

    Returns a dictionary with:
      - selected_features: list of column names
      - support_mask: boolean mask
      - estimator: fitted SFS object
    """
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=max_features,
        direction=direction,
        scoring=scoring,
        n_jobs=-1,
    )
    sfs.fit(X, y)
    support_mask = sfs.get_support()
    selected = list(X.columns[support_mask])
    return {
        "selected_features": selected,
        "support_mask": support_mask,
        "sfs": sfs,
    }


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: Iterable[str],
    class_names: Iterable[str] | None = None,
    max_depth: int | None = None,
    figsize=(20, 10),
    fontsize: int = 10,
) -> None:
    """
    Visualize a fitted sklearn DecisionTreeClassifier.

    Example
    -------
    >>> plot_decision_tree(dtree, feature_names=X.columns, class_names=dtree.classes_)
    """
    plt.figure(figsize=figsize)
    tree.plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=list(class_names) if class_names is not None else None,
        filled=True,
        fontsize=fontsize,
        max_depth=max_depth,
    )
    plt.show()


def kmeans_elbow_and_silhouette(
    df: pd.DataFrame,
    numeric_cols: list[str],
    k_range=range(2, 10),
) -> pd.DataFrame:
    """
    Run KMeans for each k in k_range, compute inertia and silhouette, and plot them.
    """
    subset = df[numeric_cols].copy()
    subset = subset.dropna()

    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(subset_scaled)
        inertia = km.inertia_
        if len(set(labels)) > 1 and len(subset_scaled) > len(set(labels)):
            sil = silhouette_score(subset_scaled, labels)
        else:
            sil = np.nan
        results.append({"k": k, "inertia": inertia, "silhouette": sil})

    res_df = pd.DataFrame(results)

    fig, ax1 = plt.subplots()
    ax1.plot(res_df["k"], res_df["inertia"], marker="o", label="Inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(res_df["k"], res_df["silhouette"], marker="s", color="g", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    plt.title("KMeans: inertia & silhouette vs k")
    fig.tight_layout()
    plt.show()

    return res_df
