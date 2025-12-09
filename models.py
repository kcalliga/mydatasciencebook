# mydatasciencebook/models.py

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def make_linear_regression() -> LinearRegression:
    """Default linear regression model."""
    return LinearRegression()


def make_logistic_regression(max_iter: int = 1000) -> LogisticRegression:
    """Default logistic regression for binary/multiclass classification."""
    return LogisticRegression(max_iter=max_iter)


def make_decision_tree_classifier(max_depth=None, random_state: int = 42) -> DecisionTreeClassifier:
    """Default decision tree classifier."""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def make_random_forest_classifier(
    n_estimators: int = 200,
    max_depth=None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Default random forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def make_kmeans(n_clusters: int = 3, random_state: int = 42) -> KMeans:
    """Default KMeans clustering."""
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
