"""Deterministic educational classification examples using scikit-learn."""

from __future__ import annotations

from typing import Literal

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = Literal["logistic_regression", "random_forest"]

RANDOM_STATE = 42


def _build_model(model_name: ModelName) -> Pipeline:
    """Construct an unfitted, seeded classifier pipeline."""
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model_name: {model_name!r}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def train_iris_classifier(model_name: ModelName = "logistic_regression") -> Pipeline:
    """Fit a classifier on the complete built-in Iris dataset and return it.

    Use :func:`evaluate_iris_classifier` to assess the model on a held-out split.
    """
    model = _build_model(model_name)
    iris = load_iris()
    model.fit(iris.data, iris.target)
    return model


def evaluate_iris_classifier(
    model_name: ModelName = "logistic_regression",
) -> dict[str, float | int | str]:
    """Evaluate a seeded model on a deterministic stratified holdout split."""
    model = _build_model(model_name)
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "model": model_name,
        "dataset": "iris",
        "random_state": RANDOM_STATE,
        "test_size": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }
