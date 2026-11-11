import pytest
from sklearn.datasets import load_iris

from ai_learning_lab.classification.sklearn_models import (
    evaluate_iris_classifier,
    train_iris_classifier,
)

MODELS = ["logistic_regression", "random_forest"]


@pytest.mark.parametrize("model_name", MODELS)
def test_train_returns_fitted_pipeline(model_name):
    model = train_iris_classifier(model_name=model_name)
    iris = load_iris()
    predictions = model.predict(iris.data[:5])
    assert len(predictions) == 5
    assert set(predictions) <= {0, 1, 2}


@pytest.mark.parametrize("model_name", MODELS)
def test_evaluate_reports_metadata(model_name):
    metrics = evaluate_iris_classifier(model_name=model_name)
    assert metrics["model"] == model_name
    assert metrics["dataset"] == "iris"
    assert metrics["random_state"] == 42
    assert metrics["test_size"] == 38
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.parametrize("model_name", MODELS)
def test_evaluate_is_deterministic(model_name):
    assert evaluate_iris_classifier(model_name) == evaluate_iris_classifier(model_name)


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unsupported model_name"):
        train_iris_classifier(model_name="svm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported model_name"):
        evaluate_iris_classifier(model_name="svm")  # type: ignore[arg-type]
