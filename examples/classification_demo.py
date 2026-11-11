"""Compare two scikit-learn classifiers on a fixed Iris holdout split."""

from ai_learning_lab.classification.sklearn_models import evaluate_iris_classifier


def main() -> None:
    for model_name in ("logistic_regression", "random_forest"):
        metrics = evaluate_iris_classifier(model_name=model_name)
        print(
            f"{metrics['model']:<20} accuracy={metrics['accuracy']:.3f} "
            f"on {metrics['test_size']} held-out samples "
            f"(random_state={metrics['random_state']})"
        )


if __name__ == "__main__":
    main()
