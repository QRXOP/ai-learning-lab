"""Train the demo sentiment classifier and show predictions.

The built-in dataset has ten hand-written sentences. It demonstrates the API;
it says nothing about how well this approach works on real text.
"""

from ai_learning_lab.nlp.sentiment import SentimentClassifier


def main() -> None:
    model = SentimentClassifier.from_demo_dataset()
    samples = [
        "I love how useful and clear this is.",
        "This is frustrating and difficult to use.",
        "It is okay, nothing special.",
    ]
    labels = model.predict(samples)
    probabilities = model.predict_proba(samples)

    print("Classes:", ", ".join(model.classes))
    for text, label, probs in zip(samples, labels, probabilities, strict=True):
        formatted = ", ".join(f"{p:.2f}" for p in probs)
        print(f"{label:>8}  [{formatted}]  {text}")


if __name__ == "__main__":
    main()
