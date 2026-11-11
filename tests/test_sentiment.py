import pytest

from ai_learning_lab.nlp.sentiment import SentimentClassifier, build_demo_sentiment_dataset


@pytest.fixture(scope="module")
def demo_model() -> SentimentClassifier:
    return SentimentClassifier.from_demo_dataset()


def test_demo_dataset_is_parallel_and_copied():
    texts, labels = build_demo_sentiment_dataset()
    assert len(texts) == len(labels) == 10
    texts.append("mutated")
    assert len(build_demo_sentiment_dataset()[0]) == 10


def test_predict_returns_known_labels(demo_model):
    predictions = demo_model.predict(["I love this.", "This is terrible."])
    assert len(predictions) == 2
    assert set(predictions) <= {"positive", "negative", "neutral"}
    assert all(isinstance(label, str) for label in predictions)


def test_predict_proba_rows_sum_to_one(demo_model):
    probabilities = demo_model.predict_proba(["Great work.", "Poor quality."])
    assert len(probabilities) == 2
    for row in probabilities:
        assert len(row) == len(demo_model.classes)
        assert sum(row) == pytest.approx(1.0)


def test_training_examples_are_recovered(demo_model):
    # A sanity check that the pipeline learns its own tiny training set,
    # not a measure of real-world accuracy.
    texts, labels = build_demo_sentiment_dataset()
    assert demo_model.predict(texts) == labels


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        SentimentClassifier().predict(["hello"])
    with pytest.raises(RuntimeError):
        SentimentClassifier().predict_proba(["hello"])


@pytest.mark.parametrize("bad", ["a plain string", [], ["ok", ""], ["ok", "   "], ["ok", 3]])
def test_invalid_texts_rejected(demo_model, bad):
    with pytest.raises(ValueError):
        demo_model.predict(bad)


def test_fit_validates_labels():
    model = SentimentClassifier()
    with pytest.raises(ValueError):
        model.fit(["a", "b"], ["positive"])
    with pytest.raises(ValueError):
        model.fit(["a", "b"], ["positive", "positive"])
    with pytest.raises(ValueError):
        model.fit(["a", "b"], ["positive", ""])
    with pytest.raises(ValueError):
        model.fit(["a", "b"], "pn")


def test_fit_returns_self():
    model = SentimentClassifier()
    assert model.fit(["good thing", "bad thing"], ["pos", "neg"]) is model
