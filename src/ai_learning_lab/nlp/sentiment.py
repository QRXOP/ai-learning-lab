"""A small educational sentiment classifier using scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_DEMO_TEXTS = [
    "I love this product and it works perfectly.",
    "This is amazing, fast, and very useful.",
    "The feature is excellent and the quality is outstanding.",
    "I dislike this service because it is slow and frustrating.",
    "The app is terrible and not worth the time.",
    "I am unhappy with the response time and reliability.",
    "The update is okay and not bad at all.",
    "The product is fine, but I expected more.",
    "This is a good experience with clear instructions.",
    "The software is poor and difficult to use.",
]
_DEMO_LABELS = [
    "positive", "positive", "positive", "negative", "negative",
    "negative", "neutral", "neutral", "positive", "negative",
]


def build_demo_sentiment_dataset() -> tuple[list[str], list[str]]:
    """Return copies of a tiny hand-authored dataset for demonstrations only."""
    return _DEMO_TEXTS.copy(), _DEMO_LABELS.copy()


class SentimentClassifier:
    """TF-IDF and logistic-regression text classifier for educational use."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self._is_fitted = False

    @property
    def classes(self) -> list[str]:
        """Class labels in the column order used by :meth:`predict_proba`."""
        self._require_fitted()
        return [str(label) for label in self.pipeline.classes_]

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        if isinstance(texts, (str, bytes)) or len(texts) == 0:
            raise ValueError("texts must be a non-empty sequence of strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("each text must be a non-empty string")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> SentimentClassifier:
        """Fit on parallel non-empty text and label sequences."""
        self._validate_texts(texts)
        if isinstance(labels, (str, bytes)) or len(texts) != len(labels):
            raise ValueError("texts and labels must be sequences of equal length")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("each label must be a non-empty string")
        if len(set(labels)) < 2:
            raise ValueError("at least two distinct labels are required")
        self.pipeline.fit(list(texts), list(labels))
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("fit the classifier before using it")

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict labels for a non-empty sequence of strings."""
        self._require_fitted()
        self._validate_texts(texts)
        return [str(label) for label in self.pipeline.predict(list(texts))]

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        """Return per-class probability estimates for each input text.

        Columns follow the order of :attr:`classes`.
        """
        self._require_fitted()
        self._validate_texts(texts)
        return self.pipeline.predict_proba(list(texts)).tolist()

    @classmethod
    def from_demo_dataset(cls) -> SentimentClassifier:
        """Return a fitted classifier using the tiny built-in teaching dataset."""
        texts, labels = build_demo_sentiment_dataset()
        return cls().fit(texts, labels)
