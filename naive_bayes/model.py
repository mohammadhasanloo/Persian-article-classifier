"""Multinomial naive Bayes over token counts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class MultinomialNaiveBayes:
    """A bag-of-words classifier with additive (Laplace) smoothing.

    ``alpha`` is the pseudo-count added to every token in every class. Without it
    a single token unseen in a class during training drives that class's
    likelihood to zero, and because the scores are summed in log space one absent
    word is enough to rule the class out entirely no matter what the rest of the
    document says.
    """

    alpha: float = 1.0
    class_counts: Counter = field(default_factory=Counter)
    token_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    vocabulary: set[str] = field(default_factory=set)

    @property
    def classes(self) -> list[str]:
        return sorted(self.class_counts)

    def fit(self, documents: list[list[str]], labels: list[str]) -> "MultinomialNaiveBayes":
        if len(documents) != len(labels):
            raise ValueError(f"{len(documents)} documents against {len(labels)} labels")
        for tokens, label in zip(documents, labels):
            self.class_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.vocabulary.update(tokens)
        return self

    def log_prior(self, label: str) -> float:
        return math.log(self.class_counts[label] / sum(self.class_counts.values()))

    def log_likelihood(self, token: str, label: str) -> float:
        """Smoothed probability of a token given a class, in log space.

        With ``alpha`` at zero a token absent from the class has probability zero,
        which is negative infinity in log space rather than an error.
        """
        counts = self.token_counts[label]
        numerator = counts.get(token, 0) + self.alpha
        if numerator == 0:
            return -math.inf
        denominator = sum(counts.values()) + self.alpha * len(self.vocabulary)
        return math.log(numerator / denominator)

    def score(self, tokens: list[str]) -> dict[str, float]:
        """Log posterior for every class, up to the shared evidence term.

        The evidence is the same for every class of a given document, so it
        cannot change which class scores highest and is not computed.
        """
        if not self.class_counts:
            raise ValueError("the model has not been fitted")
        return {
            label: self.log_prior(label)
            + sum(self.log_likelihood(t, label) for t in tokens if t in self.vocabulary)
            for label in self.classes
        }

    def predict_one(self, tokens: list[str]) -> str:
        scores = self.score(tokens)
        return max(scores, key=scores.get)

    def predict(self, documents: list[list[str]]) -> list[str]:
        return [self.predict_one(tokens) for tokens in documents]

    def most_common(self, label: str, n: int = 10) -> list[tuple[str, int]]:
        """The tokens that occur most often in one class."""
        return self.token_counts[label].most_common(n)
