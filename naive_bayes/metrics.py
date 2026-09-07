"""Per-class and averaged classification metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Report:
    per_class: tuple[ClassScore, ...]
    accuracy: float

    @property
    def macro(self) -> tuple[float, float, float]:
        """Unweighted mean over classes, so a rare class counts as much as a common one."""
        n = len(self.per_class)
        return (
            sum(c.precision for c in self.per_class) / n,
            sum(c.recall for c in self.per_class) / n,
            sum(c.f1 for c in self.per_class) / n,
        )

    @property
    def weighted(self) -> tuple[float, float, float]:
        """Mean over classes weighted by support."""
        total = sum(c.support for c in self.per_class) or 1
        return (
            sum(c.precision * c.support for c in self.per_class) / total,
            sum(c.recall * c.support for c in self.per_class) / total,
            sum(c.f1 * c.support for c in self.per_class) / total,
        )

    @property
    def micro(self) -> tuple[float, float, float]:
        """Pooled over all decisions. For single-label classification these three
        collapse to accuracy, since every error is one false positive and one
        false negative at the same time."""
        return (self.accuracy, self.accuracy, self.accuracy)

    def format(self) -> str:
        width = max(len(c.label) for c in self.per_class) + 2
        lines = [f"{'class':<{width}} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}"]
        for c in self.per_class:
            lines.append(
                f"{c.label:<{width}} {c.precision:>10.3f} {c.recall:>8.3f} "
                f"{c.f1:>8.3f} {c.support:>8}"
            )
        macro, weighted = self.macro, self.weighted
        lines.append(f"{'macro':<{width}} {macro[0]:>10.3f} {macro[1]:>8.3f} {macro[2]:>8.3f}")
        lines.append(
            f"{'weighted':<{width}} {weighted[0]:>10.3f} {weighted[1]:>8.3f} {weighted[2]:>8.3f}"
        )
        lines.append(f"{'accuracy':<{width}} {self.accuracy:>10.3f}")
        return "\n".join(lines)


def _safe_divide(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def report(actual: list[str], predicted: list[str], labels: list[str] | None = None) -> Report:
    """Score predictions, one row per class plus the averages.

    Precision and recall answer different questions and neither is sufficient
    alone: a model can reach perfect recall by predicting one class for
    everything, and high precision by predicting it almost never. F1 is their
    harmonic mean, which stays low unless both are high.
    """
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} labels against {len(predicted)} predictions")
    if not actual:
        raise ValueError("nothing to score")

    labels = labels if labels is not None else sorted(set(actual) | set(predicted))
    scores = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores.append(
            ClassScore(label, precision, recall, f1, sum(a == label for a in actual))
        )

    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    return Report(per_class=tuple(scores), accuracy=accuracy)
