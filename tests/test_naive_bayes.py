"""Tests for tokenising, fitting and scoring."""

from __future__ import annotations

import math

import pytest

from naive_bayes.metrics import report
from naive_bayes.model import MultinomialNaiveBayes
from naive_bayes.preprocessing import clean, simple_tokenize

SPORT = ["goal", "match", "team", "goal"]
TECH = ["chip", "laptop", "screen", "chip"]


@pytest.fixture
def model():
    documents = [SPORT, SPORT, TECH, TECH]
    return MultinomialNaiveBayes(alpha=1.0).fit(documents, ["sport", "sport", "tech", "tech"])


def test_tokeniser_splits_persian_and_latin_alike():
    assert simple_tokenize("سلام دنیا") == ["سلام", "دنیا"]
    assert simple_tokenize("iOS 13 released") == ["iOS", "13", "released"]


def test_clean_drops_stop_words_digits_and_short_tokens():
    assert clean(["the", "chip", "a", "7", "screen"], stop_words={"the"}) == ["chip", "screen"]


def test_fit_records_classes_and_vocabulary(model):
    assert model.classes == ["sport", "tech"]
    assert "goal" in model.vocabulary and "chip" in model.vocabulary
    assert model.class_counts["sport"] == 2


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit([SPORT], ["sport", "tech"])


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().predict_one(["goal"])


def test_documents_are_assigned_to_the_class_they_came_from(model):
    assert model.predict_one(["goal", "match"]) == "sport"
    assert model.predict_one(["chip", "laptop"]) == "tech"


def test_equal_priors_when_classes_are_balanced(model):
    assert model.log_prior("sport") == pytest.approx(math.log(0.5))


def test_smoothing_keeps_unseen_tokens_from_zeroing_a_class():
    """Without a pseudo-count one absent word rules a class out entirely."""
    unsmoothed = MultinomialNaiveBayes(alpha=0.0).fit([SPORT, TECH], ["sport", "tech"])
    smoothed = MultinomialNaiveBayes(alpha=1.0).fit([SPORT, TECH], ["sport", "tech"])

    # "chip" never occurs in the sport class.
    assert unsmoothed.log_likelihood("chip", "sport") == -math.inf
    assert math.isfinite(smoothed.log_likelihood("chip", "sport"))


def test_unknown_tokens_are_ignored_rather_than_penalised(model):
    with_unknown = model.score(["goal", "zzzzz"])
    without = model.score(["goal"])
    assert with_unknown == without


def test_most_common_ranks_tokens_within_a_class(model):
    assert model.most_common("sport", n=1) == [("goal", 4)]


def test_report_is_perfect_when_predictions_match():
    result = report(["a", "b"], ["a", "b"])
    assert result.accuracy == 1.0
    assert result.macro == (1.0, 1.0, 1.0)


def test_report_computes_precision_and_recall_per_class():
    #                 a    a    b       predicted a, a, a
    result = report(["a", "a", "b"], ["a", "a", "a"])
    by_label = {c.label: c for c in result.per_class}
    assert by_label["a"].precision == pytest.approx(2 / 3)
    assert by_label["a"].recall == pytest.approx(1.0)
    assert by_label["b"].recall == pytest.approx(0.0)


def test_macro_and_weighted_diverge_when_classes_are_imbalanced():
    actual = ["a"] * 9 + ["b"]
    predicted = ["a"] * 10
    result = report(actual, predicted)
    assert result.macro[2] != pytest.approx(result.weighted[2])


def test_micro_average_equals_accuracy():
    result = report(["a", "b", "a"], ["a", "a", "a"])
    assert result.micro[0] == pytest.approx(result.accuracy)


def test_report_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        report([], [])
    with pytest.raises(ValueError):
        report(["a"], ["a", "b"])
