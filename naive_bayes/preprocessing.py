"""Turning Persian article text into tokens."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

# Persian and Arabic letters, plus Latin and digits for loanwords and model names.
TOKEN_PATTERN = re.compile(r"[\w؀-ۿ‌]+", re.UNICODE)
ZERO_WIDTH_NON_JOINER = "‌"


def simple_tokenize(text: str) -> list[str]:
    """Split on anything that is not a word character.

    Deliberately dependency-free, so the classifier can be exercised without a
    Persian NLP stack installed. ``hazm_tokenizer`` is the better choice when the
    dependency is available.
    """
    return TOKEN_PATTERN.findall(text or "")


def hazm_tokenizer() -> Callable[[str], list[str]]:
    """Tokenise and lemmatise with hazm, normalising the text first.

    Lemmatisation rather than stemming: Persian verbs carry their tense in
    affixes, and a stemmer truncates those to strings that are not words.
    """
    from hazm import Lemmatizer, Normalizer, word_tokenize

    normaliser = Normalizer()
    lemmatiser = Lemmatizer()

    def tokenize(text: str) -> list[str]:
        return [lemmatiser.lemmatize(t) for t in word_tokenize(normaliser.normalize(text or ""))]

    return tokenize


def load_stop_words(path: Path | str | None = None) -> set[str]:
    """Read stop words, one per line, ignoring blanks and comments."""
    if path is None:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def clean(
    tokens: Iterable[str], stop_words: set[str] | None = None, min_length: int = 2
) -> list[str]:
    """Drop stop words, joiner artefacts and tokens too short to carry meaning."""
    stop_words = stop_words or set()
    kept = []
    for token in tokens:
        token = token.replace(ZERO_WIDTH_NON_JOINER, "").strip()
        if len(token) < min_length or token in stop_words or token.isdigit():
            continue
        kept.append(token)
    return kept
