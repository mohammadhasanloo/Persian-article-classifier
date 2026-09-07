"""Command line entry point: ``python -m naive_bayes.cli ...``"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from naive_bayes.metrics import report
from naive_bayes.model import MultinomialNaiveBayes
from naive_bayes.preprocessing import clean, load_stop_words, simple_tokenize

CONTENT_COLUMN = "content"
LABEL_COLUMN = "label"


def read_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Read a two-column CSV of article text and category."""
    texts, labels = [], []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get(CONTENT_COLUMN) or "").strip()
            label = (row.get(LABEL_COLUMN) or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise ValueError(f"no usable rows in {path}")
    return texts, labels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--stop-words", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="additive smoothing pseudo-count; 0 disables it")
    parser.add_argument("--hazm", action="store_true",
                        help="tokenise and lemmatise with hazm instead of the built-in splitter")
    parser.add_argument("--top-tokens", type=int, default=0,
                        help="print the N most frequent tokens per class")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.hazm:
        from naive_bayes.preprocessing import hazm_tokenizer

        tokenize = hazm_tokenizer()
    else:
        tokenize = simple_tokenize

    stop_words = load_stop_words(args.stop_words)
    prepare = lambda text: clean(tokenize(text), stop_words)  # noqa: E731

    train_texts, train_labels = read_corpus(args.train)
    test_texts, test_labels = read_corpus(args.test)

    model = MultinomialNaiveBayes(alpha=args.alpha)
    model.fit([prepare(t) for t in train_texts], train_labels)
    predicted = model.predict([prepare(t) for t in test_texts])

    print(f"alpha = {args.alpha}\n")
    print(report(test_labels, predicted, labels=model.classes).format())

    if args.top_tokens:
        print()
        for label in model.classes:
            tokens = ", ".join(t for t, _ in model.most_common(label, args.top_tokens))
            print(f"{label}: {tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
