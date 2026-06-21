"""N-gram generation and counting.

Works on token lists produced by ``src.preprocessing.tokenization``.

Public entry points:
    ngrams(tokens, n)               -> list[str]   (space-joined n-grams)
    unigrams / bigrams / trigrams(tokens)
    count_ngrams(list_of_token_lists, n) -> collections.Counter
    top_ngrams(counter, k)          -> list[(ngram, count)]
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def ngrams(tokens: list[str], n: int) -> list[str]:
    """Return space-joined n-grams from ``tokens`` (empty if too few tokens)."""
    if n < 1 or len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def unigrams(tokens: list[str]) -> list[str]:
    return ngrams(tokens, 1)


def bigrams(tokens: list[str]) -> list[str]:
    return ngrams(tokens, 2)


def trigrams(tokens: list[str]) -> list[str]:
    return ngrams(tokens, 3)


def count_ngrams(token_lists: Iterable[list[str]], n: int) -> Counter:
    """Count n-grams across a corpus (iterable of token lists)."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(ngrams(tokens, n))
    return counter


def top_ngrams(counter: Counter, k: int = 20) -> list[tuple[str, int]]:
    """Return the ``k`` most common n-grams."""
    return counter.most_common(k)
