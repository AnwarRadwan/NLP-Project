"""Tokenization and stopword removal for Arabic and English.

Designed to run on text that has already been cleaned by
``src.preprocessing.cleaning.clean_text`` (lowercased English, Arabic letters
only), but the tokenizers are robust to raw text too.

Public entry points:
    tokenize(text, lang)            -> list[str]
    remove_stopwords(tokens, lang)  -> list[str]
    preprocess_tokens(text, lang)   -> tokenize + remove_stopwords
"""

from __future__ import annotations

import re

from src.preprocessing.cleaning import normalize_arabic

# English word pattern (letters/digits).
_WORD_EN = re.compile(r"[a-z0-9]+")
# Minimum token length to keep (drops stray single characters).
_MIN_LEN = 2


# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #
EN_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "at", "by",
    "from", "as", "so", "not", "no", "do", "does", "did", "has", "have", "had",
    "will", "would", "can", "could", "should", "just", "about", "too", "very",
    "really", "than", "then", "there", "here", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "any", "some", "out", "up", "down",
    "over", "again", "more", "most", "im", "ive", "dont", "doesnt", "youre",
}

# Basic Arabic stopwords (MSA + common Palestinian-dialect function words).
# Normalized with the same rules as cleaning (e.g. ى -> ي, أ/إ/آ -> ا) so the
# entries actually match the cleaned tokens.
_AR_STOPWORDS_RAW: set[str] = {
    "في", "من", "على", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "كان", "كانت", "يكون", "انا", "انت", "انتي",
    "هو", "هي", "نحن", "هم", "هن", "ما", "لا", "لم", "لن", "قد", "كل",
    "بعض", "عند", "عندما", "كما", "لكن", "او", "ثم", "يا", "اي", "كيف",
    "وين", "ليش", "هيك", "هاد", "هاي", "هسا", "اللي", "عشان", "علشان",
    "بس", "كمان", "يعني", "انه", "انها", "اله", "الها", "هلا", "صار",
    "عم", "بدي", "بدك", "بدنا", "احنا", "انتو", "هدول", "مين",
    "شو", "قديش", "كتير", "والله", "بصراحة", "عنجد", "اشي", "زي",
}
AR_STOPWORDS: set[str] = {normalize_arabic(w) for w in _AR_STOPWORDS_RAW}


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def tokenize_english(text: str) -> list[str]:
    """English tokenization: extract alphanumeric word tokens."""
    return _WORD_EN.findall(text.lower())


def tokenize_arabic(text: str) -> list[str]:
    """Arabic tokenization: simple whitespace split."""
    return [tok for tok in text.split() if tok]


def tokenize(text: str, lang: str) -> list[str]:
    """Tokenize ``text`` according to ``lang`` ("ar" or "en")."""
    if not isinstance(text, str) or not text.strip():
        return []
    return tokenize_arabic(text) if lang == "ar" else tokenize_english(text)


# --------------------------------------------------------------------------- #
# Stopword removal
# --------------------------------------------------------------------------- #
def remove_stopwords(tokens: list[str], lang: str) -> list[str]:
    """Drop stopwords and very short tokens for the given language."""
    stop = AR_STOPWORDS if lang == "ar" else EN_STOPWORDS
    return [tok for tok in tokens if len(tok) >= _MIN_LEN and tok not in stop]


def preprocess_tokens(text: str, lang: str) -> list[str]:
    """Convenience: tokenize then remove stopwords."""
    return remove_stopwords(tokenize(text, lang), lang)
