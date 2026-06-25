"""Text cleaning and normalization for Arabic and English.

Public entry point:
    clean_text(text, lang)  ->  cleaned string  (lang is "ar" or "en")

Arabic cleaning:
    - strip URLs and emojis
    - remove diacritics (tashkeel)
    - normalize alef variants (أ إ آ -> ا) and (ى -> ي)
    - remove tatweel (ـ)
    - keep Arabic letters only (drop Latin, digits, punctuation, symbols)

English cleaning:
    - lowercase
    - strip URLs and emojis
    - remove punctuation / symbols (keep letters & digits)
    - collapse whitespace
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Regex building blocks (unicode escapes for reliability)
# --------------------------------------------------------------------------- #
# Arabic diacritics (tashkeel) and related combining marks.
_ARABIC_DIACRITICS = re.compile(
    "["
    "ؐ-ؚ"   # Arabic signs
    "ً-ٟ"   # tashkeel: fathatan..  + extras
    "ٰ"          # superscript alef
    "ۖ-ۭ"   # small high/low marks
    "]"
)
_TATWEEL = "ـ"            # ـ

# Arabic letter range used to keep-only Arabic.
_NON_ARABIC = re.compile(r"[^ء-ي\s]")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

# Broad emoji / pictograph / dingbat ranges.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0001F300-\U0001FAFF"   # symbols, pictographs, supplemental
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U00002B00-\U00002BFF"   # arrows & misc
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "]+",
    flags=re.UNICODE,
)

# Keep only English letters/digits + whitespace.
_NON_ENGLISH = re.compile(r"[^a-z0-9\s]")
_MULTISPACE = re.compile(r"\s+")

# Alef variants -> ا  ;  alef maksura -> ي
_ALEF_VARIANTS = re.compile("[أإآ]")   # أ إ آ
_ALEF_MAKSURA = "ى"                              # ى
_PLAIN_ALEF = "ا"                                # ا
_YEH = "ي"                                       # ي


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def _squeeze(text: str) -> str:
    return _MULTISPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Arabic
# --------------------------------------------------------------------------- #
def remove_diacritics(text: str) -> str:
    """Remove Arabic tashkeel / diacritical marks."""
    return _ARABIC_DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove the Arabic tatweel (kashida) elongation character."""
    return text.replace(_TATWEEL, "")


def normalize_arabic(text: str) -> str:
    """Normalize alef variants (أ إ آ -> ا) and alef maksura (ى -> ي)."""
    text = _ALEF_VARIANTS.sub(_PLAIN_ALEF, text)
    text = text.replace(_ALEF_MAKSURA, _YEH)
    return text


def clean_arabic(text: str) -> str:
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_arabic(text)
    text = _NON_ARABIC.sub(" ", text)   # keep Arabic letters only
    return _squeeze(text)


# --------------------------------------------------------------------------- #
# English
# --------------------------------------------------------------------------- #
def clean_english(text: str) -> str:
    text = text.lower()
    text = remove_urls(text)
    text = remove_emojis(text)
    text = _NON_ENGLISH.sub(" ", text)  # drop punctuation/symbols
    return _squeeze(text)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def clean_text(text: str, lang: str) -> str:
    """Clean ``text`` according to ``lang`` ("ar" or "en")."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return clean_arabic(text) if lang == "ar" else clean_english(text)
