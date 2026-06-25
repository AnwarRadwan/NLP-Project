"""Run the preprocessing pipeline over the raw BZU dataset.

Reads:   data/raw/bzu_dataset.csv
Writes:  data/processed/bzu_dataset_clean.csv

Adds columns:
    clean_text      cleaned text (language-aware)
    tokens          space-joined tokens after stopword removal
    unigram_count   number of unigrams in the row
    bigram_count    number of bigrams in the row
    trigram_count   number of trigrams in the row

Then prints: dataset shape, sample original/clean/tokens, and the top-20
unigrams, bigrams, and trigrams across the corpus.

Usage:
    python src/preprocessing/run_preprocessing.py
    python src/preprocessing/run_preprocessing.py --in data/raw/bzu_dataset.csv \
        --out data/processed/bzu_dataset_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.preprocessing.cleaning import clean_text  # noqa: E402
from src.preprocessing.ngrams import count_ngrams, top_ngrams  # noqa: E402
from src.preprocessing.tokenization import preprocess_tokens  # noqa: E402

DEFAULT_IN = ROOT / "data" / "raw" / "bzu_dataset.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    """Add preprocessing columns; return (df, list-of-token-lists)."""
    clean_texts: list[str] = []
    token_lists: list[list[str]] = []

    for text, lang in zip(df["text"].astype(str), df["language"].astype(str)):
        cleaned = clean_text(text, lang)
        tokens = preprocess_tokens(cleaned, lang)
        clean_texts.append(cleaned)
        token_lists.append(tokens)

    df = df.copy()
    df["clean_text"] = clean_texts
    df["tokens"] = [" ".join(toks) for toks in token_lists]
    df["unigram_count"] = [len(toks) for toks in token_lists]
    df["bigram_count"] = [max(0, len(toks) - 1) for toks in token_lists]
    df["trigram_count"] = [max(0, len(toks) - 2) for toks in token_lists]
    return df, token_lists


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_top(title: str, pairs: list[tuple[str, int]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rank, (gram, count) in enumerate(pairs, 1):
        print(f"{rank:>2}. {gram}  ({count})")


def report(df: pd.DataFrame, token_lists: list[list[str]]) -> None:
    print(f"Dataset shape: {df.shape}")

    # Pick one Arabic and one English example for illustration.
    def first_index(lang: str) -> int | None:
        idx = df.index[df["language"] == lang]
        return int(idx[0]) if len(idx) else None

    for lang, label in (("en", "English"), ("ar", "Arabic")):
        i = first_index(lang)
        if i is None:
            continue
        print(f"\n=== Sample ({label}) ===")
        print(f"Original : {df.at[i, 'text']}")
        print(f"Clean    : {df.at[i, 'clean_text']}")
        print(f"Tokens   : {df.at[i, 'tokens']}")

    _print_top("Top 20 unigrams", top_ngrams(count_ngrams(token_lists, 1), 20))
    _print_top("Top 20 bigrams", top_ngrams(count_ngrams(token_lists, 2), 20))
    _print_top("Top 20 trigrams", top_ngrams(count_ngrams(token_lists, 3), 20))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess the BZU dataset.")
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN,
                   help=f"input CSV (default {DEFAULT_IN})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output CSV (default {DEFAULT_OUT})")
    return p.parse_args()


def main() -> None:
    # Ensure Arabic prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    if not args.inp.exists():
        raise SystemExit(f"Input not found: {args.inp}\n"
                         f"Generate it first: python src/data/generate_dataset.py")

    df = pd.read_csv(args.inp, encoding="utf-8-sig")
    df, token_lists = preprocess_dataframe(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows -> {args.out}\n")

    report(df, token_lists)


if __name__ == "__main__":
    main()
