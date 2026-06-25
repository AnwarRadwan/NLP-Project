"""Smoke test / demo for the TF-IDF search engine.

Builds (or loads) the TF-IDF artifacts, prints the vocabulary size and matrix
shape, then runs:
    1. bilingual free-text queries (intent preference + de-duplication active),
    2. category-filtered queries (hard filter via ``category=...``).

Usage:
    python src/retrieval/test_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.search_engine import (  # noqa: E402
    CAT_COURSE,
    CAT_DECISIONS,
    SearchEngine,
)

# Free-text queries (Arabic + English).
TEST_QUERIES: dict[str, list[str]] = {
    "Arabic": ["رسوم الجامعة", "اضراب", "دكتور ممتاز", "التسجيل"],
    "English": ["tuition fees", "strike", "great professor", "registration"],
}

# Category-filtered queries: (query, category).
FILTERED_QUERIES: list[tuple[str, str]] = [
    ("دكتور ممتاز", CAT_COURSE),
    ("great professor", CAT_COURSE),
    ("اضراب", CAT_DECISIONS),
    ("strike", CAT_DECISIONS),
]

TOP_K = 5


def _print_results(header: str, results: list[dict]) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    if not results:
        print("  (no matching documents)")
        return
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. score={r['similarity_score']:.4f} "
              f"[{r['category']} | {r['sentiment']}]")
        print(f"     {r['text']}")


def main() -> None:
    # Ensure Arabic prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    engine = SearchEngine.from_default()

    print("=" * 64)
    print("TF-IDF Search Engine")
    print("=" * 64)
    print(f"Vocabulary size    : {engine.vocab_size}")
    print(f"TF-IDF matrix shape: {engine.matrix.shape}")

    # 1) Free-text queries (intent preference + de-duplication).
    for group, queries in TEST_QUERIES.items():
        print("\n" + "#" * 64)
        print(f"# {group} queries (intent-aware, de-duplicated)")
        print("#" * 64)
        for query in queries:
            _print_results(f"Query: {query!r}", engine.search(query, top_k=TOP_K))

    # 2) Category-filtered queries.
    print("\n" + "#" * 64)
    print("# Category-filtered queries")
    print("#" * 64)
    for query, category in FILTERED_QUERIES:
        _print_results(
            f"Query: {query!r}  (category={category!r})",
            engine.search(query, top_k=TOP_K, category=category),
        )


if __name__ == "__main__":
    main()
