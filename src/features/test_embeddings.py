"""Train + demo the Skip-Gram word embeddings.

Steps:
    1. Train word vectors on the corpus (or load if already trained).
    2. Print vocabulary size and embedding dimension.
    3. Show nearest neighbours for seed words (Arabic + English).
    4. Show a sentence-embedding shape and a related-vs-unrelated cosine check.

Usage:
    python src/features/test_embeddings.py
    python src/features/test_embeddings.py --reuse   # load instead of retrain
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.embeddings import VECTORS_FILE, WordEmbeddings, train_word2vec  # noqa: E402

# In-vocabulary seed words (the corpus keeps the Arabic article, e.g. الاضراب,
# and uses the plural "professors").
SEED_WORDS = ["دكتور", "الاضراب", "التسجيل", "professors", "tuition", "registration"]

SENTENCE_PAIRS = [
    # (text A, text B, expected relation)
    ("الدكتور بشرح المادة منيح", "المحاضرات حلوة والدكتور متعاون", "related"),
    ("tuition fees are too high", "students declared a strike", "less related"),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    reuse = "--reuse" in sys.argv

    print("=" * 60)
    print("Word Embeddings (Skip-Gram Word2Vec)")
    print("=" * 60)

    if reuse and VECTORS_FILE.exists():
        emb = WordEmbeddings.load()
        print(f"Loaded existing embeddings from {VECTORS_FILE}")
    else:
        emb = train_word2vec(verbose=True)

    print(f"\nVocabulary size : {len(emb.itos)}")
    print(f"Embedding dim   : {emb.dim}")

    # 3) Nearest neighbours for seed words.
    print("\n--- Nearest neighbours ---")
    for word in SEED_WORDS:
        if word in emb:
            neighbours = ", ".join(f"{w} ({s:.2f})" for w, s in emb.most_similar(word, 5))
            print(f"{word:>14} -> {neighbours}")
        else:
            print(f"{word:>14} -> (out of vocabulary)")

    # 4) Sentence embeddings + cosine sanity check.
    print("\n--- Sentence embeddings ---")
    sample_vec = emb.embed_text("الدكتور رائع وشرح المادة")
    print(f"Sample sentence embedding shape: {sample_vec.shape}")
    for a, b, relation in SENTENCE_PAIRS:
        va, vb = emb.embed_text(a), emb.embed_text(b)
        print(f"cosine({relation:12}) = {WordEmbeddings.cosine(va, vb):.4f}")
        print(f"    A: {a}")
        print(f"    B: {b}")

    print("\nDONE")


if __name__ == "__main__":
    main()
