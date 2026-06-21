"""Word & sentence embeddings for the BZU corpus (Skip-Gram Word2Vec).

A self-contained embeddings layer that is independent of the TF-IDF / retrieval
/ classifier code. It trains classical **Skip-Gram with Negative Sampling**
word vectors (implemented in PyTorch) on the already-cleaned ``tokens`` column
of ``data/processed/bzu_dataset_clean.csv`` and provides:

    * word vectors + nearest-neighbour lookup (``most_similar``)
    * sentence / document embeddings via mean-pooling of word vectors
    * inference on raw Arabic/English text (reuses the existing cleaning +
      tokenization, read-only)

Artifacts (NumPy, no pickled custom classes):
    models_store/embeddings/word2vec.npz          (vectors + words)
    models_store/embeddings/embeddings_config.json

Training entry point:
    train_word2vec(...) / python src/features/embeddings.py

Inference entry point:
    WordEmbeddings.load() -> .vector / .most_similar / .embed_text / .document_matrix

No classifiers are built here — embeddings only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & defaults
# --------------------------------------------------------------------------- #
DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
EMB_DIR = ROOT / "models_store" / "embeddings"
VECTORS_FILE = EMB_DIR / "word2vec.npz"
CONFIG_FILE = EMB_DIR / "embeddings_config.json"

DEFAULT_DIM = 100
DEFAULT_WINDOW = 2
DEFAULT_MIN_COUNT = 5
DEFAULT_NEGATIVES = 5
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 1024
DEFAULT_LR = 0.01
RANDOM_STATE = 42
_NOISE_TABLE_SIZE = 1_000_000


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_token_lists(data_path: Path = DEFAULT_DATA) -> list[list[str]]:
    """Read the pre-tokenized ``tokens`` column into a list of token lists."""
    if not data_path.exists():
        raise SystemExit(
            f"Processed dataset not found: {data_path}\n"
            f"Run: python src/preprocessing/run_preprocessing.py"
        )
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    return [str(t).split() for t in df["tokens"].fillna("").tolist()]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def build_vocab(token_lists, min_count: int = DEFAULT_MIN_COUNT):
    """Return (itos, stoi, counts) for tokens occurring >= ``min_count`` times."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    itos = [w for w, c in counter.most_common() if c >= min_count]
    stoi = {w: i for i, w in enumerate(itos)}
    counts = np.array([counter[w] for w in itos], dtype=np.float64)
    return itos, stoi, counts


# --------------------------------------------------------------------------- #
# Training data (skip-gram pairs) + negative sampling table
# --------------------------------------------------------------------------- #
def _build_pairs(token_lists, stoi, window: int):
    """Generate (center, context) index pairs within ``window``."""
    centers: list[int] = []
    contexts: list[int] = []
    for tokens in token_lists:
        ids = [stoi[t] for t in tokens if t in stoi]
        n = len(ids)
        for i, center in enumerate(ids):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(center)
                    contexts.append(ids[j])
    return (np.asarray(centers, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64))


def _build_noise_table(counts: np.ndarray, size: int = _NOISE_TABLE_SIZE) -> np.ndarray:
    """Unigram^0.75 sampling table for drawing negative samples quickly."""
    weights = counts ** 0.75
    probs = weights / weights.sum()
    table = np.floor(probs * size).astype(np.int64)
    indices = np.repeat(np.arange(len(counts)), table)
    if len(indices) == 0:  # degenerate corpus guard
        indices = np.arange(len(counts))
    return indices


# --------------------------------------------------------------------------- #
# Training (PyTorch skip-gram with negative sampling)
# --------------------------------------------------------------------------- #
def train_word2vec(
    data_path: Path = DEFAULT_DATA,
    dim: int = DEFAULT_DIM,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    negatives: int = DEFAULT_NEGATIVES,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    save: bool = True,
    verbose: bool = True,
) -> "WordEmbeddings":
    """Train skip-gram word vectors and return a ``WordEmbeddings`` instance."""
    import torch  # imported lazily so inference needs only NumPy
    import torch.nn.functional as F

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    token_lists = load_token_lists(data_path)
    itos, stoi, counts = build_vocab(token_lists, min_count)
    if not itos:
        raise SystemExit("Empty vocabulary — lower --min_count.")

    centers, contexts = _build_pairs(token_lists, stoi, window)
    noise = _build_noise_table(counts)
    vocab_size = len(itos)

    if verbose:
        print(f"Vocab size : {vocab_size}")
        print(f"Pairs      : {len(centers):,}")
        print(f"Dim={dim} window={window} neg={negatives} epochs={epochs}")

    # Two embedding tables: input (center) and output (context).
    in_emb = torch.nn.Embedding(vocab_size, dim)
    out_emb = torch.nn.Embedding(vocab_size, dim)
    torch.nn.init.uniform_(in_emb.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(out_emb.weight)
    optimizer = torch.optim.Adam(list(in_emb.parameters()) + list(out_emb.parameters()), lr=lr)

    n = len(centers)
    order = np.arange(n)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(order)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            c = torch.from_numpy(centers[batch])
            o = torch.from_numpy(contexts[batch])
            neg_idx = torch.from_numpy(
                noise[np.random.randint(0, len(noise), size=(len(batch), negatives))]
            )

            v = in_emb(c)                       # (B, D)
            u_pos = out_emb(o)                  # (B, D)
            u_neg = out_emb(neg_idx)            # (B, K, D)

            pos_loss = F.logsigmoid((v * u_pos).sum(1))                  # (B,)
            neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)      # (B, K)
            neg_loss = F.logsigmoid(-neg_score).sum(1)                   # (B,)
            loss = -(pos_loss + neg_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose:
            print(f"  epoch {epoch}/{epochs}  loss={total_loss / max(1, n_batches):.4f}")

    vectors = in_emb.weight.detach().cpu().numpy().astype(np.float32)
    emb = WordEmbeddings(vectors, itos, l2_normalize=True)

    if save:
        config = {
            "dim": dim, "window": window, "min_count": min_count,
            "negatives": negatives, "epochs": epochs, "vocab_size": vocab_size,
            "pairs": int(len(centers)),
        }
        emb.save(VECTORS_FILE, CONFIG_FILE, config)
        if verbose:
            print(f"Saved -> {VECTORS_FILE}")
    return emb


# --------------------------------------------------------------------------- #
# Embeddings container (inference; NumPy only)
# --------------------------------------------------------------------------- #
class WordEmbeddings:
    """Trained word vectors with similarity + sentence-embedding helpers."""

    def __init__(self, vectors: np.ndarray, itos: list[str], l2_normalize: bool = True):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.dim = int(vectors.shape[1])
        self.vectors = vectors.astype(np.float32)
        # Unit-normalized copy for cosine similarity.
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.unit = self.vectors / norms if l2_normalize else self.vectors

    # -- persistence --------------------------------------------------------
    def save(self, vectors_path: Path = VECTORS_FILE,
             config_path: Path = CONFIG_FILE, config: dict | None = None) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(vectors_path, vectors=self.vectors, words=np.array(self.itos))
        if config is not None:
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    @classmethod
    def load(cls, vectors_path: Path = VECTORS_FILE) -> "WordEmbeddings":
        if not Path(vectors_path).exists():
            raise FileNotFoundError(
                f"Embeddings not found: {vectors_path}\n"
                f"Train them first: python src/features/embeddings.py"
            )
        data = np.load(vectors_path, allow_pickle=False)
        return cls(data["vectors"], [str(w) for w in data["words"]])

    # -- lookups ------------------------------------------------------------
    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def vector(self, word: str) -> np.ndarray | None:
        idx = self.stoi.get(word)
        return None if idx is None else self.vectors[idx]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        """Return the ``topn`` nearest words by cosine similarity."""
        idx = self.stoi.get(word)
        if idx is None:
            return []
        sims = self.unit @ self.unit[idx]
        best = np.argsort(sims)[::-1]
        out = [(self.itos[i], float(sims[i])) for i in best if i != idx]
        return out[:topn]

    # -- sentence / document embeddings ------------------------------------
    def embed_tokens(self, tokens) -> np.ndarray:
        """Mean-pool the vectors of known tokens (zeros if none are known)."""
        vecs = [self.vectors[self.stoi[t]] for t in tokens if t in self.stoi]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def embed_text(self, text: str, lang: str | None = None) -> np.ndarray:
        """Embed raw text: clean + tokenize (read-only reuse) then mean-pool."""
        from src.preprocessing.cleaning import clean_text
        from src.preprocessing.tokenization import preprocess_tokens

        import re
        if lang is None:
            lang = "ar" if re.search(r"[؀-ۿ]", text or "") else "en"
        tokens = preprocess_tokens(clean_text(text, lang), lang)
        return self.embed_tokens(tokens)

    def document_matrix(self, token_lists) -> np.ndarray:
        """Stack mean-pooled embeddings for many token lists -> (N, dim)."""
        return np.vstack([self.embed_tokens(toks) for toks in token_lists])

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse
    p = argparse.ArgumentParser(description="Train Skip-Gram word embeddings.")
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--min_count", type=int, default=DEFAULT_MIN_COUNT)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = p.parse_args()

    train_word2vec(dim=args.dim, window=args.window,
                   min_count=args.min_count, epochs=args.epochs)


if __name__ == "__main__":
    main()
