"""LSTM classifiers for the BZU platform (sentiment + category).

Sequence models over the dataset's ``tokens`` column. Unlike the averaged
Word2Vec neural net, the LSTM learns its **own embedding layer**, and for the
group-aware evaluation the vocabulary + embeddings are built on the *train split
only* — so the held-out estimate is leakage-free and fairly comparable to the
Logistic Regression and Neural Network baselines.

Architecture (PyTorch):
    Embedding(padding_idx=0) -> LSTM -> Dropout -> Linear(num_classes) -> Softmax

Pipeline pieces required by the task:
    * vocabulary creation        (build_vocab)
    * sequence encoding          (encode)
    * padding + lengths          (encode -> pack_padded_sequence)
    * batching with DataLoader   (train_lstm)

Saved artifacts (self-contained: weights + vocab + config):
    models_store/lstm_sentiment.pt
    models_store/lstm_category.pt

No AraBERT here — LSTM only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

# Reuse (do NOT modify) the LR split parameters for an identical comparison.
from src.models.logistic_regression import RANDOM_STATE  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & hyper-parameters
# --------------------------------------------------------------------------- #
DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
MODELS_DIR = ROOT / "models_store"
LSTM_SENTIMENT_PATH = MODELS_DIR / "lstm_sentiment.pt"
LSTM_CATEGORY_PATH = MODELS_DIR / "lstm_category.pt"

PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 40
MIN_COUNT = 1

EMBED_DIM = 100
HIDDEN_DIM = 128
DROPOUT = 0.3
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMClassifier(nn.Module):
    """Embedding -> LSTM -> Dropout -> Linear. Logits out (softmax at predict)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float, pad_idx: int = PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)                       # (B, L, E)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)               # h_n: (1, B, H)
        h = self.dropout(h_n[-1])                      # last layer hidden state
        return self.fc(h)                             # (B, num_classes)


# --------------------------------------------------------------------------- #
# Data / vocab / encoding
# --------------------------------------------------------------------------- #
def load_dataframe(data_path: Path = DEFAULT_DATA) -> pd.DataFrame:
    if not data_path.exists():
        raise SystemExit(
            f"Processed dataset not found: {data_path}\n"
            f"Run: python src/preprocessing/run_preprocessing.py"
        )
    return pd.read_csv(data_path, encoding="utf-8-sig")


def token_lists_from_df(df: pd.DataFrame) -> list[list[str]]:
    return [str(t).split() for t in df["tokens"].fillna("")]


def build_vocab(token_lists, min_count: int = MIN_COUNT) -> dict[str, int]:
    """Create {token: index}; reserves <pad>=0 and <unk>=1."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common() if c >= min_count]
    return {w: i for i, w in enumerate(itos)}


def encode(token_lists, stoi: dict, max_len: int = MAX_LEN):
    """Encode token lists to padded id arrays + true lengths."""
    n = len(token_lists)
    ids = np.zeros((n, max_len), dtype=np.int64)   # 0 == <pad>
    lengths = np.ones(n, dtype=np.int64)
    for i, tokens in enumerate(token_lists):
        seq = [stoi.get(t, UNK_IDX) for t in tokens[:max_len]]
        if not seq:                                # never feed a zero-length seq
            seq = [UNK_IDX]
        ids[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return ids, lengths


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
def _class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_lstm(ids, lengths, y_str, vocab_size, labels,
               epochs: int = EPOCHS, verbose: bool = False):
    """Train the LSTM on encoded sequences; return the fitted model."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[v] for v in y_str], dtype=np.int64)

    model = LSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, len(labels), DROPOUT)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_idx, len(labels)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ids), torch.from_numpy(lengths),
                      torch.from_numpy(y_idx)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for xb, lb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    epoch {epoch:>2}/{epochs}  loss={epoch_loss / max(1, n):.4f}")
    return model


def predict_proba(model, ids, lengths, batch_size: int = 256) -> np.ndarray:
    """Softmax probabilities for encoded sequences (batched for safety)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            xb = torch.from_numpy(ids[start:start + batch_size])
            lb = torch.from_numpy(lengths[start:start + batch_size])
            logits = model(xb, lb)
            out.append(torch.softmax(logits, dim=1).numpy())
    return np.vstack(out)


def evaluate_task(token_lists, y_str, train_idx, test_idx, labels) -> dict:
    """Build train-only vocab, train on train split, evaluate on held-out split."""
    train_tokens = [token_lists[i] for i in train_idx]
    stoi = build_vocab(train_tokens, MIN_COUNT)         # vocab from TRAIN only
    ids, lengths = encode(token_lists, stoi, MAX_LEN)

    model = train_lstm(ids[train_idx], lengths[train_idx], y_str[train_idx],
                       len(stoi), labels)
    probs = predict_proba(model, ids[test_idx], lengths[test_idx])
    y_pred = np.array([labels[i] for i in probs.argmax(axis=1)])
    return {
        "y_test": y_str[test_idx],
        "y_pred": y_pred,
        "labels": labels,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "vocab_size": len(stoi),
        "accuracy": accuracy_score(y_str[test_idx], y_pred),
    }


# --------------------------------------------------------------------------- #
# Persistence (.pt) + raw-text prediction
# --------------------------------------------------------------------------- #
def save_model(model, stoi, labels, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": stoi,
            "labels": list(labels),
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "max_len": MAX_LEN,
        },
        path,
    )


def load_model(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Train it first: python src/models/test_lstm.py"
        )
    ckpt = torch.load(path, map_location="cpu")
    model = LSTMClassifier(len(ckpt["vocab"]), ckpt["embed_dim"], ckpt["hidden_dim"],
                           len(ckpt["labels"]), ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "stoi": ckpt["vocab"],
            "labels": ckpt["labels"], "max_len": ckpt["max_len"]}


def predict_text(bundle: dict, text: str, lang: str | None = None) -> tuple[str, float]:
    """Clean + tokenize raw text (read-only reuse), encode, and classify."""
    import re

    from src.preprocessing.cleaning import clean_text
    from src.preprocessing.tokenization import preprocess_tokens

    if lang is None:
        lang = "ar" if re.search(r"[؀-ۿ]", text or "") else "en"
    tokens = preprocess_tokens(clean_text(text, lang), lang)
    ids, lengths = encode([tokens], bundle["stoi"], bundle.get("max_len", MAX_LEN))
    probs = predict_proba(bundle["model"], ids, lengths)[0]
    idx = int(probs.argmax())
    return bundle["labels"][idx], float(probs[idx])


def train_final_and_save(token_lists, y_str, labels, path: Path) -> dict:
    """Train on ALL data (vocab from all data) and save; return a usable bundle."""
    stoi = build_vocab(token_lists, MIN_COUNT)
    ids, lengths = encode(token_lists, stoi, MAX_LEN)
    model = train_lstm(ids, lengths, y_str, len(stoi), labels)
    save_model(model, stoi, labels, path)
    return {"model": model, "stoi": stoi, "labels": labels, "max_len": MAX_LEN}
