"""Feed-Forward Neural Network classifiers over Word2Vec document embeddings.

Two tasks share the same input representation — a document embedding produced by
**averaging** the Skip-Gram word vectors (``src/features/embeddings.py``) of a
text's tokens:

    Task 1 — Sentiment : positive / negative / neutral
    Task 2 — Category  : Course Feedback / Student Decisions / University Discussions

Architecture (PyTorch):
    Input(embedding_dim) -> Linear -> ReLU -> Dropout
                          -> Linear -> ReLU -> Dropout
                          -> Linear(num_classes)

Evaluation reuses the *exact* group-aware split from the Logistic Regression
module (same ``build_group_keys`` + ``GroupShuffleSplit`` parameters), so the NN
and LR numbers are directly comparable.

Saved artifacts:
    models_store/neural_network_sentiment.pt
    models_store/neural_network_category.pt

This builds a feed-forward network only — no LSTM / AraBERT.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.features.embeddings import WordEmbeddings  # noqa: E402
# Reuse (do NOT modify) the LR split parameters for an identical comparison.
from src.models.logistic_regression import RANDOM_STATE, TEST_SIZE  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & hyper-parameters
# --------------------------------------------------------------------------- #
DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
MODELS_DIR = ROOT / "models_store"
NN_SENTIMENT_PATH = MODELS_DIR / "neural_network_sentiment.pt"
NN_CATEGORY_PATH = MODELS_DIR / "neural_network_category.pt"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.3
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FeedForwardNet(nn.Module):
    """Simple MLP: (Linear -> ReLU -> Dropout) x N -> Linear(num_classes)."""

    def __init__(self, input_dim: int, hidden_dims, num_classes: int, dropout: float):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Data / features
# --------------------------------------------------------------------------- #
def load_embeddings() -> WordEmbeddings:
    """Load the trained Word2Vec vectors (built by src/features/embeddings.py)."""
    return WordEmbeddings.load()


def load_dataframe(data_path: Path = DEFAULT_DATA) -> pd.DataFrame:
    if not data_path.exists():
        raise SystemExit(
            f"Processed dataset not found: {data_path}\n"
            f"Run: python src/preprocessing/run_preprocessing.py"
        )
    return pd.read_csv(data_path, encoding="utf-8-sig")


def document_embeddings(df: pd.DataFrame, emb: WordEmbeddings) -> np.ndarray:
    """Average word vectors per row -> (n_docs, embedding_dim) float32 matrix."""
    token_lists = [str(t).split() for t in df["tokens"].fillna("")]
    return emb.document_matrix(token_lists).astype(np.float32)


def group_aware_split(groups: np.ndarray):
    """Same GroupShuffleSplit as Logistic Regression (identical indices)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    n = len(groups)
    train_idx, test_idx = next(gss.split(np.zeros((n, 1)), groups=groups))
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Standardization & class weights
# --------------------------------------------------------------------------- #
def _fit_standardizer(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(X: np.ndarray, mean, std) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Training & inference
# --------------------------------------------------------------------------- #
def train_model(X: np.ndarray, y_str, labels, epochs: int = EPOCHS, verbose: bool = False):
    """Train the MLP on (X, y_str); return (model, mean, std)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    mean, std = _fit_standardizer(X)
    Xs = _apply_standardizer(X, mean, std)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[v] for v in y_str], dtype=np.int64)

    model = FeedForwardNet(X.shape[1], HIDDEN_DIMS, len(labels), DROPOUT)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_idx, len(labels)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y_idx)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    epoch {epoch:>3}/{epochs}  loss={epoch_loss / max(1, n):.4f}")
    return model, mean, std


def predict_proba(model, mean, std, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(_apply_standardizer(X, mean, std)))
        return torch.softmax(logits, dim=1).numpy()


def evaluate_task(X, y_str, train_idx, test_idx, labels) -> dict:
    """Train on the group-train split, predict on the held-out group-test split."""
    model, mean, std = train_model(X[train_idx], y_str[train_idx], labels)
    probs = predict_proba(model, mean, std, X[test_idx])
    y_pred = np.array([labels[i] for i in probs.argmax(axis=1)])
    from sklearn.metrics import accuracy_score
    return {
        "y_test": y_str[test_idx],
        "y_pred": y_pred,
        "labels": labels,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": accuracy_score(y_str[test_idx], y_pred),
    }


# --------------------------------------------------------------------------- #
# Persistence (.pt) + raw-text prediction
# --------------------------------------------------------------------------- #
def save_model(model, labels, mean, std, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(model.net[0].in_features),
            "hidden_dims": list(HIDDEN_DIMS),
            "num_classes": len(labels),
            "dropout": DROPOUT,
            "labels": list(labels),
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
        },
        path,
    )


def load_model(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Train it first: python src/models/test_neural_network.py"
        )
    ckpt = torch.load(path, map_location="cpu")
    model = FeedForwardNet(ckpt["input_dim"], ckpt["hidden_dims"],
                           ckpt["num_classes"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "labels": ckpt["labels"],
        "mean": np.array(ckpt["mean"], dtype=np.float32),
        "std": np.array(ckpt["std"], dtype=np.float32),
    }


def predict_text(bundle: dict, emb: WordEmbeddings, text: str,
                 lang: str | None = None) -> tuple[str, float]:
    """Embed raw text, run the MLP, return (label, confidence)."""
    vec = emb.embed_text(text, lang).reshape(1, -1).astype(np.float32)
    probs = predict_proba(bundle["model"], bundle["mean"], bundle["std"], vec)[0]
    idx = int(probs.argmax())
    return bundle["labels"][idx], float(probs[idx])


def train_final_and_save(X, y_str, labels, path: Path) -> dict:
    """Train on ALL data (for deployment) and save; return a usable bundle."""
    model, mean, std = train_model(X, y_str, labels)
    save_model(model, labels, mean, std, path)
    return {"model": model, "labels": labels, "mean": mean, "std": std}
