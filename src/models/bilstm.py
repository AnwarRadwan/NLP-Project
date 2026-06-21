"""Improved (Bi)LSTM classifier with pretrained embeddings + training tricks.

Builds on the plain LSTM (``src/models/lstm.py``, reused read-only) and adds the
improvements requested for better generalization on the group-aware split:

    * bidirectional LSTM
    * embedding layer initialized from the existing Word2Vec vectors
      (``models_store/embeddings/word2vec.npz``) for tokens that match
    * early stopping on a group-aware validation macro-F1
    * gradient clipping
    * ReduceLROnPlateau learning-rate scheduler
    * up to 25 epochs

Architecture:
    Embedding(pretrained, padding_idx=0)
      -> BiLSTM -> Dropout(concat last fwd+bwd hidden) -> Linear -> Softmax

Saved artifacts (self-contained: weights + vocab + config):
    models_store/bilstm_sentiment.pt
    models_store/bilstm_category.pt

Reuses the same group-aware split + template groups as LR / NN / LSTM so all
four models are compared on identical held-out data. No AraBERT here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.models.logistic_regression import RANDOM_STATE  # noqa: E402
# Reuse (read-only) the vocab / encoding / prediction utilities from the LSTM.
from src.models.lstm import (  # noqa: E402
    EMBED_DIM,
    MAX_LEN,
    MIN_COUNT,
    PAD_IDX,
    UNK_IDX,
    build_vocab,
    encode,
    predict_proba,
)

# --------------------------------------------------------------------------- #
# Paths & hyper-parameters
# --------------------------------------------------------------------------- #
MODELS_DIR = ROOT / "models_store"
BILSTM_SENTIMENT_PATH = MODELS_DIR / "bilstm_sentiment.pt"
BILSTM_CATEGORY_PATH = MODELS_DIR / "bilstm_category.pt"

HIDDEN_DIM = 128
DROPOUT = 0.3
MAX_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM = 5.0
VAL_SIZE = 0.15
EARLY_STOP_PATIENCE = 5


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BiLSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> Dropout -> Linear. Logits out."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout,
                 pad_idx=PAD_IDX, bidirectional=True, pretrained=None, freeze=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
            with torch.no_grad():
                self.embedding.weight[pad_idx].zero_()
            if freeze:
                self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=bidirectional)
        dirs = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * dirs, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # last layer fwd + bwd
        else:
            h = h_n[-1]
        return self.fc(self.dropout(h))


# --------------------------------------------------------------------------- #
# Pretrained embedding matrix
# --------------------------------------------------------------------------- #
def build_pretrained_matrix(stoi: dict, emb, embed_dim: int = EMBED_DIM):
    """Init an embedding matrix from Word2Vec where tokens match; return (mat, hits)."""
    rng = np.random.default_rng(RANDOM_STATE)
    mat = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim,
                      size=(len(stoi), embed_dim)).astype(np.float32)
    mat[PAD_IDX] = 0.0
    hits = 0
    for token, idx in stoi.items():
        vec = emb.vector(token)
        if vec is not None and len(vec) == embed_dim:
            mat[idx] = vec
            hits += 1
    return torch.from_numpy(mat), hits


def _class_weights(y_idx, num_classes):
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _val_macro_f1(model, ids, lengths, y_idx, num_classes) -> float:
    pred = predict_proba(model, ids, lengths).argmax(axis=1)
    return f1_score(y_idx, pred, average="macro",
                    labels=list(range(num_classes)), zero_division=0)


# --------------------------------------------------------------------------- #
# Training (early stopping + clipping + scheduler)
# --------------------------------------------------------------------------- #
def train_bilstm(ids, lengths, y_str, groups, vocab_size, labels,
                 pretrained=None, bidirectional=True, epochs=MAX_EPOCHS,
                 verbose=False):
    """Train with a group-aware validation split; return (model, best_val_f1)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[v] for v in y_str], dtype=np.int64)
    num_classes = len(labels)

    # Group-aware train/val split (validation templates disjoint from train).
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    tr, va = next(gss.split(np.zeros((len(ids), 1)), groups=groups))

    model = BiLSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes,
                             DROPOUT, bidirectional=bidirectional, pretrained=pretrained)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_idx[tr], num_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ids[tr]), torch.from_numpy(lengths[tr]),
                      torch.from_numpy(y_idx[tr])),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, lb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

        val_f1 = _val_macro_f1(model, ids[va], lengths[va], y_idx[va], num_classes)
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"    epoch {epoch:>2}/{epochs}  val_macroF1={val_f1:.4f}"
                  f"  (best={best_f1:.4f})")
        if bad >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"    early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


# --------------------------------------------------------------------------- #
# Evaluation / final training
# --------------------------------------------------------------------------- #
def evaluate_task(token_lists, y_str, train_idx, test_idx, groups, labels,
                  emb=None, bidirectional=True, use_pretrained=True,
                  verbose=False) -> dict:
    """Train-only vocab, optional pretrained init, group-aware val/early stop."""
    train_tokens = [token_lists[i] for i in train_idx]
    stoi = build_vocab(train_tokens, MIN_COUNT)
    ids, lengths = encode(token_lists, stoi, MAX_LEN)

    pretrained, hits = (None, 0)
    if use_pretrained and emb is not None:
        pretrained, hits = build_pretrained_matrix(stoi, emb, EMBED_DIM)

    model, best_val = train_bilstm(
        ids[train_idx], lengths[train_idx], y_str[train_idx], groups[train_idx],
        len(stoi), labels, pretrained=pretrained, bidirectional=bidirectional,
        verbose=verbose,
    )
    probs = predict_proba(model, ids[test_idx], lengths[test_idx])
    y_pred = np.array([labels[i] for i in probs.argmax(axis=1)])
    return {
        "y_test": y_str[test_idx],
        "y_pred": y_pred,
        "labels": labels,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "vocab_size": len(stoi),
        "pretrained_hits": hits,
        "val_macro_f1": best_val,
        "accuracy": accuracy_score(y_str[test_idx], y_pred),
    }


# --------------------------------------------------------------------------- #
# Persistence + raw-text prediction
# --------------------------------------------------------------------------- #
def save_model(model, stoi, labels, bidirectional, path: Path) -> None:
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
            "bidirectional": bidirectional,
        },
        path,
    )


def load_model(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Train it first: python src/models/test_bilstm.py"
        )
    ckpt = torch.load(path, map_location="cpu")
    model = BiLSTMClassifier(len(ckpt["vocab"]), ckpt["embed_dim"], ckpt["hidden_dim"],
                             len(ckpt["labels"]), ckpt["dropout"],
                             bidirectional=ckpt["bidirectional"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "stoi": ckpt["vocab"], "labels": ckpt["labels"],
            "max_len": ckpt["max_len"]}


def predict_text(bundle: dict, text: str, lang: str | None = None) -> tuple[str, float]:
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


def train_final_and_save(token_lists, y_str, groups, labels, path: Path,
                         emb=None, bidirectional=True, use_pretrained=True) -> dict:
    """Train on ALL data (vocab from all data) with early stopping; save."""
    stoi = build_vocab(token_lists, MIN_COUNT)
    ids, lengths = encode(token_lists, stoi, MAX_LEN)
    pretrained = None
    if use_pretrained and emb is not None:
        pretrained, _ = build_pretrained_matrix(stoi, emb, EMBED_DIM)
    model, _ = train_bilstm(ids, lengths, y_str, groups, len(stoi), labels,
                            pretrained=pretrained, bidirectional=bidirectional)
    save_model(model, stoi, labels, bidirectional, path)
    return {"model": model, "stoi": stoi, "labels": labels, "max_len": MAX_LEN}
