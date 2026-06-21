"""AraBERT-based classifier for the BZU platform.

Uses **AraBERT** (``aubmindlab/bert-base-arabertv02``) as a *frozen contextual
encoder*: each document is turned into a 768-d sentence embedding by mean-pooling
AraBERT's last hidden states (masked by attention). A lightweight classifier head
(the project's existing MLP, reused read-only from ``neural_network.py``) is then
trained on those embeddings using the **same group-aware split** as LR / NN /
LSTM / BiLSTM.

Why frozen-encoder (linear probe) instead of full fine-tuning?
    The environment is CPU-only; back-propagating through a 135M-parameter BERT
    over thousands of examples is impractical here. Encoding the corpus once is
    feasible, and the AraBERT features still give a strong representation. Full
    fine-tuning is a drop-in upgrade on a GPU (noted in the test output).

Notes
    * AraBERT is an Arabic model; the corpus is bilingual, so English rows are
      represented sub-optimally. The test script also reports an Arabic-only
      score to show AraBERT's strength on its target language.
    * Document embeddings are cached to avoid re-encoding on reruns.

Saved artifacts:
    models_store/arabert/arabert_embeddings.npy   (cached doc embeddings)
    models_store/arabert_sentiment.pt             (classifier head)
    models_store/arabert_category.pt              (classifier head)

No further models built here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the existing MLP head + helpers (read-only) for the classifier on top.
from src.models.neural_network import (  # noqa: E402
    load_model as nn_load_model,
    predict_proba as nn_predict_proba,
    save_model as nn_save_model,
    train_model as nn_train_model,
)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
ARABERT_DIR = ROOT / "models_store" / "arabert"
EMB_CACHE = ARABERT_DIR / "arabert_embeddings.npy"
ARABERT_SENTIMENT_PATH = ROOT / "models_store" / "arabert_sentiment.pt"
ARABERT_CATEGORY_PATH = ROOT / "models_store" / "arabert_category.pt"

MAX_LEN = 64
BATCH_SIZE = 32
EMBED_DIM = 768


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class AraBERTEncoder:
    """Frozen AraBERT sentence encoder (mean-pooled last hidden states)."""

    def __init__(self, model_name: str = MODEL_NAME, max_len: int = MAX_LEN):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Optional AraBERT preprocessing (Arabic normalization).
        try:
            from arabert.preprocess import ArabertPreprocessor
            self._pre = ArabertPreprocessor(model_name="bert-base-arabertv02")
        except Exception:
            self._pre = None

    def _preprocess(self, text: str) -> str:
        text = "" if text is None else str(text)
        if self._pre is not None:
            try:
                return self._pre.preprocess(text)
            except Exception:
                return text
        return text

    def encode(self, texts, batch_size: int = BATCH_SIZE, verbose: bool = False) -> np.ndarray:
        torch = self.torch
        texts = [self._preprocess(t) for t in texts]
        out = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state          # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()        # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            out.append((summed / counts).cpu().numpy().astype(np.float32))
            if verbose and (start // batch_size) % 20 == 0:
                print(f"    encoded {min(start + batch_size, n)}/{n}")
        return np.vstack(out)


# --------------------------------------------------------------------------- #
# Corpus encoding (cached)
# --------------------------------------------------------------------------- #
def encode_corpus(df: pd.DataFrame, encoder: AraBERTEncoder | None = None,
                  cache: bool = True, verbose: bool = True) -> np.ndarray:
    """Encode the ``text`` column to AraBERT embeddings, using a disk cache."""
    if cache and EMB_CACHE.exists():
        X = np.load(EMB_CACHE)
        if X.shape[0] == len(df):
            if verbose:
                print(f"Loaded cached AraBERT embeddings {X.shape} from {EMB_CACHE}")
            return X
    if encoder is None:
        encoder = AraBERTEncoder()
    if verbose:
        print(f"Encoding {len(df)} documents with AraBERT (CPU)…")
    X = encoder.encode(df["text"].astype(str).tolist(), verbose=verbose)
    if cache:
        ARABERT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMB_CACHE, X)
        if verbose:
            print(f"Saved AraBERT embeddings -> {EMB_CACHE}")
    return X


# --------------------------------------------------------------------------- #
# Classifier head (reuses the project MLP) + persistence
# --------------------------------------------------------------------------- #
def train_and_save_head(X, y_str, labels, path: Path) -> dict:
    """Train the MLP head on AraBERT features (all data) and save as .pt."""
    model, mean, std = nn_train_model(X, y_str, labels)
    nn_save_model(model, labels, mean, std, path)
    return {"model": model, "labels": labels, "mean": mean, "std": std}


def load_head(path: Path) -> dict:
    return nn_load_model(path)


def predict_text(head_bundle: dict, encoder: AraBERTEncoder, text: str) -> tuple[str, float]:
    """Encode raw text with AraBERT and classify with the trained head."""
    vec = encoder.encode([text]).astype(np.float32)
    probs = nn_predict_proba(head_bundle["model"], head_bundle["mean"],
                             head_bundle["std"], vec)[0]
    idx = int(probs.argmax())
    return head_bundle["labels"][idx], float(probs[idx])
