"""Service layer bridging the NLP backend and the Streamlit UI.

Centralizes cached resource loading (dataset, search engine, trained models),
prediction helpers, label localization, and small chart helpers. Every loader
fails gracefully — returning ``None`` instead of raising — so pages can show a
friendly message rather than a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.i18n.translations import t  # noqa: E402
from src.models.logistic_regression import (  # noqa: E402
    CATEGORY_PATH,
    SENTIMENT_PATH,
    detect_language,
    load_bundle,
    predict_category,
    predict_sentiment,
)

DATA_PATH = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
MODELS_DIR = ROOT / "models_store"

# Map internal label -> translation key for localized display.
_SENTIMENT_KEYS = {
    "positive": "sent_positive",
    "negative": "sent_negative",
    "neutral": "sent_neutral",
}
_CATEGORY_KEYS = {
    "Course Feedback": "cat_course_feedback",
    "Student Decisions": "cat_student_decisions",
    "University Discussions": "cat_university_discussions",
}


# --------------------------------------------------------------------------- #
# Cached resource loaders (return None on failure)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    """Load the processed dataset, or None if it is missing/unreadable."""
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_search_engine():
    """Load (or build) the TF-IDF search engine, or None on failure."""
    try:
        from src.retrieval.search_engine import SearchEngine
        return SearchEngine.from_default()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_sentiment_bundle() -> dict | None:
    try:
        return load_bundle(SENTIMENT_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_category_bundle() -> dict | None:
    try:
        return load_bundle(CATEGORY_PATH)
    except Exception:
        return None


# One representative artifact per trained model family (LR, NN, LSTM, BiLSTM, AraBERT).
_MODEL_FAMILY_FILES = (
    "logistic_regression_sentiment.joblib",  # Logistic Regression
    "neural_network_sentiment.pt",           # Neural Network
    "lstm_sentiment.pt",                     # LSTM
    "bilstm_sentiment.pt",                   # BiLSTM
    "arabert_sentiment.pt",                  # AraBERT
)


def model_count() -> int:
    """Number of trained model families with artifacts on disk (e.g. 5 when all present)."""
    return sum((MODELS_DIR / fname).exists() for fname in _MODEL_FAMILY_FILES)


# --------------------------------------------------------------------------- #
# Predictions (use the inference enhancement layers built earlier)
# --------------------------------------------------------------------------- #
def analyze_sentiment(text: str) -> tuple[str, float] | None:
    """Predict sentiment with Arabic positive-synonym expansion enabled."""
    bundle = get_sentiment_bundle()
    if bundle is None:
        return None
    lang = detect_language(text)
    return predict_sentiment(bundle, text, lang, expand=True)


def analyze_category(text: str) -> tuple[str, float] | None:
    """Predict category with the keyword rule/boost layer enabled."""
    bundle = get_category_bundle()
    if bundle is None:
        return None
    lang = detect_language(text)
    return predict_category(bundle, text, lang, use_rules=True)


# --------------------------------------------------------------------------- #
# Localization helpers
# --------------------------------------------------------------------------- #
def localize_sentiment(label: str) -> str:
    return t(_SENTIMENT_KEYS.get(label, label))


def localize_category(label: str) -> str:
    return t(_CATEGORY_KEYS.get(label, label))


# --------------------------------------------------------------------------- #
# Chart helper
# --------------------------------------------------------------------------- #
def pie_chart(counts: pd.Series, title: str):
    """Return a matplotlib pie figure styled for the dark theme."""
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_alpha(0.0)
    colors = plt.cm.Blues_r([0.35, 0.55, 0.75, 0.9][: len(counts)])
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[str(i) for i in counts.index],
        autopct="%1.1f%%",
        colors=colors,
        textprops={"color": "#e6ebf5"},
        startangle=90,
    )
    ax.set_title(title, color="#f4f7ff")
    ax.axis("equal")
    return fig
