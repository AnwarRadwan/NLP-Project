"""Logistic Regression baselines for the BZU platform (with honest evaluation).

Two classifiers share a single TF-IDF representation built from ``clean_text``
(the same vectorizer used by the retrieval layer):

    Task 1 — Sentiment : positive / negative / neutral
    Task 2 — Category  : Course Feedback / Student Decisions / University Discussions

Pipeline per task:  clean_text -> TF-IDF -> Logistic Regression.

Evaluation
----------
Because the dataset is template-generated, a plain random split lets near-
duplicate rows from the same template appear in *both* train and test, which
inflates accuracy to ~100%. We therefore report two estimates:

    * random split        — stratified train_test_split (optimistic / leaky)
    * group-aware split    — GroupShuffleSplit + GroupKFold, grouping rows by a
                             "template skeleton" so whole templates are held out

Inference helpers
-----------------
    * Arabic positive-synonym expansion (ممتاز/رائع/... -> in-vocabulary words)
    * a category rule/boost post-processing layer (keyword -> category)

Both are optional layers on top of the Logistic Regression core.

Saved artifacts (models trained on ALL data):
    models_store/logistic_regression_sentiment.joblib
    models_store/logistic_regression_category.joblib
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
)

from src.features import tfidf  # noqa: E402
from src.preprocessing.cleaning import clean_text, normalize_arabic  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
MODELS_DIR = ROOT / "models_store"
SENTIMENT_PATH = MODELS_DIR / "logistic_regression_sentiment.joblib"
CATEGORY_PATH = MODELS_DIR / "logistic_regression_category.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD_SPLITS = 5

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
CAT_COURSE = "Course Feedback"
CAT_DECISIONS = "Student Decisions"
CAT_DISCUSSIONS = "University Discussions"
CATEGORY_LABELS = [CAT_COURSE, CAT_DECISIONS, CAT_DISCUSSIONS]

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """Return 'ar' if the text contains Arabic characters, else 'en'."""
    return "ar" if _ARABIC_RE.search(text or "") else "en"


def _new_lr() -> LogisticRegression:
    """Construct a Logistic Regression with the project's standard settings."""
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # neutral / Student Decisions are smaller
        random_state=RANDOM_STATE,
    )


# --------------------------------------------------------------------------- #
# Data & features
# --------------------------------------------------------------------------- #
def load_dataset(data_path: Path = DEFAULT_DATA) -> pd.DataFrame:
    if not data_path.exists():
        raise SystemExit(
            f"Processed dataset not found: {data_path}\n"
            f"Run: python src/preprocessing/run_preprocessing.py"
        )
    return pd.read_csv(data_path, encoding="utf-8-sig")


def prepare_features(df: pd.DataFrame):
    """Fit one shared TF-IDF on ``clean_text``; return (vectorizer, X)."""
    texts = df["clean_text"].fillna("").astype(str).tolist()
    vectorizer, X = tfidf.build_tfidf(texts)
    return vectorizer, X


# --------------------------------------------------------------------------- #
# Group keys ("template skeletons") for leakage-free evaluation
# --------------------------------------------------------------------------- #
def _build_filler_tokens() -> set[str]:
    """Tokens that vary *within* a template (slot fillers), normalized.

    Removing these from a row's ``clean_text`` leaves the template skeleton, so
    rows generated from the same template collapse to the same group key.
    """
    from src.data.generate_dataset import COURSES, HASHTAGS, POOLS, PROFESSORS

    toks: set[str] = set()

    def add(value: str, lang: str) -> None:
        for tok in clean_text(value, lang).split():
            toks.add(tok)

    for item in COURSES + PROFESSORS:
        add(item["en"], "en")
        add(item["ar"], "ar")
    for lang in ("en", "ar"):
        for values in POOLS[lang].values():
            for value in values:
                add(value, lang)
        for tag in HASHTAGS[lang]:
            add(tag, lang)
    return toks


def build_group_keys(df: pd.DataFrame) -> np.ndarray:
    """Map each row to its template-skeleton signature (the grouping key)."""
    filler = _build_filler_tokens()

    def signature(clean: str) -> str:
        return " ".join(tok for tok in str(clean).split() if tok not in filler)

    return df["clean_text"].fillna("").map(signature).to_numpy()


# --------------------------------------------------------------------------- #
# Evaluation (returns eval-ready result blocks; printing happens elsewhere)
# --------------------------------------------------------------------------- #
def _fit_predict(X_train, y_train, X_test, y_test, labels) -> dict:
    clf = _new_lr()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "clf": clf,
        "y_test": y_test,
        "y_pred": y_pred,
        "labels": labels,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "accuracy": accuracy_score(y_test, y_pred),
    }


def evaluate_random(X, y, labels) -> dict:
    """Stratified random train/test split (optimistic — allows template leakage)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return _fit_predict(X_tr, y_tr, X_te, y_te, labels)


def evaluate_group(X, y, groups, labels) -> dict:
    """Group-aware split: whole template skeletons held out (no leakage)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return _fit_predict(X[train_idx], y[train_idx], X[test_idx], y[test_idx], labels)


def group_kfold_accuracy(X, y, groups, n_splits: int = KFOLD_SPLITS):
    """Mean/std accuracy across GroupKFold folds (robust group-aware estimate)."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = _new_lr()
        clf.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_task(X, y, groups, labels) -> dict:
    """Run both random and group-aware evaluations for one task."""
    g_mean, g_std = group_kfold_accuracy(X, y, groups)
    return {
        "random": evaluate_random(X, y, labels),
        "group": evaluate_group(X, y, groups, labels),
        "gkf_mean": g_mean,
        "gkf_std": g_std,
    }


# --------------------------------------------------------------------------- #
# Final models (trained on ALL data) + persistence
# --------------------------------------------------------------------------- #
def train_final(X, y) -> LogisticRegression:
    clf = _new_lr()
    clf.fit(X, y)
    return clf


def save_bundle(vectorizer, clf, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf}, path)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Train it first: python src/models/test_logistic_regression.py"
        )
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Inference layer 1: Arabic positive-synonym expansion
# --------------------------------------------------------------------------- #
# Map out-of-vocabulary positive words to in-vocabulary equivalents so the
# sentiment model can "see" the positive signal.
POSITIVE_SYNONYMS = {
    "ممتاز": "رائعة",
    "ممتازة": "رائعة",
    "ممتازين": "رائعة",
    "رائع": "رائعة",
    "رهيب": "رائعة",
    "منيح": "منيح",       # already in vocabulary
    "واضح": "واضحة",
}


def expand_positive_synonyms(text: str) -> str:
    """Replace known positive synonyms with in-vocabulary canonical forms."""
    return " ".join(POSITIVE_SYNONYMS.get(tok, tok) for tok in text.split())


# --------------------------------------------------------------------------- #
# Inference layer 2: category rule/boost post-processing
# --------------------------------------------------------------------------- #
# (category, [normalized keyword/phrase, ...]). Keywords are normalized the same
# way as clean_text so they match the cleaned query (e.g. إلكتروني -> الكتروني).
_CATEGORY_RULES_RAW = [
    (CAT_DISCUSSIONS, ["التسجيل", "ريتاج", "شعب", "سحب واضافة"]),
    (CAT_COURSE, ["دكتور", "دكتورة", "شرح", "محاضرة", "امتحان", "واجب"]),
    (CAT_DECISIONS, ["اضراب", "اعتصام", "تعليق دوام", "إلكتروني", "وجاهي", "مجلس الطلبة"]),
]
CATEGORY_RULES = [
    (cat, [normalize_arabic(kw) for kw in kws]) for cat, kws in _CATEGORY_RULES_RAW
]
CATEGORY_RULE_BOOST = 0.6


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_sentiment(
    bundle: dict, text: str, lang: str | None = None, expand: bool = False
) -> tuple[str, float]:
    """Predict sentiment; optionally apply Arabic positive-synonym expansion."""
    lang = lang or detect_language(text)
    cleaned = clean_text(text, lang)
    if expand and lang == "ar":
        cleaned = expand_positive_synonyms(cleaned)
    X = bundle["vectorizer"].transform([cleaned])
    proba = bundle["model"].predict_proba(X)[0]
    classes = bundle["model"].classes_
    idx = int(proba.argmax())
    return str(classes[idx]), float(proba[idx])


def predict_category(
    bundle: dict, text: str, lang: str | None = None, use_rules: bool = False
) -> tuple[str, float]:
    """Predict category; optionally apply the keyword rule/boost layer."""
    lang = lang or detect_language(text)
    cleaned = clean_text(text, lang)
    X = bundle["vectorizer"].transform([cleaned])
    proba = bundle["model"].predict_proba(X)[0]
    classes = list(bundle["model"].classes_)
    scores = dict(zip(classes, proba))

    if use_rules:
        for category, keywords in CATEGORY_RULES:
            if category in scores and any(kw in cleaned for kw in keywords):
                scores[category] += CATEGORY_RULE_BOOST

    label = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[label] / total if total > 0 else 0.0
    return label, float(confidence)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_and_evaluate_all(data_path: Path = DEFAULT_DATA, save: bool = True) -> dict:
    """Build features, evaluate (random + group-aware), train+save final models."""
    df = load_dataset(data_path)
    vectorizer, X = prepare_features(df)
    groups = build_group_keys(df)

    y_sent = df["sentiment"].to_numpy()
    y_cat = df["category"].to_numpy()

    sentiment_eval = evaluate_task(X, y_sent, groups, SENTIMENT_LABELS)
    category_eval = evaluate_task(X, y_cat, groups, CATEGORY_LABELS)

    # Final models are trained on ALL data for deployment / custom predictions.
    sentiment_clf = train_final(X, y_sent)
    category_clf = train_final(X, y_cat)

    if save:
        save_bundle(vectorizer, sentiment_clf, SENTIMENT_PATH)
        save_bundle(vectorizer, category_clf, CATEGORY_PATH)

    return {
        "vectorizer": vectorizer,
        "vocab_size": len(vectorizer.vocabulary_),
        "n_docs": X.shape[0],
        "n_groups": len(set(groups)),
        "sentiment_eval": sentiment_eval,
        "category_eval": category_eval,
        "sentiment_bundle": {"vectorizer": vectorizer, "model": sentiment_clf},
        "category_bundle": {"vectorizer": vectorizer, "model": category_clf},
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Build through the importable package so the pickled TF-IDF tokenizer is
    # qualified as src.features.tfidf (not __main__) and stays loadable.
    from src.models import logistic_regression as pkg
    out = pkg.train_and_evaluate_all()
    print(f"docs={out['n_docs']} vocab={out['vocab_size']} groups={out['n_groups']}")
    print(f"Saved -> {SENTIMENT_PATH.name}, {CATEGORY_PATH.name}")


if __name__ == "__main__":
    main()
