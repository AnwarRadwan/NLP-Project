"""TF-IDF feature extraction for the bilingual BZU corpus.

Builds a single shared (Arabic + English) TF-IDF model over the ``clean_text``
column of the processed dataset, and persists both the fitted vectorizer and
the document-term matrix so the retrieval layer can reuse them.

Public entry points:
    build_tfidf(texts)                  -> (vectorizer, matrix)
    save_artifacts(vectorizer, matrix)  -> writes to models_store/tfidf/
    load_artifacts()                    -> (vectorizer, matrix)
    build_and_save(data_path, out_dir)  -> build from CSV, save, print stats

A light Arabic-aware tokenizer strips the leading definite article ``ال`` so
that queries like ``التسجيل`` match documents containing ``التسجيل`` (both
reduce to ``تسجيل``). Because the tokenizer lives inside the vectorizer, the
exact same normalization is applied to documents at fit time and to queries at
transform time.

No classifiers are trained here — this is feature extraction + IR only.
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
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from src.preprocessing.tokenization import AR_STOPWORDS, EN_STOPWORDS  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
DEFAULT_MODELS_DIR = ROOT / "models_store" / "tfidf"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
MATRIX_FILE = "tfidf_matrix.npz"

# --------------------------------------------------------------------------- #
# Tokenizer (module-level so the fitted vectorizer stays picklable)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")   # word tokens of length >= 2
_AL = "ال"                                          # Arabic definite article


def _strip_al(token: str) -> str:
    """Strip a leading ``ال`` when a reasonable stem (>= 3 chars) remains.

    The length guard prevents over-stripping short words such as ``الله``.
    """
    if token.startswith(_AL) and len(token) - 2 >= 3:
        return token[2:]
    return token


def arabic_aware_tokenizer(doc: str) -> list[str]:
    """Tokenize and apply light Arabic definite-article normalization."""
    return [_strip_al(tok) for tok in _TOKEN_RE.findall(doc)]


# Combined bilingual stopword list, passed through the same tokenizer
# normalization (e.g. ال-stripping) so the entries match produced tokens.
_STOPWORDS = sorted(
    {_strip_al(w) for w in (EN_STOPWORDS | AR_STOPWORDS) if len(w) >= 2}
)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_tfidf(
    texts,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    sublinear_tf: bool = True,
):
    """Fit a TF-IDF vectorizer on ``texts`` and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(
        tokenizer=arabic_aware_tokenizer,
        stop_words=_STOPWORDS,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=None,   # silence warning: we use a custom tokenizer
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(vectorizer, matrix, out_dir: Path = DEFAULT_MODELS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / VECTORIZER_FILE)
    sparse.save_npz(out_dir / MATRIX_FILE, matrix)


def load_artifacts(out_dir: Path = DEFAULT_MODELS_DIR):
    """Load (vectorizer, matrix); raises FileNotFoundError if missing."""
    vec_path = out_dir / VECTORIZER_FILE
    mat_path = out_dir / MATRIX_FILE
    if not vec_path.exists() or not mat_path.exists():
        raise FileNotFoundError(
            f"TF-IDF artifacts not found in {out_dir}. "
            f"Build them first: python src/features/tfidf.py"
        )
    vectorizer = joblib.load(vec_path)
    matrix = sparse.load_npz(mat_path)
    return vectorizer, matrix


def artifacts_exist(out_dir: Path = DEFAULT_MODELS_DIR) -> bool:
    return (out_dir / VECTORIZER_FILE).exists() and (out_dir / MATRIX_FILE).exists()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_and_save(
    data_path: Path = DEFAULT_DATA,
    out_dir: Path = DEFAULT_MODELS_DIR,
    verbose: bool = True,
):
    """Read the processed CSV, fit TF-IDF on ``clean_text``, save, return."""
    if not data_path.exists():
        raise SystemExit(
            f"Processed dataset not found: {data_path}\n"
            f"Run: python src/preprocessing/run_preprocessing.py"
        )
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    texts = df["clean_text"].fillna("").astype(str).tolist()

    vectorizer, matrix = build_tfidf(texts)
    save_artifacts(vectorizer, matrix, out_dir)

    if verbose:
        print(f"Built TF-IDF from: {data_path}")
        print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")
        print(f"TF-IDF matrix shape: {matrix.shape}")
        print(f"Saved artifacts -> {out_dir}")
    return vectorizer, matrix


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Build through the importable package module so the pickled tokenizer
    # reference is qualified as ``src.features.tfidf.arabic_aware_tokenizer``
    # (not ``__main__``), keeping the saved vectorizer loadable elsewhere.
    from src.features import tfidf as pkg
    pkg.build_and_save()


if __name__ == "__main__":
    main()
