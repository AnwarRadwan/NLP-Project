"""Information retrieval over the BZU corpus (TF-IDF + cosine similarity).

Loads the persisted TF-IDF vectorizer/matrix and the processed dataset, then
ranks documents against a free-text query by cosine similarity. Arabic and
English queries are both supported.

Quality features layered on top of plain TF-IDF retrieval:
    * Near-duplicate filtering — results that differ only by emojis,
      punctuation, or hashtags are collapsed (the highest-scoring copy is kept).
    * Optional category filter — ``search(query, top_k, category=...)`` restricts
      results to a single module (e.g. "Course Feedback").
    * Intent-aware preference — when the query mentions professor/course words
      the engine softly prefers Course Feedback; for strike/decision words it
      prefers Student Decisions. This re-weights scores, it does not replace
      cosine similarity (zero-similarity documents are never surfaced).

Public entry points:
    SearchEngine.from_default()                  -> ready-to-use engine
    engine.search(query, top_k=10, category=None) -> list[dict]
    search(query, top_k=10, category=None)        -> module-level convenience

Each result is a dict with: text, category, sentiment, similarity_score
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from src.features import tfidf  # noqa: E402
from src.features.tfidf import _strip_al  # noqa: E402  (token normalization)
from src.preprocessing.cleaning import clean_text, normalize_arabic, remove_emojis  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"
DEFAULT_MODELS_DIR = tfidf.DEFAULT_MODELS_DIR

# Canonical category names (match the dataset).
CAT_COURSE = "Course Feedback"
CAT_DECISIONS = "Student Decisions"
CAT_DISCUSSIONS = "University Discussions"
CATEGORIES = [CAT_COURSE, CAT_DECISIONS, CAT_DISCUSSIONS]

# How strongly to prefer the intent-matched category (multiplicative, ranking
# only — the reported similarity_score remains the true cosine value).
PREFER_BOOST = 2.0

# Any character in the Arabic Unicode block marks the query as Arabic.
_ARABIC_RE = re.compile(r"[؀-ۿ]")

# --------------------------------------------------------------------------- #
# Intent vocabularies (normalized: lowercase EN; alef-normalized, ال-stripped AR)
# --------------------------------------------------------------------------- #
_PROF_COURSE_WORDS = {
    # English
    "professor", "professors", "prof", "dr", "doctor", "course", "courses",
    "lecture", "lectures", "exam", "exams", "midterm", "final", "quiz",
    "grading", "grade", "grades", "lab", "labs", "project", "projects",
    "attendance", "teaching", "instructor", "excellent",
    # Arabic
    "دكتور", "مدرس", "استاذ", "مادة", "محاضرة", "محاضرات", "امتحان",
    "ميدتيرم", "فاينل", "كويز", "مشروع", "مختبر", "علامات", "حضور",
    "كورس", "ممتاز", "تدريس", "شرح",
}
_DECISION_WORDS = {
    # English
    "strike", "strikes", "protest", "protests", "boycott", "suspension",
    "suspend", "suspending", "online", "person", "inperson", "hybrid",
    "schedule", "movement", "assembly", "union",
    # Arabic
    "اضراب", "اعتصام", "مقاطعة", "تعليق", "اونلاين", "وجاهي", "دوام",
    "جدول", "حراك", "مظاهرة", "تعطيل",
}

# Drop hashtags (incl. Arabic), then keep only letters/digits for a stable key.
_HASHTAG_RE = re.compile(r"#[\w؀-ۿ]+")
_DEDUP_STRIP_RE = re.compile(r"[^0-9a-zء-ي]+")
_WS_RE = re.compile(r"\s+")


def detect_language(text: str) -> str:
    """Return 'ar' if the text contains Arabic characters, else 'en'."""
    return "ar" if _ARABIC_RE.search(text or "") else "en"


def dedup_key(text: str) -> str:
    """Normalized key for near-duplicate detection.

    Ignores emojis, hashtags, punctuation and case, so two records that differ
    only by those produce the same key.
    """
    t = _HASHTAG_RE.sub(" ", text or "")
    t = remove_emojis(t).lower()
    t = normalize_arabic(t)
    t = _DEDUP_STRIP_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _query_tokens(cleaned_query: str) -> set[str]:
    """Tokens of a cleaned query, with ال-stripping for Arabic matching."""
    return {_strip_al(tok) for tok in cleaned_query.split() if tok}


def infer_preferred_category(cleaned_query: str) -> str | None:
    """Infer a preferred category from query keywords (or None)."""
    tokens = _query_tokens(cleaned_query)
    prof = len(tokens & _PROF_COURSE_WORDS)
    decision = len(tokens & _DECISION_WORDS)
    if prof == 0 and decision == 0:
        return None
    if prof > decision:
        return CAT_COURSE
    if decision > prof:
        return CAT_DECISIONS
    return None  # tie -> no preference


class SearchEngine:
    """TF-IDF + cosine-similarity search over the BZU dataset."""

    def __init__(self, vectorizer, matrix, metadata: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix              # shape: (n_docs, vocab)
        self.metadata = metadata.reset_index(drop=True)
        self._categories = self.metadata["category"].to_numpy()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_default(
        cls,
        data_path: Path = DEFAULT_DATA,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ) -> "SearchEngine":
        """Load artifacts (building them on first use) and the metadata CSV."""
        if tfidf.artifacts_exist(models_dir):
            vectorizer, matrix = tfidf.load_artifacts(models_dir)
        else:
            vectorizer, matrix = tfidf.build_and_save(data_path, models_dir)

        metadata = pd.read_csv(data_path, encoding="utf-8-sig")
        if matrix.shape[0] != len(metadata):
            # Dataset and matrix are out of sync — rebuild to stay consistent.
            vectorizer, matrix = tfidf.build_and_save(data_path, models_dir)
            metadata = pd.read_csv(data_path, encoding="utf-8-sig")
        return cls(vectorizer, matrix, metadata)

    # -- introspection ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Return up to ``top_k`` similar documents.

        Parameters
        ----------
        query : str
            Free-text query (Arabic or English).
        top_k : int
            Maximum number of (de-duplicated) results.
        category : str | None
            If given, restrict results to that category (hard filter). If None,
            an intent-based soft preference may be applied.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        lang = detect_language(query)
        cleaned = clean_text(query, lang)
        if not cleaned:
            return []

        query_vec = self.vectorizer.transform([cleaned])
        # TF-IDF rows are L2-normalized, so this dot product is cosine similarity.
        cosine = cosine_similarity(query_vec, self.matrix).ravel().astype(float)

        # Ranking scores may be boosted toward a preferred category; the
        # *reported* score always stays the true cosine value.
        rank_scores = cosine
        if category is None:
            preferred = infer_preferred_category(cleaned)
            if preferred is not None:
                boost = np.where(self._categories == preferred, PREFER_BOOST, 1.0)
                rank_scores = cosine * boost

        ranked = rank_scores.argsort()[::-1]

        results: list[dict] = []
        seen_keys: set[str] = set()
        for idx in ranked:
            if rank_scores[idx] <= 0.0:
                break  # remaining documents share no terms with the query
            row = self.metadata.iloc[int(idx)]

            if category is not None and row["category"] != category:
                continue

            key = dedup_key(str(row.get("text", "")))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            results.append(
                {
                    "text": row.get("text", ""),
                    "category": row.get("category", ""),
                    "sentiment": row.get("sentiment", ""),
                    "similarity_score": round(float(cosine[idx]), 4),
                }
            )
            if len(results) >= top_k:
                break
        return results


# --------------------------------------------------------------------------- #
# Module-level convenience (lazy, cached engine)
# --------------------------------------------------------------------------- #
_ENGINE: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine.from_default()
    return _ENGINE


def search(query: str, top_k: int = 10, category: str | None = None) -> list[dict]:
    """Search the BZU corpus using the default cached engine."""
    return get_engine().search(query, top_k=top_k, category=category)
