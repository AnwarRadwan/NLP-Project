"""SQLite persistence layer (lightweight, fail-safe).

Stores two things:
    * dataset_records  — a copy of the processed bilingual dataset
    * prediction_logs  — a history of predictions made from the UI pages

Every function is defensive: on any SQLite/IO error it returns a safe default
(0, [], or False) instead of raising, so the Streamlit app never crashes
because of the database.

Public API:
    init_db()                  -> bool
    load_dataset_to_db()       -> int   (rows in dataset_records)
    log_prediction(...)        -> bool
    get_prediction_logs(limit) -> list[dict]
    dataset_record_count()     -> int
    prediction_log_count()     -> int
    bootstrap()                -> dict  (init + load on startup)
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "bzu_platform.db"
DATA_PATH = ROOT / "data" / "processed" / "bzu_dataset_clean.csv"

_DATASET_COLUMNS = [
    "id", "text", "language", "category",
    "sentiment", "course_name", "professor_name",
]


@contextmanager
def _connect():
    """Yield a connection, committing on success and always closing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def init_db() -> bool:
    """Create the tables if they do not exist. Returns True on success."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_records (
                    id             INTEGER PRIMARY KEY,
                    text           TEXT,
                    language       TEXT,
                    category       TEXT,
                    sentiment      TEXT,
                    course_name    TEXT,
                    professor_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp            TEXT,
                    source_page          TEXT,
                    input_text           TEXT,
                    predicted_sentiment  TEXT,
                    sentiment_confidence REAL,
                    predicted_category   TEXT,
                    category_confidence  REAL
                )
                """
            )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #
def _count(table: str) -> int:
    try:
        with _connect() as conn:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def dataset_record_count() -> int:
    return _count("dataset_records")


def prediction_log_count() -> int:
    return _count("prediction_logs")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_dataset_to_db(data_path: Path = DATA_PATH) -> int:
    """Load the processed CSV into dataset_records if the table is empty.

    Returns the number of rows in dataset_records (0 on failure).
    """
    try:
        if dataset_record_count() > 0:
            return dataset_record_count()          # already loaded
        if not Path(data_path).exists():
            return 0

        import pandas as pd
        df = pd.read_csv(data_path, encoding="utf-8-sig").fillna("")
        for col in _DATASET_COLUMNS:
            if col not in df.columns:
                return 0                            # unexpected schema

        rows = [
            (
                int(r["id"]), str(r["text"]), str(r["language"]),
                str(r["category"]), str(r["sentiment"]),
                str(r["course_name"]), str(r["professor_name"]),
            )
            for _, r in df.iterrows()
        ]
        with _connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO dataset_records "
                "(id, text, language, category, sentiment, course_name, professor_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Prediction logging
# --------------------------------------------------------------------------- #
def log_prediction(
    source_page: str,
    input_text: str,
    predicted_sentiment: str | None = None,
    sentiment_confidence: float | None = None,
    predicted_category: str | None = None,
    category_confidence: float | None = None,
) -> bool:
    """Append one prediction to prediction_logs. Returns True on success."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    timestamp, source_page, input_text,
                    predicted_sentiment, sentiment_confidence,
                    predicted_category, category_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    source_page,
                    input_text,
                    predicted_sentiment,
                    float(sentiment_confidence) if sentiment_confidence is not None else None,
                    predicted_category,
                    float(category_confidence) if category_confidence is not None else None,
                ),
            )
        return True
    except Exception:
        return False


def get_prediction_logs(limit: int = 50) -> list[dict]:
    """Return the most recent prediction logs (newest first)."""
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Startup helper
# --------------------------------------------------------------------------- #
def bootstrap() -> dict:
    """Initialize schema and load the dataset (idempotent). Safe to call often."""
    ok = init_db()
    loaded = load_dataset_to_db() if ok else 0
    return {
        "ok": ok,
        "db_path": str(DB_PATH),
        "dataset_records": loaded,
        "prediction_logs": prediction_log_count(),
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(bootstrap())
