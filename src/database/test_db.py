"""Smoke test for the SQLite persistence layer.

Initializes the database, loads the dataset (if empty), logs one demo
prediction, then prints:
    * database path
    * dataset_records count
    * prediction_logs count
    * latest 5 prediction logs

Usage:
    python src/database/test_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import db  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    status = db.bootstrap()
    print("Database path        :", db.DB_PATH)
    print("Init OK              :", status["ok"])
    print("dataset_records count:", db.dataset_record_count())
    print("prediction_logs count:", db.prediction_log_count())

    # Demonstrate logging end-to-end with one demo entry.
    db.log_prediction(
        source_page="test_db",
        input_text="التسجيل هالفصل كان فوضى",
        predicted_sentiment="negative",
        sentiment_confidence=0.83,
        predicted_category="University Discussions",
        category_confidence=0.61,
    )
    print("prediction_logs count (after demo log):", db.prediction_log_count())

    print("\nLatest 5 prediction logs:")
    logs = db.get_prediction_logs(limit=5)
    if not logs:
        print("  (none)")
    for row in logs:
        print(f"  #{row['id']} [{row['timestamp']}] {row['source_page']} | "
              f"sent={row['predicted_sentiment']}({row['sentiment_confidence']}) "
              f"cat={row['predicted_category']}({row['category_confidence']}) | "
              f"text={row['input_text']!r}")


if __name__ == "__main__":
    main()
