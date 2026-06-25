"""Train, evaluate, and demo the LSTM classifiers, and compare LR vs NN vs LSTM.

Steps:
    1. Encode the corpus tokens; evaluate the LSTM on the SAME group-aware split
       used by Logistic Regression and the Neural Network.
    2. Print metrics + classification report and save confusion matrices.
    3. Build a 3-model comparison (LR / NN / LSTM) for sentiment and category.
    4. Train final LSTM models on all data, save .pt files, predict custom inputs.

Usage:
    python src/models/test_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.confusion_matrix import display_confusion  # noqa: E402
from src.evaluation.metrics import compute_metrics, print_evaluation  # noqa: E402

# Reused (not modified) modules for an identical split + fair comparison.
from src.models.logistic_regression import (  # noqa: E402
    CATEGORY_LABELS,
    SENTIMENT_LABELS,
    build_group_keys,
    evaluate_group,
    prepare_features,
)
from src.models.neural_network import (  # noqa: E402
    document_embeddings,
    evaluate_task as nn_evaluate_task,
    group_aware_split,
    load_embeddings,
)
from src.models.lstm import (  # noqa: E402
    LSTM_CATEGORY_PATH,
    LSTM_SENTIMENT_PATH,
    evaluate_task as lstm_evaluate_task,
    load_dataframe,
    predict_text,
    token_lists_from_df,
    train_final_and_save,
)

REPORTS_DIR = ROOT / "reports"

CUSTOM_EXAMPLES = [
    "الدكتور رائع وشرح المادة ممتاز",
    "لازم نعمل اضراب ضد الرسوم",
    "التسجيل هالفصل كان فوضى",
    "The professor explains very well",
    "Students are planning a strike",
    "Registration was very confusing",
]


def _acc_f1(res: dict) -> tuple[float, float]:
    f1 = compute_metrics(res["y_test"], res["y_pred"],
                         labels=res["labels"], average="macro")["f1"]
    return res["accuracy"], f1


def _comparison_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title} — group-aware comparison")
    print(f"  {'Model':<22}{'Accuracy':>10}{'Macro-F1':>10}")
    print("  " + "-" * 42)
    for name, res in rows:
        acc, f1 = _acc_f1(res)
        print(f"  {name:<22}{acc:>10.4f}{f1:>10.4f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # --- Shared data + identical group-aware split ------------------------
    df = load_dataframe()
    token_lists = token_lists_from_df(df)
    groups = build_group_keys(df)
    train_idx, test_idx = group_aware_split(groups)
    y_sent = df["sentiment"].to_numpy()
    y_cat = df["category"].to_numpy()
    print(f"docs={len(df)}  template_groups={len(set(groups))}  "
          f"train={len(train_idx)}  test={len(test_idx)}")

    # --- LSTM evaluation ---------------------------------------------------
    print("\nTraining LSTM (sentiment)…")
    lstm_sent = lstm_evaluate_task(token_lists, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    print("Training LSTM (category)…")
    lstm_cat = lstm_evaluate_task(token_lists, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    print_evaluation("Sentiment — LSTM (group-aware)",
                     lstm_sent["y_test"], lstm_sent["y_pred"], labels=SENTIMENT_LABELS)
    display_confusion(lstm_sent["y_test"], lstm_sent["y_pred"], SENTIMENT_LABELS,
                      title="Sentiment — LSTM (group-aware)",
                      out_path=REPORTS_DIR / "confusion_lstm_sentiment.png")

    print_evaluation("Category — LSTM (group-aware)",
                     lstm_cat["y_test"], lstm_cat["y_pred"], labels=CATEGORY_LABELS)
    display_confusion(lstm_cat["y_test"], lstm_cat["y_pred"], CATEGORY_LABELS,
                      title="Category — LSTM (group-aware)",
                      out_path=REPORTS_DIR / "confusion_lstm_category.png")

    # --- LR + NN on the SAME split (for comparison) -----------------------
    print("\nComputing Logistic Regression + Neural Network on the same split…")
    _, X_tfidf = prepare_features(df)
    lr_sent = evaluate_group(X_tfidf, y_sent, groups, SENTIMENT_LABELS)
    lr_cat = evaluate_group(X_tfidf, y_cat, groups, CATEGORY_LABELS)

    emb = load_embeddings()
    X_emb = document_embeddings(df, emb)
    nn_sent = nn_evaluate_task(X_emb, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    nn_cat = nn_evaluate_task(X_emb, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (group-aware split)")
    print("=" * 60)
    _comparison_table("Sentiment", [
        ("Logistic Regression", lr_sent),
        ("Neural Network", nn_sent),
        ("LSTM", lstm_sent),
    ])
    _comparison_table("Category", [
        ("Logistic Regression", lr_cat),
        ("Neural Network", nn_cat),
        ("LSTM", lstm_cat),
    ])

    # --- Final LSTM models (all data) + custom predictions ----------------
    print("\nTraining final LSTM models on all data and saving…")
    sent_bundle = train_final_and_save(token_lists, y_sent, SENTIMENT_LABELS, LSTM_SENTIMENT_PATH)
    cat_bundle = train_final_and_save(token_lists, y_cat, CATEGORY_LABELS, LSTM_CATEGORY_PATH)
    print(f"Saved -> {LSTM_SENTIMENT_PATH.name}, {LSTM_CATEGORY_PATH.name}")

    print("\n" + "=" * 60)
    print("  Custom example predictions (LSTM)")
    print("=" * 60)
    for text in CUSTOM_EXAMPLES:
        s_label, s_conf = predict_text(sent_bundle, text)
        c_label, c_conf = predict_text(cat_bundle, text)
        lang = "ar" if any("؀" <= ch <= "ۿ" for ch in text) else "en"
        print(f"\nText [{lang}]: {text}")
        print(f"  sentiment = {s_label:<8} ({s_conf:.3f})   "
              f"category = {c_label:<22} ({c_conf:.3f})")


if __name__ == "__main__":
    main()
