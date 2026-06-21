"""Evaluate the improved BiLSTM and compare LR / NN / LSTM / BiLSTM.

Steps:
    1. Re-run the ORIGINAL LSTM (unchanged) on the group-aware split — kept for
       comparison.
    2. Train + evaluate the BiLSTM (bidirectional, pretrained Word2Vec init,
       early stopping, gradient clipping, LR scheduler) on the SAME split.
    3. Compute LR + NN on the same split.
    4. Print the 4-model comparison (sentiment + category), save BiLSTM
       confusion matrices, save the BiLSTM models, and run custom predictions.

Usage:
    python src/models/test_bilstm.py
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
    evaluate_task as lstm_evaluate_task,
    load_dataframe,
    token_lists_from_df,
)
from src.models.bilstm import (  # noqa: E402
    BILSTM_CATEGORY_PATH,
    BILSTM_SENTIMENT_PATH,
    evaluate_task as bilstm_evaluate_task,
    predict_text,
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
    print(f"  {'Model':<28}{'Accuracy':>10}{'Macro-F1':>10}")
    print("  " + "-" * 48)
    for name, res in rows:
        acc, f1 = _acc_f1(res)
        print(f"  {name:<28}{acc:>10.4f}{f1:>10.4f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_dataframe()
    token_lists = token_lists_from_df(df)
    groups = build_group_keys(df)
    train_idx, test_idx = group_aware_split(groups)
    y_sent = df["sentiment"].to_numpy()
    y_cat = df["category"].to_numpy()
    print(f"docs={len(df)}  template_groups={len(set(groups))}  "
          f"train={len(train_idx)}  test={len(test_idx)}")

    emb = load_embeddings()

    # 1) Original LSTM (unchanged) — kept for comparison.
    print("\n[1/4] Original LSTM …")
    lstm_sent = lstm_evaluate_task(token_lists, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    lstm_cat = lstm_evaluate_task(token_lists, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    # 2) BiLSTM + pretrained embeddings.
    print("[2/4] BiLSTM + pretrained embeddings …")
    bi_sent = bilstm_evaluate_task(token_lists, y_sent, train_idx, test_idx, groups,
                                   SENTIMENT_LABELS, emb=emb)
    bi_cat = bilstm_evaluate_task(token_lists, y_cat, train_idx, test_idx, groups,
                                  CATEGORY_LABELS, emb=emb)
    print(f"   pretrained token coverage: sentiment={bi_sent['pretrained_hits']}/"
          f"{bi_sent['vocab_size']}, category={bi_cat['pretrained_hits']}/{bi_cat['vocab_size']}")

    # 3) LR + NN on the same split.
    print("[3/4] Logistic Regression + Neural Network …")
    _, X_tfidf = prepare_features(df)
    lr_sent = evaluate_group(X_tfidf, y_sent, groups, SENTIMENT_LABELS)
    lr_cat = evaluate_group(X_tfidf, y_cat, groups, CATEGORY_LABELS)
    X_emb = document_embeddings(df, emb)
    nn_sent = nn_evaluate_task(X_emb, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    nn_cat = nn_evaluate_task(X_emb, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    # BiLSTM detailed metrics + confusion matrices.
    print_evaluation("Sentiment — BiLSTM (group-aware)",
                     bi_sent["y_test"], bi_sent["y_pred"], labels=SENTIMENT_LABELS)
    display_confusion(bi_sent["y_test"], bi_sent["y_pred"], SENTIMENT_LABELS,
                      title="Sentiment — BiLSTM (group-aware)",
                      out_path=REPORTS_DIR / "confusion_bilstm_sentiment.png")
    print_evaluation("Category — BiLSTM (group-aware)",
                     bi_cat["y_test"], bi_cat["y_pred"], labels=CATEGORY_LABELS)
    display_confusion(bi_cat["y_test"], bi_cat["y_pred"], CATEGORY_LABELS,
                      title="Category — BiLSTM (group-aware)",
                      out_path=REPORTS_DIR / "confusion_bilstm_category.png")

    # 4-model comparison.
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (group-aware split)")
    print("=" * 60)
    _comparison_table("Sentiment", [
        ("Logistic Regression", lr_sent),
        ("Neural Network", nn_sent),
        ("LSTM", lstm_sent),
        ("BiLSTM + pretrained", bi_sent),
    ])
    _comparison_table("Category", [
        ("Logistic Regression", lr_cat),
        ("Neural Network", nn_cat),
        ("LSTM", lstm_cat),
        ("BiLSTM + pretrained", bi_cat),
    ])

    # Final BiLSTM models on all data + save.
    print("\n[4/4] Training final BiLSTM models on all data and saving …")
    sent_bundle = train_final_and_save(token_lists, y_sent, groups, SENTIMENT_LABELS,
                                       BILSTM_SENTIMENT_PATH, emb=emb)
    cat_bundle = train_final_and_save(token_lists, y_cat, groups, CATEGORY_LABELS,
                                      BILSTM_CATEGORY_PATH, emb=emb)
    print(f"Saved -> {BILSTM_SENTIMENT_PATH.name}, {BILSTM_CATEGORY_PATH.name}")

    print("\n" + "=" * 60)
    print("  Custom example predictions (BiLSTM + pretrained)")
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
