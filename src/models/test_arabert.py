"""Evaluate the AraBERT classifier and compare LR / NN / LSTM / BiLSTM / AraBERT.

Steps:
    1. Encode the corpus with AraBERT (cached) -> 768-d document embeddings.
    2. Train + evaluate the AraBERT head on the SAME group-aware split.
    3. Compute LR / NN / LSTM / BiLSTM on the same split.
    4. Print the 5-model comparison (sentiment + category), save AraBERT
       confusion matrices and models, then run custom predictions.

Usage:
    python src/models/test_arabert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

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
from src.models.bilstm import evaluate_task as bilstm_evaluate_task  # noqa: E402
from src.models.arabert import (  # noqa: E402
    ARABERT_CATEGORY_PATH,
    ARABERT_SENTIMENT_PATH,
    AraBERTEncoder,
    encode_corpus,
    predict_text,
    train_and_save_head,
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
    languages = df["language"].to_numpy()
    print(f"docs={len(df)}  template_groups={len(set(groups))}  "
          f"train={len(train_idx)}  test={len(test_idx)}")

    # 1) AraBERT document embeddings (cached) + 2) head evaluation.
    print("\n[AraBERT] encoding corpus …")
    encoder = AraBERTEncoder()
    X_bert = encode_corpus(df, encoder)
    print(f"[AraBERT] embeddings: {X_bert.shape}")

    ab_sent = nn_evaluate_task(X_bert, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    ab_cat = nn_evaluate_task(X_bert, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    # Arabic-only view (AraBERT's target language).
    ar_mask = languages[test_idx] == "ar"
    if ar_mask.any():
        ar_acc_sent = (ab_sent["y_pred"][ar_mask] == ab_sent["y_test"][ar_mask]).mean()
        ar_acc_cat = (ab_cat["y_pred"][ar_mask] == ab_cat["y_test"][ar_mask]).mean()
        print(f"[AraBERT] Arabic-only test accuracy: "
              f"sentiment={ar_acc_sent:.4f}, category={ar_acc_cat:.4f} "
              f"(n={int(ar_mask.sum())})")

    print_evaluation("Sentiment — AraBERT (group-aware)",
                     ab_sent["y_test"], ab_sent["y_pred"], labels=SENTIMENT_LABELS)
    display_confusion(ab_sent["y_test"], ab_sent["y_pred"], SENTIMENT_LABELS,
                      title="Sentiment — AraBERT (group-aware)",
                      out_path=REPORTS_DIR / "confusion_arabert_sentiment.png")
    print_evaluation("Category — AraBERT (group-aware)",
                     ab_cat["y_test"], ab_cat["y_pred"], labels=CATEGORY_LABELS)
    display_confusion(ab_cat["y_test"], ab_cat["y_pred"], CATEGORY_LABELS,
                      title="Category — AraBERT (group-aware)",
                      out_path=REPORTS_DIR / "confusion_arabert_category.png")

    # 3) The other four models on the same split.
    print("\nComputing LR / NN / LSTM / BiLSTM on the same split …")
    _, X_tfidf = prepare_features(df)
    lr_sent = evaluate_group(X_tfidf, y_sent, groups, SENTIMENT_LABELS)
    lr_cat = evaluate_group(X_tfidf, y_cat, groups, CATEGORY_LABELS)

    w2v = load_embeddings()
    X_emb = document_embeddings(df, w2v)
    nn_sent = nn_evaluate_task(X_emb, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    nn_cat = nn_evaluate_task(X_emb, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    lstm_sent = lstm_evaluate_task(token_lists, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    lstm_cat = lstm_evaluate_task(token_lists, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    bi_sent = bilstm_evaluate_task(token_lists, y_sent, train_idx, test_idx, groups,
                                   SENTIMENT_LABELS, emb=w2v)
    bi_cat = bilstm_evaluate_task(token_lists, y_cat, train_idx, test_idx, groups,
                                  CATEGORY_LABELS, emb=w2v)

    # 4) 5-model comparison.
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (group-aware split)")
    print("=" * 60)
    _comparison_table("Sentiment", [
        ("Logistic Regression", lr_sent),
        ("Neural Network", nn_sent),
        ("LSTM", lstm_sent),
        ("BiLSTM + pretrained", bi_sent),
        ("AraBERT (frozen)", ab_sent),
    ])
    _comparison_table("Category", [
        ("Logistic Regression", lr_cat),
        ("Neural Network", nn_cat),
        ("LSTM", lstm_cat),
        ("BiLSTM + pretrained", bi_cat),
        ("AraBERT (frozen)", ab_cat),
    ])

    # Final AraBERT heads on all data + custom predictions.
    print("\nTraining final AraBERT heads on all data and saving …")
    sent_bundle = train_and_save_head(X_bert, y_sent, SENTIMENT_LABELS, ARABERT_SENTIMENT_PATH)
    cat_bundle = train_and_save_head(X_bert, y_cat, CATEGORY_LABELS, ARABERT_CATEGORY_PATH)
    print(f"Saved -> {ARABERT_SENTIMENT_PATH.name}, {ARABERT_CATEGORY_PATH.name}")

    print("\n" + "=" * 60)
    print("  Custom example predictions (AraBERT)")
    print("=" * 60)
    for text in CUSTOM_EXAMPLES:
        s_label, s_conf = predict_text(sent_bundle, encoder, text)
        c_label, c_conf = predict_text(cat_bundle, encoder, text)
        lang = "ar" if any("؀" <= ch <= "ۿ" for ch in text) else "en"
        print(f"\nText [{lang}]: {text}")
        print(f"  sentiment = {s_label:<8} ({s_conf:.3f})   "
              f"category = {c_label:<22} ({c_conf:.3f})")


if __name__ == "__main__":
    main()
