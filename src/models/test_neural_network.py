"""Train, evaluate, and demo the Feed-Forward Neural Network classifiers.

Steps:
    1. Build Word2Vec document embeddings for the corpus.
    2. Evaluate the NN on the same group-aware split used by Logistic Regression.
    3. Print metrics + classification report and save confusion matrices.
    4. Compare NN vs Logistic Regression (group-aware) side by side.
    5. Train final models on all data, save .pt files, and predict custom inputs.

Usage:
    python src/models/test_neural_network.py
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
from src.models.logistic_regression import (  # noqa: E402  (reused, not modified)
    CATEGORY_LABELS,
    SENTIMENT_LABELS,
    build_group_keys,
    evaluate_group,
    prepare_features,
)
from src.models.neural_network import (  # noqa: E402
    NN_CATEGORY_PATH,
    NN_SENTIMENT_PATH,
    document_embeddings,
    evaluate_task,
    group_aware_split,
    load_dataframe,
    load_embeddings,
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


def _macro_f1(result: dict) -> float:
    return compute_metrics(result["y_test"], result["y_pred"],
                           labels=result["labels"], average="macro")["f1"]


def _report_task(title: str, nn_res: dict, lr_res: dict, labels, fig: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print("Group-aware comparison (accuracy / macro-F1):")
    print(f"  Neural Network      : acc={nn_res['accuracy']:.4f}  macroF1={_macro_f1(nn_res):.4f}")
    print(f"  Logistic Regression : acc={lr_res['accuracy']:.4f}  macroF1={_macro_f1(lr_res):.4f}")

    print_evaluation(f"{title} — Neural Network (group-aware)",
                     nn_res["y_test"], nn_res["y_pred"], labels=labels)
    display_confusion(nn_res["y_test"], nn_res["y_pred"], labels,
                      title=f"{title} — NN (group-aware)", out_path=REPORTS_DIR / fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 1) Embeddings + document features.
    emb = load_embeddings()
    df = load_dataframe()
    X = document_embeddings(df, emb)
    groups = build_group_keys(df)
    train_idx, test_idx = group_aware_split(groups)
    y_sent = df["sentiment"].to_numpy()
    y_cat = df["category"].to_numpy()

    print(f"Embedding dim: {X.shape[1]} | docs: {X.shape[0]} | "
          f"template groups: {len(set(groups))}")
    print(f"Group split -> train={len(train_idx)}, test={len(test_idx)}")

    # 2) NN evaluation on the group-aware split.
    nn_sent = evaluate_task(X, y_sent, train_idx, test_idx, SENTIMENT_LABELS)
    nn_cat = evaluate_task(X, y_cat, train_idx, test_idx, CATEGORY_LABELS)

    # 4) Logistic Regression on the SAME split (same groups + params).
    _, X_tfidf = prepare_features(df)
    lr_sent = evaluate_group(X_tfidf, y_sent, groups, SENTIMENT_LABELS)
    lr_cat = evaluate_group(X_tfidf, y_cat, groups, CATEGORY_LABELS)

    _report_task("Task 1 — Sentiment Classification",
                 nn_sent, lr_sent, SENTIMENT_LABELS, "confusion_nn_sentiment.png")
    _report_task("Task 2 — Category Classification",
                 nn_cat, lr_cat, CATEGORY_LABELS, "confusion_nn_category.png")

    # 5) Final models trained on all data + saved, then custom predictions.
    print("\n" + "=" * 70)
    print("  Training final models on all data and saving (.pt)")
    print("=" * 70)
    sent_bundle = train_final_and_save(X, y_sent, SENTIMENT_LABELS, NN_SENTIMENT_PATH)
    cat_bundle = train_final_and_save(X, y_cat, CATEGORY_LABELS, NN_CATEGORY_PATH)
    print(f"Saved -> {NN_SENTIMENT_PATH.name}, {NN_CATEGORY_PATH.name}")

    print("\n" + "=" * 70)
    print("  Custom example predictions (Neural Network)")
    print("=" * 70)
    for text in CUSTOM_EXAMPLES:
        s_label, s_conf = predict_text(sent_bundle, emb, text)
        c_label, c_conf = predict_text(cat_bundle, emb, text)
        lang = "ar" if any("؀" <= ch <= "ۿ" for ch in text) else "en"
        print(f"\nText [{lang}]: {text}")
        print(f"  sentiment = {s_label:<8} ({s_conf:.3f})   "
              f"category = {c_label:<22} ({c_conf:.3f})")


if __name__ == "__main__":
    main()
