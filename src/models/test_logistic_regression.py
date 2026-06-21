"""Train, evaluate, and demo the Logistic Regression baselines.

Steps:
    1. Train on shared TF-IDF features and evaluate each task with BOTH a random
       split and a group-aware split (template skeletons held out).
    2. Print the random-vs-group comparison + full metrics for the group split.
    3. Display confusion matrices (text + saved PNG) for the group-aware split.
    4. Predict sentiment + category + confidence on custom Arabic/English inputs,
       printing BEFORE (plain LR) and AFTER (synonym expansion + category rules).

Usage:
    python src/models/test_logistic_regression.py
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
    CATEGORY_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_PATH,
    detect_language,
    predict_category,
    predict_sentiment,
    train_and_evaluate_all,
)

REPORTS_DIR = ROOT / "reports"

# Custom examples (Arabic + English) for the prediction demo.
CUSTOM_EXAMPLES = [
    "الدكتور رائع وشرح المادة ممتاز",
    "لازم نعمل اضراب ضد الرسوم",
    "التسجيل هالفصل كان فوضى",
    "The professor explains very well",
    "Students are planning a strike",
    "Registration was very confusing",
]


def _macro_f1(result: dict) -> float:
    m = compute_metrics(result["y_test"], result["y_pred"],
                        labels=result["labels"], average="macro")
    return m["f1"]


def _evaluate_task(title: str, eval_block: dict, labels, fig_name: str) -> None:
    rnd = eval_block["random"]
    grp = eval_block["group"]

    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    # 2) Random vs group-aware comparison.
    print("Split comparison (accuracy / macro-F1):")
    print(f"  Random split      : acc={rnd['accuracy']:.4f}  macroF1={_macro_f1(rnd):.4f}"
          f"   (train={rnd['n_train']}, test={rnd['n_test']})")
    print(f"  Group-aware split : acc={grp['accuracy']:.4f}  macroF1={_macro_f1(grp):.4f}"
          f"   (train={grp['n_train']}, test={grp['n_test']})")
    print(f"  Group K-Fold acc  : {eval_block['gkf_mean']:.4f} ± {eval_block['gkf_std']:.4f}")
    print("  -> the random split is optimistic; the group-aware numbers are the "
          "honest estimate.")

    # Full metrics + confusion for the group-aware (realistic) split.
    print_evaluation(f"{title} — group-aware split",
                     grp["y_test"], grp["y_pred"], labels=labels)
    display_confusion(grp["y_test"], grp["y_pred"], labels,
                      title=f"{title} (group-aware)",
                      out_path=REPORTS_DIR / fig_name)


def _run_custom_examples(sentiment_bundle: dict, category_bundle: dict) -> None:
    print("\n" + "=" * 70)
    print("  Custom example predictions — BEFORE vs AFTER")
    print("  BEFORE = plain Logistic Regression")
    print("  AFTER  = + Arabic positive-synonym expansion + category rule boost")
    print("=" * 70)
    for text in CUSTOM_EXAMPLES:
        lang = detect_language(text)

        b_sent, b_sc = predict_sentiment(sentiment_bundle, text, lang, expand=False)
        b_cat, b_cc = predict_category(category_bundle, text, lang, use_rules=False)
        a_sent, a_sc = predict_sentiment(sentiment_bundle, text, lang, expand=True)
        a_cat, a_cc = predict_category(category_bundle, text, lang, use_rules=True)

        print(f"\nText [{lang}]: {text}")
        print(f"  BEFORE  sentiment={b_sent:<8} ({b_sc:.3f})   "
              f"category={b_cat:<22} ({b_cc:.3f})")
        print(f"  AFTER   sentiment={a_sent:<8} ({a_sc:.3f})   "
              f"category={a_cat:<22} ({a_cc:.3f})")


def main() -> None:
    # Ensure Arabic prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = train_and_evaluate_all(save=True)
    print(f"Shared TF-IDF: vocab={out['vocab_size']}, docs={out['n_docs']}, "
          f"template groups={out['n_groups']}")
    print(f"Saved models -> {SENTIMENT_PATH.name}, {CATEGORY_PATH.name}")

    _evaluate_task("Task 1 — Sentiment Classification",
                   out["sentiment_eval"], SENTIMENT_LABELS, "confusion_sentiment.png")
    _evaluate_task("Task 2 — Category Classification",
                   out["category_eval"], CATEGORY_LABELS, "confusion_category.png")

    _run_custom_examples(out["sentiment_bundle"], out["category_bundle"])


if __name__ == "__main__":
    main()
