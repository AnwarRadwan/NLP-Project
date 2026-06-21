"""Classification evaluation metrics.

Thin, well-documented wrappers around scikit-learn so every model in the
project reports metrics the same way: accuracy, precision, recall, F1 (macro
and weighted), and a full per-class classification report.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, labels=None, average: str = "macro") -> dict:
    """Return a dict with accuracy and (precision, recall, f1) for ``average``.

    ``zero_division=0`` keeps the call safe if a class is never predicted.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average": average,
    }


def get_classification_report(y_true, y_pred, labels=None) -> str:
    """Full per-class precision/recall/F1/support report (4 decimals)."""
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=4
    )


def print_evaluation(title: str, y_true, y_pred, labels=None) -> dict:
    """Print accuracy + macro/weighted P/R/F1 and the classification report.

    Returns the macro metrics dict for optional further use.
    """
    macro = compute_metrics(y_true, y_pred, labels=labels, average="macro")
    weighted = compute_metrics(y_true, y_pred, labels=labels, average="weighted")

    print(f"\n=== {title} ===")
    print(f"Accuracy            : {macro['accuracy']:.4f}")
    print(f"Precision (macro)   : {macro['precision']:.4f}")
    print(f"Recall    (macro)   : {macro['recall']:.4f}")
    print(f"F1-score  (macro)   : {macro['f1']:.4f}")
    print(f"Precision (weighted): {weighted['precision']:.4f}")
    print(f"Recall    (weighted): {weighted['recall']:.4f}")
    print(f"F1-score  (weighted): {weighted['f1']:.4f}")
    print("\nClassification Report:")
    print(get_classification_report(y_true, y_pred, labels=labels))
    return macro
