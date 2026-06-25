"""Confusion-matrix computation, text display, and figure export.

Uses a non-interactive matplotlib backend so it works headless (terminal /
servers). ``display_confusion`` prints a labeled text table and, when an output
path is given, also saves a PNG heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402


def compute_confusion(y_true, y_pred, labels):
    """Return the confusion matrix (rows=true, cols=pred) for ``labels``."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def confusion_as_dataframe(cm, labels) -> pd.DataFrame:
    """Wrap a confusion matrix in a labeled DataFrame for readable printing."""
    return pd.DataFrame(
        cm,
        index=[f"true:{lbl}" for lbl in labels],
        columns=[f"pred:{lbl}" for lbl in labels],
    )


def save_confusion_plot(cm, labels, title: str, out_path: Path) -> Path:
    """Render the confusion matrix as a PNG heatmap and return its path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def display_confusion(
    y_true,
    y_pred,
    labels,
    title: str,
    out_path: Path | None = None,
):
    """Print the confusion matrix as text and optionally save a PNG.

    Returns the raw confusion-matrix array.
    """
    cm = compute_confusion(y_true, y_pred, labels)
    print(f"\nConfusion Matrix — {title}")
    print(confusion_as_dataframe(cm, labels).to_string())
    if out_path is not None:
        saved = save_confusion_plot(cm, labels, title, out_path)
        print(f"(saved heatmap -> {saved})")
    return cm
