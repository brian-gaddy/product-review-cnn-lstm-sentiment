"""Evaluation utilities for binary product-review sentiment classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Good Review", "Bad Review"]


def evaluate_predictions(y_true, y_pred) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return confusion-matrix and precision/recall/F1 report tables."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_df = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    matrix_df.index.name = "true_label"

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="class")
    return matrix_df, report_df


def save_evaluation_artifacts(y_true, y_pred, output_dir="data/processed", figures_dir="figures") -> None:
    """Save exact evaluation CSVs and a confusion-matrix figure from model predictions."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix_df, report_df = evaluate_predictions(y_true, y_pred)
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")
    report_df.to_csv(output_dir / "classification_report.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix_df.values, cmap="Blues")
    ax.set_title("CNN-LSTM Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix_df.iloc[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    class_rows = report_df[report_df["class"].isin(CLASS_NAMES)]
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, class_rows[metric], width, label=metric)
    ax.set_title("Per-Class Precision, Recall, and F1-Score")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, class_rows["class"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_metrics.png", dpi=180)
    plt.close(fig)
