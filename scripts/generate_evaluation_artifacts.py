"""Generate exact confusion matrix and classification report from saved predictions.

Expected CSV columns:
- true_label
- predicted_label

Labels may be Boolean, 0/1, or the strings `Good Review` and `Bad Review`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation import save_evaluation_artifacts


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize supported sentiment labels to 0=Good Review and 1=Bad Review."""
    mapping = {
        False: 0,
        True: 1,
        "False": 0,
        "True": 1,
        "Good Review": 0,
        "Bad Review": 1,
        "good_review": 0,
        "bad_review": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
    }
    normalized = series.map(mapping)
    if normalized.isna().any():
        unknown = sorted(series[normalized.isna()].astype(str).unique())
        raise ValueError(f"Unsupported labels: {unknown}")
    return normalized.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV containing true_label and predicted_label")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    required = {"true_label", "predicted_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_true = normalize_labels(predictions["true_label"])
    y_pred = normalize_labels(predictions["predicted_label"])
    save_evaluation_artifacts(y_true, y_pred)
    print("Saved confusion_matrix.csv, classification_report.csv, confusion_matrix.png, and per_class_metrics.png")


if __name__ == "__main__":
    main()
