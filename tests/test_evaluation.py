import pandas as pd

from src.evaluation import evaluate_predictions
from scripts.generate_evaluation_artifacts import normalize_labels


def test_evaluate_predictions_returns_binary_matrix_and_report():
    matrix, report = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1])
    assert matrix.shape == (2, 2)
    assert matrix.loc["Good Review", "Good Review"] == 1
    assert matrix.loc["Bad Review", "Bad Review"] == 2
    assert {"precision", "recall", "f1-score", "support"}.issubset(report.columns)


def test_normalize_labels_supports_readable_labels():
    labels = pd.Series(["Good Review", "Bad Review", "Good Review"])
    assert normalize_labels(labels).tolist() == [0, 1, 0]
