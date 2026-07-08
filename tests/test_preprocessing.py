import pandas as pd
import pytest

from src.modeling import rating_to_label
from src.preprocessing import create_target, summarize_reviews, validate_review_columns


def sample_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reviews.text": ["Great product", "Poor quality", "Average"],
            "reviews.rating": [5, 1, 3],
        }
    )


def test_validate_review_columns_accepts_required_columns():
    assert validate_review_columns(sample_reviews())


def test_validate_review_columns_rejects_missing_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_review_columns(pd.DataFrame({"reviews.text": ["ok"]}))


def test_create_target_marks_ratings_below_four_as_bad():
    df = create_target(sample_reviews())
    assert df["target_bad_review"].tolist() == [False, True, True]


def test_summarize_reviews_returns_expected_counts():
    metrics = summarize_reviews(sample_reviews())
    assert metrics["reviews"] == 3
    assert metrics["bad_review_count"] == 2
    assert metrics["good_review_count"] == 1


def test_rating_to_label_returns_readable_labels():
    assert rating_to_label(True) == "Bad Review"
    assert rating_to_label(False) == "Good Review"
