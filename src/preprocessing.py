"""Data loading and preprocessing utilities for product-review sentiment modeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_COLUMN = "reviews.text"
RATING_COLUMN = "reviews.rating"
TARGET_COLUMN = "target_bad_review"
RAW_DATA_PATH = Path("data/raw/product_reviews_full_dataset.csv")


def load_reviews(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load product reviews from CSV."""
    return pd.read_csv(path)


def validate_review_columns(df: pd.DataFrame) -> bool:
    """Validate that required text and rating columns exist."""
    missing = [column for column in [TEXT_COLUMN, RATING_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target where True means rating below 4."""
    validate_review_columns(df)
    clean = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()
    clean[TARGET_COLUMN] = clean[RATING_COLUMN] < 4
    return clean


def summarize_reviews(df: pd.DataFrame) -> dict[str, int | float]:
    """Return core review and sentiment distribution metrics."""
    clean = create_target(df)
    bad_count = int(clean[TARGET_COLUMN].sum())
    total = int(len(clean))
    return {
        "reviews": total,
        "bad_review_count": bad_count,
        "good_review_count": int(total - bad_count),
        "bad_review_rate": float(bad_count / total),
        "good_review_rate": float(1 - bad_count / total),
    }
