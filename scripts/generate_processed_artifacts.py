"""Regenerate processed summaries for product-review sentiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocessing import create_target, load_reviews, summarize_reviews


def main() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews()
    clean = create_target(df)
    summary = summarize_reviews(df)
    summary["source_rows"] = int(len(df))
    summary["columns"] = int(df.shape[1])

    (output_dir / "key_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "key_metrics.csv", index=False)

    rating_distribution = clean["reviews.rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="review_count")
    rating_distribution.to_csv(output_dir / "rating_distribution.csv", index=False)

    print("Processed artifacts regenerated.")


if __name__ == "__main__":
    main()
