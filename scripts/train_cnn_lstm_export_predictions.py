"""Train the CNN-LSTM model and export exact test predictions.

This script reproduces the notebook workflow using the local raw CSV, then saves:
- data/processed/predictions.csv
- data/processed/confusion_matrix.csv
- data/processed/classification_report.csv
- figures/confusion_matrix.png
- figures/per_class_metrics.png

The raw dataset is intentionally not committed to Git. Place it at:
    data/raw/product_reviews_full_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import save_evaluation_artifacts
from src.modeling import MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, build_cnn_lstm_model
from src.preprocessing import RATING_COLUMN, RAW_DATA_PATH, TARGET_COLUMN, TEXT_COLUMN, create_target, load_reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAW_DATA_PATH), help="Path to raw review CSV")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {data_path}. See DATA_ACCESS.md and place the CSV at the expected path."
        )

    # TensorFlow imports are kept inside main so lightweight CI tests do not require TensorFlow.
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = create_target(load_reviews(data_path))
    texts = df[TEXT_COLUMN].astype(str).values
    y = df[TARGET_COLUMN].astype(int).values

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train_text)

    x_train = tokenizer.texts_to_sequences(x_train_text)
    x_test = tokenizer.texts_to_sequences(x_test_text)

    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    model = build_cnn_lstm_model()
    model.fit(
        x_train,
        y_train_cat,
        validation_split=0.20,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    predictions = pd.DataFrame(
        {
            "review_text": x_test_text,
            "true_label": y_test,
            "predicted_label": y_pred,
            "prob_good_review": probabilities[:, 0],
            "prob_bad_review": probabilities[:, 1],
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    save_evaluation_artifacts(y_test, y_pred)

    metrics = pd.DataFrame(
        [
            {
                "model": "CNN-LSTM hybrid",
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_words": MAX_NB_WORDS,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "test_rows": int(len(y_test)),
            }
        ]
    )
    metrics.to_csv(output_dir / "model_results_reproduced.csv", index=False)

    print(f"Saved predictions and evaluation artifacts. Test accuracy: {test_accuracy:.4f}; loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
