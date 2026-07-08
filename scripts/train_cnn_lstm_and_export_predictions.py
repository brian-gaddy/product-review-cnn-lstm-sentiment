"""Train CNN-LSTM review classifier and export exact test predictions.

This script reruns the repository's full text-classification workflow using the local raw CSV,
then writes:
- data/processed/predictions.csv
- data/processed/confusion_matrix.csv
- data/processed/classification_report.csv
- figures/confusion_matrix.png
- figures/per_class_metrics.png

The full raw CSV is intentionally not stored in Git. Place it at:
    data/raw/product_reviews_full_dataset.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from src.evaluation import save_evaluation_artifacts

RAW_PATH = Path("data/raw/product_reviews_full_dataset.csv")
PREDICTIONS_PATH = Path("data/processed/predictions.csv")
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150
RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Set common random seeds for more reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def build_model() -> Sequential:
    """Build the CNN-LSTM hybrid architecture used in the notebook."""
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=50, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Follow DATA_ACCESS.md and place the full raw CSV at this path."
        )

    set_seeds()
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df["target"] = df["reviews.rating"] < 4
    df = df.dropna(subset=["reviews.text", "reviews.rating"]).copy()

    X = df["reviews.text"].astype(str)
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_train)

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=MAX_SEQUENCE_LENGTH,
    )
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=MAX_SEQUENCE_LENGTH,
    )

    y_train_one_hot = to_categorical(y_train, num_classes=2)
    y_test_one_hot = to_categorical(y_test, num_classes=2)

    model = build_model()
    model.fit(
        x_train,
        y_train_one_hot,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    predictions = pd.DataFrame(
        {
            "review_index": X_test.index,
            "true_label": y_test.to_numpy(),
            "predicted_label": y_pred,
            "prob_good_review": probabilities[:, 0],
            "prob_bad_review": probabilities[:, 1],
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    save_evaluation_artifacts(y_test.to_numpy(), y_pred)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print("Saved confusion matrix and precision/recall/F1 artifacts.")


if __name__ == "__main__":
    main()
