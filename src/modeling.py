"""CNN-LSTM model and tokenizer utilities for product-review classification."""

from __future__ import annotations

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150


def build_cnn_lstm_model(max_words: int = MAX_NB_WORDS, max_sequence_length: int = MAX_SEQUENCE_LENGTH):
    """Build the CNN-LSTM hybrid architecture when TensorFlow is available."""
    try:
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, LSTM, MaxPooling1D
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise ImportError("TensorFlow is required to build the CNN-LSTM model.") from exc

    model = Sequential(
        [
            Embedding(input_dim=max_words, output_dim=50, input_length=max_sequence_length),
            Conv1D(filters=64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=5),
            Dropout(0.2),
            Conv1D(filters=64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=5),
            Dropout(0.2),
            LSTM(64),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def rating_to_label(is_bad_review: bool) -> str:
    """Convert the Boolean model target into a readable label."""
    return "Bad Review" if bool(is_bad_review) else "Good Review"
