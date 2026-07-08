"""Streamlit dashboard for product-review CNN-LSTM sentiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Review CNN-LSTM Sentiment", layout="wide")
st.title("Product Review Sentiment Classification with CNN-LSTM")
st.caption("Hybrid deep-learning workflow for classifying product reviews as good or bad.")

with Path("data/processed/key_metrics.json").open("r", encoding="utf-8") as f:
    metrics = json.load(f)

target_distribution = pd.read_csv("data/processed/target_distribution.csv")
rating_distribution = pd.read_csv("data/processed/rating_distribution.csv")
model_results = pd.read_csv("data/processed/model_results.csv")
history = pd.read_csv("data/processed/training_history.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Modeling Reviews", f"{metrics['modeling_rows']:,}")
c2.metric("Bad Review Rate", f"{metrics['bad_review_rate']:.2%}")
c3.metric("Test Accuracy", f"{metrics['cnn_lstm_test_accuracy']:.2%}")
c4.metric("Vocabulary Limit", f"{metrics['max_words']:,}")

st.subheader("Target Distribution")
st.bar_chart(target_distribution.set_index("target_label")["review_count"])

st.subheader("Rating Distribution")
st.bar_chart(rating_distribution.set_index("rating")["review_count"])

st.subheader("Model Result")
st.dataframe(model_results, use_container_width=True)

st.subheader("Training History")
st.line_chart(history.set_index("epoch")[["train_accuracy", "validation_accuracy"]])
st.line_chart(history.set_index("epoch")[["train_loss", "validation_loss"]])

st.subheader("Local Inference")
st.write(
    "The repository does not commit trained model weights. After training locally, use the notebook "
    "or extend this app to load a saved `.keras` model and tokenizer for live review classification."
)
review_text = st.text_area("Paste a product review for workflow demonstration")
if review_text:
    st.info("Live prediction requires a locally saved model and tokenizer. This input is ready for preprocessing.")
