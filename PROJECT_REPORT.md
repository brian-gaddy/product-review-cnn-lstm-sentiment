# Project Report: Product Review CNN-LSTM Sentiment Classification

## Executive Summary

This project classifies product reviews as good or bad using a CNN-LSTM hybrid neural network. The dataset contains 71,044 source rows and 71,008 usable modeling rows after removing records without review text or rating. The observed bad-review rate is 13.94%.

The trained CNN-LSTM workflow achieved **92.04% test accuracy** with a test loss of 0.2578.

## Business Objective

The objective is to help product, customer experience, and marketing teams classify large volumes of review text into good and bad sentiment groups. This supports review monitoring, service recovery, product-quality triage, and trend analysis.

## Data Preparation

The target is derived from `reviews.rating`. Reviews with ratings below 4 are labeled as bad reviews; ratings 4 and 5 are labeled as good reviews. The text feature is `reviews.text`. Records missing review text or rating are excluded from model training.

## Model Design

The architecture uses a 20,000-word tokenizer vocabulary and pads review sequences to 150 tokens. The model includes an embedding layer, two Conv1D and MaxPooling blocks with dropout, an LSTM layer, and a two-neuron softmax output layer.

## Evaluation

The notebook reports a test accuracy of **92.04%** and a test loss of 0.2578. Training accuracy improved across five epochs, while validation accuracy remained near 91%, indicating useful signal with some overfitting risk.

## Recommendations

- Add precision, recall, F1, and confusion matrix reporting.
- Compare the CNN-LSTM model against logistic regression, TF-IDF, random forest, and transformer-based baselines.
- Add class-weighting or threshold tuning because bad reviews represent only 13.94% of usable records.
- Save the trained model and tokenizer for reproducible inference.
- Deploy the Streamlit app with a saved model artifact.

## Limitations

The target is derived from rating rather than manually labeled sentiment. The model may learn rating-specific language patterns but still requires validation before production use. Customer sentiment should be reviewed with human oversight when used for service recovery or business decisions.
