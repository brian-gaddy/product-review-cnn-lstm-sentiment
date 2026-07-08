# Model Card: Product Review CNN-LSTM Sentiment Classifier

## Model Purpose

The model classifies product-review text into good or bad review categories using a CNN-LSTM neural network.

## Intended Use

- Review sentiment triage
- Customer experience analytics
- Product-quality monitoring
- Portfolio demonstration of NLP sequence modeling

## Not Intended For

The model should not be used as the only source of truth for customer refunds, account actions, product removals, or employee performance decisions.

## Dataset

The dataset contains 71,044 source records and 71,008 usable text/rating records. The target is derived from `reviews.rating`, where ratings below 4 are classified as bad reviews.

## Model Type

CNN-LSTM hybrid neural network for binary text classification.

## Inputs

The primary model input is `reviews.text`, tokenized into integer sequences with a maximum vocabulary size of 20,000 and padded to 150 tokens.

## Output

The model outputs probabilities for two classes: good review and bad review.

## Evaluation

The notebook reports test accuracy of **92.04%** and test loss of 0.2578.

## Limitations

The target is rating-derived, the bad-review class is a minority class, and accuracy alone may not reflect business performance. A production workflow should add precision, recall, F1, confusion matrix, drift monitoring, and human review.
