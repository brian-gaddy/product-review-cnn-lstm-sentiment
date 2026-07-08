# Data Access

The full raw product-review CSV is intentionally excluded from this repository.

`product_reviews_full_dataset.csv` is approximately 94.77 MB. Although this is below GitHub's 100 MB hard file limit, it is well above GitHub's recommended 50 MB maximum for normal Git repositories. Keeping the raw CSV out of Git makes cloning and CI faster and avoids repository bloat.

## Expected local path

Place the full CSV at:

```text
data/raw/product_reviews_full_dataset.csv
```

The preprocessing and evaluation scripts expect that path by default.

## Data schema used by this project

The modeling workflow requires at minimum:

- `reviews.text` — review text used as the NLP feature
- `reviews.rating` — numeric review rating used to derive the binary target

The target definition is:

- rating below 4 -> bad review
- rating 4 or 5 -> good review

## Reproducing the analysis

After placing the CSV at the expected local path:

```bash
pip install -r requirements.txt
python scripts/generate_processed_artifacts.py
```

Then run the notebook to train the CNN-LSTM model and generate predictions. The evaluation utility can save the confusion matrix and precision/recall/F1 report from those predictions.

## Git policy

The full raw CSV is ignored by `.gitignore` and should not be committed to normal Git history. Processed summaries and lightweight evaluation artifacts may be committed.
