# Usage: Data Pipeline Order

This file is the runbook for getting processed data into the expected locations.

## 0) Run from repo root

```bash
cd /home/ryanr/workspace/steam_recommendations
```

## 1) Install package (editable)

Run this once per new environment (or after dependency changes):

```bash
pip install -e .
```

## 2) Clean raw CSV -> cleaned Parquet

Uses `configs/clean_reviews.json`.

```bash
python scripts/clean_reviews.py configs/clean_reviews.json
```

Expected output path is controlled by `output_path` in the config.

Current config target:

- `data/interim/steam_reviews_cleaned_english.parquet`

## 3) Split cleaned Parquet -> train/val/test Parquet

Uses `configs/split_reviews.json`.

```bash
python scripts/split_reviews.py configs/split_reviews.json
```

Current config targets:

- `data/processed/steam_reviews_cleaned_english_train.parquet`
- `data/processed/steam_reviews_cleaned_english_val.parquet`
- `data/processed/steam_reviews_cleaned_english_test.parquet`

## 4) Quick output checks

Check that expected files exist:

```bash
ls -lh data/interim/steam_reviews_cleaned_english.parquet
ls -lh data/processed/steam_reviews_cleaned_english_train.parquet
ls -lh data/processed/steam_reviews_cleaned_english_val.parquet
ls -lh data/processed/steam_reviews_cleaned_english_test.parquet
```

Optional: check row counts quickly:

```bash
python - <<'PY'
import pandas as pd
paths = {
    "train": "data/processed/steam_reviews_cleaned_english_train.parquet",
    "val": "data/processed/steam_reviews_cleaned_english_val.parquet",
    "test": "data/processed/steam_reviews_cleaned_english_test.parquet",
}
for name, p in paths.items():
    print(name, len(pd.read_parquet(p)))
PY
```

## 5) Baseline notebook (optional)

To run dumb baselines after data is prepared:

- Open `notebooks/models/model_000_dumb.ipynb`
- Run all cells

---

## Order Summary

1. `python scripts/clean_reviews.py configs/clean_reviews.json`
2. `python scripts/split_reviews.py configs/split_reviews.json`
3. (Optional) run `notebooks/models/model_000_dumb.ipynb`

