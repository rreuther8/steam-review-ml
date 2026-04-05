# Usage: Data Pipeline Order

This file is the runbook for getting processed data into the expected locations.

- **Interim:** cleaned table, then **raw-scale** train/val/test splits (`data/interim/`).
- **Processed:** modeling-ready Parquets with `_norm_*` columns; normalization params are saved in `artifacts/`.

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

The cleaned Parquet has **no** `review_word_count` or `review_length_chars` (those are added when splitting).

## 3) Split cleaned Parquet -> train/val/test (interim)

Uses `configs/split_reviews.json`.

```bash
python scripts/split_reviews.py configs/split_reviews.json
```

Current config targets:

- `data/interim/steam_reviews_cleaned_english_train.parquet`
- `data/interim/steam_reviews_cleaned_english_val.parquet`
- `data/interim/steam_reviews_cleaned_english_test.parquet`

After each row is assigned to train/val/test, the split step runs **`feature_engineering`** (`review_word_count`, `review_length_chars`) and then **`review_age_seconds`**: seconds from `timestamp_created` to the **maximum `timestamp_created` in the training split only** (two-pass stream for the reference; then a second pass writes outputs).

## 4) Normalize splits -> modeling Parquets (processed)

Fits caps/quantiles on **train** only; applies the same parameters to val and test.

Uses `configs/normalize_splits.json`.

```bash
python scripts/normalize_split_parquets.py configs/normalize_splits.json
```

Outputs:

- `data/processed/steam_reviews_cleaned_english_train_norm.parquet`
- `data/processed/steam_reviews_cleaned_english_val_norm.parquet`
- `data/processed/steam_reviews_cleaned_english_test_norm.parquet`
- `artifacts/normalization_params.json`

## 5) Quick output checks

```bash
ls -lh data/interim/steam_reviews_cleaned_english.parquet
ls -lh data/interim/steam_reviews_cleaned_english_train.parquet
ls -lh data/interim/steam_reviews_cleaned_english_val.parquet
ls -lh data/interim/steam_reviews_cleaned_english_test.parquet
ls -lh data/processed/steam_reviews_cleaned_english_*_norm.parquet
ls -lh artifacts/normalization_params.json
```

Optional: row counts for interim splits:

```bash
python -c "
import pandas as pd
for name, p in [
    ('train', 'data/interim/steam_reviews_cleaned_english_train.parquet'),
    ('val', 'data/interim/steam_reviews_cleaned_english_val.parquet'),
    ('test', 'data/interim/steam_reviews_cleaned_english_test.parquet'),
]:
    print(name, len(pd.read_parquet(p)))
"
```

## 6) Tabular baseline / modeling notebooks (optional)

Use the **`data/processed/..._norm.parquet`** files (raw columns are still present; `_norm_*` columns are added).

These live under **`notebooks/models/tabular/`** (numeric / engineered features — separate from recommender work).

- `notebooks/models/tabular/model_000_dumb_002.ipynb`
- `notebooks/models/tabular/model_001_linreg__votes_helpful.ipynb`
- `notebooks/models/tabular/model_002_logreg__recommended.ipynb`

## 7) Recommender artifacts (v1)

After processed train Parquet exists, build **game profiles** (train split, positive reviews only):

- Notebook: `notebooks/models/game_embeddings/recs_001_game_profiles.ipynb`
- Output: `artifacts/recs/game_profile_reviews.parquet` — one row per thumbs-up review (capped per game); input for **per-review embed + mean** in `recs_002`.

**Dense game vectors** (TensorFlow + TensorFlow Hub; see `pyproject.toml` `[recs]` extra and `recs_002` notebook):

- Notebook: `notebooks/models/game_embeddings/recs_002_embed_game_profiles.ipynb`
- Outputs: `artifacts/recs/game_profile_embeddings.npz`, `game_profile_embedding_index.parquet`, `game_profile_embedding_meta.json`

**Query + top‑K (smoke test / demo)** — same TF Hub model as `recs_002` (URL read from `game_profile_embedding_meta.json`):

- Notebook: `notebooks/models/query_embeddings/recs_003_query_retrieve.ipynb`

See `docs/recommender_transition_plan.md` for the full v1 path.

---

## Order Summary

1. `python scripts/clean_reviews.py configs/clean_reviews.json`
2. `python scripts/split_reviews.py configs/split_reviews.json`
3. `python scripts/normalize_split_parquets.py configs/normalize_splits.json`
4. (Optional) run tabular modeling notebooks
5. (Recommender v1) run `notebooks/models/game_embeddings/recs_001_game_profiles.ipynb`
6. (Optional) install recs extras and run `notebooks/models/game_embeddings/recs_002_embed_game_profiles.ipynb`
7. (Optional) run `notebooks/models/query_embeddings/recs_003_query_retrieve.ipynb` after `recs_002` artifacts exist



## Full run

```
python scripts/clean_reviews.py configs/clean_reviews.json && \
python scripts/split_reviews.py configs/split_reviews.json && \
python scripts/normalize_split_parquets.py configs/normalize_splits.json
```