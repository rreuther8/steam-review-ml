# Normalization: plan and follow-up checks

## Current status

- Deciding on the transformation(s) for numeric features (log vs cap vs standardize, and which columns).
- No implementation of the "quick checks" below yet—those will go in once the transformation is chosen.

## Plan

1. **Decide transformation** – Use eda_006 and the data_filtering spec to pick, per column or group:
   - Long-tailed counts (votes, playtime): log(1+x), cap at a percentile, or both.
   - Other numerics (e.g. author.num_games_owned, author.num_reviews, review_length_chars): same options if skewed.
   - Scaling: which columns get StandardScaler (or similar) for linear/distance models, if any.
2. **Add a "Quick checks" section** – After the chosen transform is applied (in a notebook or script), run the checks below to validate that the transform preserves or improves signal and doesn’t break assumptions.

## Quick checks (to add later)

Run these **after** applying the chosen transformation to the modeling dataframe (filtered + feature-selected). They don’t need to be implemented until the normalization method is fixed.

- **Feature vs target (from eda_004)**  
  Re-run the same "numeric feature vs recommended" and "vs helpfulness" views (e.g. boxplots/violins) on the **transformed** features. Confirm separation is preserved or improved; if it gets worse, revisit the transform for that column.

- **Correlation (from eda_005)**  
  Correlation heatmap on the **transformed** numerics. Use it to spot redundancy and multicollinearity before modeling (and to decide regularization or dropping one of a pair).

- **Outliers (from eda_005)**  
  Use the same outlier logic (e.g. z-score or percentile) on transformed columns to confirm that capping/trimming choices are consistent and that no new issues appear.

- **review_length_chars**  
  If this (or other count-like derived numerics) is included in the model, check its distribution after any log/cap and include it in the same correlation and feature-vs-target checks.

- **Target transform (if regression on votes_helpful)**  
  If the target is `votes_helpful` (regression), consider log(1 + votes_helpful) and document it in the same place as the feature transforms.

## Where this lives

- **Decisions and spec:** [data_filtering.md](data_filtering.md) Section 4 (Normalization and feature transforms).
- **Exploration and plots:** [notebooks/eda/eda_006_normalization.ipynb](../notebooks/eda/eda_006_normalization.ipynb).
- **Quick-checks section:** To be added in eda_006 (or a short "Validation" subsection) once the normalization method is decided.


# Normalization: chosen transforms and pipeline plan

## 1. Chosen transforms (per column)

Decisions from EDA (see [notebooks/eda/eda_006_normalization_002.ipynb](../notebooks/eda/eda_006_normalization_002.ipynb)):

| Column | Transform |
|--------|-----------|
| `votes_helpful` | Cap at 99th percentile, then log1p |
| `author.playtime_last_two_weeks` | Cap at 99th percentile, then log1p |
| `votes_funny` | log1p only |
| `author.playtime_at_review` | log1p only |
| `author.num_games_owned` | log1p only |
| `author.num_reviews` | log1p only |
| `review_word_count` | log1p only |

- **Cap-then-log:** fit the cap (e.g. 99th percentile) on training data; apply `min(x, cap)` then `log1p`.
- **Log only:** `log1p(x)` with no capping.

Same behavior is implemented in three equivalent ways in the ref notebook: manual (NumPy), scikit-learn (`ColumnTransformer` + `CapThenLogTransformer`), and TensorFlow (`SteamNormalizer` layer).

---

## 2. Pipeline: where normalization runs

Normalization is done **once** as a data-preparation step, not per model.

1. **Fit the normalizer** on a large portion of training data (e.g. full train or a large sample).
2. **Save the normalizer** (e.g. fitted caps + column list, or a small TF/Keras model that applies the same logic). This artifact is needed later only for **inference** on new raw inputs.
3. **Transform** train, validation, and test sets with the fitted normalizer.
4. **Save the normalized datasets** in final form (e.g. Parquet or CSV). All downstream models use these pre-normalized datasets for training and evaluation.

Effects:

- All models (NN, logistic regression, linear regression, random baseline) consume the same pre-normalized data; no need to run the normalizer again during training.
- Training code stays simple: load pre-normalized data and fit.
- One place to re-run if you ever change the normalization (e.g. different percentile or columns): re-fit normalizer, re-transform, overwrite or version the saved datasets.

**CLI (repo):** From the project root, with `src` on `PYTHONPATH` (or an editable install), run:

`PYTHONPATH=src python scripts/normalize_split_parquets.py configs/normalize_splits.json`

The sample config [`configs/normalize_splits.json`](../../configs/normalize_splits.json) reads the split Parquets from [`configs/split_reviews.json`](../../configs/split_reviews.json) and writes `*_norm.parquet` siblings plus [`data/processed/normalization_params.json`](../../data/processed/normalization_params.json) (paths are editable in the normalize config). Optional key `normalization_rules` overrides the default rule table from code. Normalized column names are `_norm_<source>` with `.` replaced by `__` in `<source>`.

---

## 3. Inference (e.g. in the app)

New inputs are raw, so they must be normalized before calling any model.

- **Load the saved normalizer** (or its parameters: caps + column list) once at startup or on first request.
- For each new sample: run the normalizer on the raw feature vector, then pass the normalized vector to the model (NN, logreg, or linear).

So we keep two things from the pipeline:

- **Pre-normalized datasets** — used for training and validation.
- **Saved normalizer** — used only at inference to transform new raw data.

---

## 4. References

- **Clean ref (transforms + manual/sklearn/TF):** [notebooks/eda/eda_006_normalization_002.ipynb](../notebooks/eda/eda_006_normalization_002.ipynb)
- **Data filtering and feature selection:** [data_filtering.md](data_filtering.md)
- **Older notes and quick checks:** [normalization_notes.md](normalization_notes.md)