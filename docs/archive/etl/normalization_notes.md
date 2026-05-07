# Normalization

## Status and validation

- **Transforms are fixed** for the listed numeric columns (cap at train 99th percentile + `log1p`, or `log1p` only). See the table below.
- **Implementation:** [`src/steam_review_ml/transforms/normalization.py`](../../src/steam_review_ml/transforms/normalization.py) — `fit_normalization`, `add_normalized_columns`, `make_norm_col_name`, `inverse_norm_votes_helpful`.
- **Batch step:** [`scripts/normalize_split_parquets.py`](../../scripts/normalize_split_parquets.py) with [`configs/normalize_splits.json`](../../configs/normalize_splits.json). Fits on **train** only; writes **`data/processed/..._norm.parquet`** plus **`artifacts/normalization_params.json`**. Interim **raw-scale** splits live under `data/interim/` (see [`configs/split_reviews.json`](../../configs/split_reviews.json)).
- **Tests:** [`tests/test_normalization.py`](../../tests/test_normalization.py).
- **Notebook checks (artifacts):** [`notebooks/etl/eda_010_normalization_validation.ipynb`](../../notebooks/etl/eda_010_normalization_validation.ipynb) — row counts, schema, recomputation vs JSON, optional inverse spot check for `votes_helpful`.
- **Runbook:** [docs/usage_pipeline.md](../usage_pipeline.md).

## Optional EDA checks (signal, not plumbing)

Use these when you **change the transform table**, add features, or want reassurance before modeling. They assume you are working on a dataframe that already includes **`_norm_*`** columns (e.g. loaded from `*_norm.parquet` or built in a notebook).

**Runnable on-repo:** [Section 6 of `eda_010_normalization_validation.ipynb`](../../notebooks/etl/eda_010_normalization_validation.ipynb) runs a **subsampled** version of these checks against processed train (`*_norm.parquet`): raw vs `_norm_*` skew/histograms, separation by `recommended` / `is_helpful`, `_norm_*` correlation heatmap, coarse tail (z) stats, and scatters vs `_norm_votes_helpful`.

- **Feature vs target (from eda_004)**  
  Re-run numeric feature vs `recommended` and vs helpfulness views (e.g. boxplots/violins) on **`_norm_*`**. Confirm separation is preserved or improved; if it worsens for a column, revisit that transform.

- **Correlation (from eda_005)**  
  Correlation heatmap on **`_norm_*`** numerics to spot redundancy and multicollinearity before modeling.

- **Outliers**  
  Re-run your usual outlier logic on **`_norm_*`** to confirm capping/logging behaved as expected and no new pathologies appear.

- **`review_word_count` / other derived counts**  
  If included in the model, include them in the same distribution and correlation checks. (Current pipeline table includes `review_word_count` only; extend checks if you add more columns.)

- **Regression target**  
  For `votes_helpful` regression, the table uses cap + `log1p` on train; use **`_norm_votes_helpful`** in the modeling frame and **`inverse_norm_votes_helpful`** when you need raw-scale metrics or baselines.

## Where this lives

- **Spec:** [data_filtering.md](data_filtering.md) Section 4 (Normalization and feature transforms) — keep aligned with the table below.
- **Exploration / TF–sklearn refs:** [notebooks/eda/eda_006_normalization_002.ipynb](../../notebooks/eda/eda_006_normalization_002.ipynb), [eda_006_normalization.ipynb](../../notebooks/eda/eda_006_normalization.ipynb).

---

## Chosen transforms (per column)

Decisions from EDA (see [notebooks/eda/eda_006_normalization_002.ipynb](../../notebooks/eda/eda_006_normalization_002.ipynb)):

| Column | Transform |
|--------|-----------|
| `votes_helpful` | Cap at 99th percentile, then log1p |
| `author.playtime_last_two_weeks` | Cap at 99th percentile, then log1p |
| `votes_funny` | log1p only |
| `author.playtime_at_review` | log1p only |
| `author.num_games_owned` | log1p only |
| `author.num_reviews` | log1p only |
| `review_word_count` | log1p only |

- **Cap-then-log:** fit the cap on **training** data only; apply `min(x, cap)` then `log1p`.
- **Log only:** `log1p(x)` with no capping.

The reference notebook also shows equivalent manual, scikit-learn, and TensorFlow patterns. The **repo default** for saved Parquets is the Python helpers above.

---

## Pipeline: where normalization runs

Normalization is done **once** as a data-preparation step, not per model.

1. **Fit** on the training split only (full train column slice used by `fit_normalization`).
2. **Save** fitted parameters to **`normalization_params.json`**
3. **Transform** train, validation, and test with the same frozen params.
4. **Save** modeling Parquets under **`data/processed/`** (`*_norm.parquet`). Downstream notebooks load these; raw source columns remain alongside `_norm_*`.

Effects:

- All models can share the same pre-normalized tables.
- One place to re-run if you change percentiles or columns: re-fit, re-transform, version or overwrite outputs.

**CLI:** From repo root (with `pip install -e .` or `PYTHONPATH=src`):

`PYTHONPATH=src python scripts/normalize_split_parquets.py configs/normalize_splits.json`

[`configs/normalize_splits.json`](../../configs/normalize_splits.json) reads **interim** split paths (same outputs as [`configs/split_reviews.json`](../../configs/split_reviews.json)) and writes processed `*_norm.parquet` plus [`artifacts/normalization_params.json`](../../artifacts/normalization_params.json). Optional JSON key `normalization_rules` overrides the default rule dict from code. Normalized column names: **`_norm_<source>`** with `.` → `__` in `<source>`.

---

## Inference (e.g. in the app)

New inputs are raw; normalize them with the **saved** params (or the same code path) before calling the model.

- Load `normalization_params.json` (or equivalent) once.
- For each row: build `_norm_*` from raw fields, then score.

Artifacts to keep:

- **Pre-normalized datasets** for training and evaluation.
- **Saved params** (and code using `add_normalized_columns`) for inference on new raw rows.

---

## References

- **Transforms reference notebook:** [notebooks/eda/eda_006_normalization_002.ipynb](../../notebooks/eda/eda_006_normalization_002.ipynb)
- **Data filtering / feature selection:** [data_filtering.md](data_filtering.md)
- **Pipeline runbook:** [docs/usage_pipeline.md](../usage_pipeline.md)
- **Artifact validation notebook:** [notebooks/etl/eda_010_normalization_validation.ipynb](../../notebooks/etl/eda_010_normalization_validation.ipynb)
