# Steam Recommendations: Branch Review & Todo Plan

## Branch review: `rreuther/preprocessing-001`

**What’s done on this branch:**

- **Repo setup**
  - `pyproject.toml`, `src/steam_review_ml` layout, data under `data/raw/`, configs in `configs/`.
  - README with goals (sentiment + helpfulness prediction, FastAPI frontend).

- **EDA**
  - Split EDA across notebooks in `notebooks/eda/`: targets (eda_001), quality (eda_002), text (eda_003), features (eda_004), numeric (eda_005), normalization explore/reference (eda_006), categorical counts (eda_007).
  - ETL/clean pipeline notebook in `notebooks/etl/eda_008_clean_pipeline.ipynb`.
  - Docs: `docs/eda_plan.md`, `docs/data_filtering.md`, `docs/normalization_notes.md`, `docs/large_data_cleaning_options.md`.

- **Preprocessing**
  - **Data pipeline:** `filter_reviews`, `select_features`, `feature_engineering` in `src/steam_review_ml/data/preprocess.py` (language filter, vote sentinel, empty/short reviews, negative playtime, column selection, derived features).
  - **Streaming:** `iter_clean_chunks` in `loaders.py` (chunked CSV → filter → dedupe by `review_id` → select_features → feature_engineering); `write_parquet_chunked` in `export.py`.
  - **Script:** `scripts/clean_reviews.py` reads JSON config, runs load + clean + export; logging + tqdm progress bar.
  - **Config:** `configs/clean_reviews.json` (input_path, output_path, chunksize, language).
  - **Tests:** `tests/test_preprocess.py` for filtering and feature selection.

---

## Generic todo plan (what’s next)

High-level phases; order is a suggestion. Details (e.g. exact metrics, model choices) live in code and other docs.

| Phase | Status | Notes |
|-------|--------|--------|
| **1. Repo setup** | Done | Structure, deps, README, data source. |
| **2. EDA** | Done | Targets, quality, text, features, numeric, normalization exploration. |
| **3. Preprocessing** | Done | Filter + feature selection + streaming export to Parquet; tests. |
| **4. Train/val/test split** | Todo | Split cleaned Parquet (e.g. by time or stratified by `app_id`/target); ensure dedupe is before split; persist split indices or paths. |
| **5. Normalization / feature prep** | Todo | Per `docs/normalization_notes.md`: cap/log for long-tail numerics, optional scaling; fit on train, save artifact; apply to val/test and to inference inputs later. |
| **6. Baselines** | Todo | Simple baselines for (1) `recommended` (e.g. majority class, logistic regression on metadata-only) and (2) helpfulness (e.g. predict 0, or simple regressor). Establish metrics and report. |
| **7. Sentiment model(s)** | Todo | Models that use text + metadata to predict `recommended`; e.g. sklearn pipeline (TF–IDF + numeric) or small neural net; compare to baseline. |
| **8. Helpfulness model(s)** | Todo | Models for `votes_helpful` or `is_helpful` (regression and/or classification); same feature set; compare to baseline. |
| **9. Evaluation & metrics** | Todo | Fix metrics (e.g. F1 for sentiment, RMSE or classification metrics for helpfulness); document train/val/test results and any stratification. |
| **10. API / frontend** | Todo | FastAPI backend that loads trained models and exposes endpoints; simple GUI: user enters review (and optional metadata), get predicted sentiment and helpfulness. |

---

## Quick reference

- **Run preprocessing:**  
  `python scripts/clean_reviews.py configs/clean_reviews.json`
- **Filtering/feature spec:**  
  `docs/data_filtering.md`
- **Normalization strategy:**  
  `docs/normalization_notes.md`
- **EDA order and notebooks:**  
  `docs/eda_plan.md`
