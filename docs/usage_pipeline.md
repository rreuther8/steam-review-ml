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

**TensorFlow + TensorFlow Hub** (for `recs_002`–`recs_004`, `ContentRetriever`, and the optional API): use **conda-forge** for the TF stack, then install this repo with pip (no `tensorflow` from PyPI in the same env).

**NVIDIA Blackwell (e.g. RTX 5070, sm_120):** PyPI wheels are built with a fixed set of SMs (often up through **sm_90**). They may **not register your GPU** (empty `list_physical_devices("GPU")`) or hit library/ cuDNN errors. Conda-forge’s **GPU** builds are labeled with **`cuda128`** in the package build string (CUDA 12.8) and include **sm_120** in `get_build_info()` when you need native Blackwell kernels.

```bash
conda activate <your-env>
# Remove any pip TensorFlow metapackage so conda owns the namespace:
pip uninstall -y tensorflow tensorflow-intel 2>/dev/null || true

# Prefer a CUDA 12.8 GPU build (build string contains cuda128, not cpu_):
conda install -c conda-forge "tensorflow==2.19.1=*cuda128*" tensorflow-hub

pip install -e .
pip install -e '.[api]'   # optional: FastAPI server
```

**Sanity checks**

- `conda list tensorflow` — **Build** should look like `cuda128py311h…`, not `pypi` or `cpu_py…`.
- After `import tensorflow as tf`, `tf.sysconfig.get_build_info()["cuda_compute_capabilities"]` should list **`sm_120`** / **`compute_120`** (not only through `compute_90`).
- Jupyter: kernel must be **this** conda env (not another env that still has pip TF 2.2x).

If the solver still picks a **cpu** build, tighten constraints or create a fresh env with `conda-forge` as the only channel for TF-related packages.

Pip **cannot** install conda-forge CUDA TensorFlow, so there is no `[recs]` extra that replaces this. For **pip-only** environments (e.g. a minimal container), use `pip install -e '.[recs-pip]'` instead — do **not** mix that with conda-managed TensorFlow.

After processed train Parquet exists, build **game profiles** (train split, positive reviews only):

- Notebook: `notebooks/models/game_embeddings/recs_001_game_profiles.ipynb`
- Output: `artifacts/recs/game_profile_reviews.parquet` — one row per thumbs-up review (capped per game); input for **per-review embed + mean** in `recs_002`.

**Dense game vectors** (TensorFlow + TensorFlow Hub; see TF install notes above and `recs_002` notebook):

- Notebook: `notebooks/models/game_embeddings/recs_002_embed_game_profiles.ipynb`
- Outputs: `artifacts/recs/game_profile_embeddings.npz`, `game_profile_embedding_index.parquet`, `game_profile_embedding_meta.json`

**Query + top‑K (smoke test / demo)** — same TF Hub model as `recs_002` (URL read from `game_profile_embedding_meta.json`):

- Notebook: `notebooks/models/query_embeddings/recs_003_query_retrieve.ipynb`

**Offline eval (same-user held-out likes proxy)** — default **val** queries (`*_val_norm.parquet`); **raw / structured** vs **random** and **train popularity**; train-pool multi + time windows; **MAP@K** / **NDCG@K**. For a **one-shot test holdout** after freezing the method: `RECS004_EVAL_SPLIT=test` (requires `*_test_norm.parquet`).

- Notebook: `notebooks/models/query_embeddings/recs_004_eval_same_user_proxy.ipynb`

**4-way raw/structured comparison + regression baseline (recs_006):**

- Build structured index artifact first: `notebooks/models/game_embeddings/recs_005_structured_game_embeddings.ipynb`
- Run comparison/eval: `notebooks/models/query_embeddings/recs_006_eval_queries.ipynb`
- Save/compare `raw_raw` regression guard:

```bash
python scripts/check_recs_006_regression.py
```

Decision/log artifacts:
- `docs/retrieval_decision_log.md`
- `artifacts/recs/active_retrieval_config.json`
- `artifacts/recs/eval_review_style_4way_proxy_baseline_raw_raw.json`

**Programmatic retrieval (v1 wire)** — `steam_review_ml.recommender.ContentRetriever` loads `artifacts/recs/` and exposes `top_k(...)` (raw or structured). Optional HTTP: TF + Hub as above, then `pip install -e '.[api]'`, then  
`uvicorn steam_review_ml.api:create_app --factory --host 127.0.0.1 --port 8000` (or `steam_review_ml.api.app:create_app`).

Endpoints: **`GET /ui`** — browser UI (game typeahead + review draft → recommendations); **`GET /games`** (`q` = optional substring on `app_name`, `limit`) for a typeahead picker; **`GET /recommendations`** with **`exclude_app_id`** set to the selected game so it never appears in results.

See `docs/recommender_transition_plan.md` for the full v1 path.

---

## Order Summary

1. `python scripts/clean_reviews.py configs/clean_reviews.json`
2. `python scripts/split_reviews.py configs/split_reviews.json`
3. `python scripts/normalize_split_parquets.py configs/normalize_splits.json`
4. (Optional) run tabular modeling notebooks
5. (Recommender v1) run `notebooks/models/game_embeddings/recs_001_game_profiles.ipynb`
6. (Optional) install TF + Hub (conda-forge or `.[recs-pip]`) and run `notebooks/models/game_embeddings/recs_002_embed_game_profiles.ipynb`
7. (Optional) run `notebooks/models/query_embeddings/recs_003_query_retrieve.ipynb` after `recs_002` artifacts exist
8. (Optional) run `notebooks/models/query_embeddings/recs_004_eval_same_user_proxy.ipynb` for proxy metrics (default **val**; `RECS004_EVAL_SPLIT=test` for final holdout)
9. (Optional) serve recommendations: `uvicorn steam_review_ml.api:create_app --factory` (requires TF + Hub + `.[api]`; pip-only stack: `.[api,recs-pip]`; repo root on `PYTHONPATH` or editable install)



## Full run

```
python scripts/clean_reviews.py configs/clean_reviews.json && \
python scripts/split_reviews.py configs/split_reviews.json && \
python scripts/normalize_split_parquets.py configs/normalize_splits.json
```