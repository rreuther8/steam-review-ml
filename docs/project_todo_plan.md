# Steam Recommendations: Project todo plan

## Alignment with the recommender transition plan

The **[recommender transition plan](recommender_transition_plan.md)** sets the **north star**: **v1** = **content-led retrieval** (embed **raw** review vs game profiles as **default** until structured beats raw on **val**; **structured** = `extract_preferences` + `build_embedding_input` as **ablation**), then **@K / proxy metrics**, then **v2 hybrid** reranking. **Preference extraction** stays in scope as product/experiment tooling, not the default embed until validated. **Review coaching** is **optional** and **separate** from preference extraction (see product vision). **ALS / collaborative filtering is deferred** until after that baseline exists.

This todo list follows that priority:

| Lane | Role |
|------|------|
| **Primary** | Game profiles + similarity retrieval (**raw embed default**; structured when it wins on val) → proxy / @K evaluation → API → v2 hybrid. **Preference extraction** supports structured path and UX. |
| **Supporting** | Data pipeline, **tabular** `recommended` / `votes_helpful` models (baselines + simple learners). These support **analysis** and **optional v2 features**; they are **not** a replacement for extraction + retrieval. |

If time is tight, **do not** let tabular modeling block **game profiles + similarity retrieval + proxy eval**; structured extraction can iterate in parallel.

---

## Repository foundation (on `main`)

**What’s already merged** — the bullets below describe the core **repo setup, EDA, and preprocessing** stack. **Train/val/test splits**, **normalization**, and **model notebooks** landed in later merges; see the **roadmap tables** below and `docs/usage_pipeline.md`. This doc is **not** tied to one old feature branch—use your current `main` (or branch) as ground truth.

- **Repo setup**
  - `pyproject.toml`, `src/steam_review_ml` layout, data under `data/raw/`, configs in `configs/`.
  - README with goals (sentiment + helpful-vote modeling, FastAPI frontend).

- **EDA**
  - Split EDA across notebooks in `notebooks/eda/`: targets (eda_001), quality (eda_002), text (eda_003), features (eda_004), numeric (eda_005), normalization explore/reference (eda_006), categorical counts (eda_007).
  - ETL/clean pipeline notebook in `notebooks/etl/eda_008_clean_pipeline.ipynb`.
  - Docs: `docs/eda_plan.md`, `docs/data_filtering.md`, `docs/normalization_notes.md`, `docs/large_data_cleaning_options.md`.

- **Preprocessing**
  - **Data pipeline:** `filter_reviews`, `select_features`, `feature_engineering` in `src/steam_review_ml/data/preprocess.py` (language filter, vote sentinel, empty/short reviews, negative playtime, column selection, derived features).
  - **Streaming:** `iter_clean_chunks` in `loaders.py` (chunked CSV → filter → dedupe by `review_id` → `select_features` only); text counts run in `scripts/split_reviews.py` after splitting. `write_parquet_chunked` in `export.py`.
  - **Script:** `scripts/clean_reviews.py` reads JSON config, runs load + clean + export; logging + tqdm progress bar.
  - **Config:** `configs/clean_reviews.json` (input_path, output_path, chunksize, language).
  - **Tests:** `tests/test_preprocess.py` for filtering and feature selection.

- **Tabular baselines & simple models (supporting lane)** — under `notebooks/models/tabular/`:  
  - `model_000_baseline_dumb.ipynb` — dumb baselines for **`recommended`** and **`_norm_votes_helpful`**.  
  - `model_001_regression_votes_helpful.ipynb` — baselines + **linear regression** on normalized helpful votes; metrics e.g. `artifacts/metrics/votes_helpful_metrics.csv` when run.  
  - `model_002_classification_recommended.ipynb` — baselines + **logistic regression** on `recommended` (full numeric feature set + 3-feature variant).  

- **Recommender v1 (primary lane) — content index + demo retrieval** — notebooks under `notebooks/models/game_embeddings/` and `notebooks/models/query_embeddings/`: **`recs_001_game_profile_reviews.ipynb`** → **`artifacts/recs/embeddings/game_profile/default/game_profile_reviews.parquet`**; **`recs_002_game_embeddings_raw.ipynb`** → dense **per-game** vectors (TF Hub USE, mean pool, L2 norm) under `artifacts/recs/embeddings/game_profile/default/`; **`recs_003_query_retrieve_smoke.ipynb`** → hand-written query → embed → **top‑K** vs `X`.

---

## Roadmap (by lane)

**Suggested execution order for the primary lane** matches **`docs/recommender_transition_plan.md`** → game profiles + demo retrieval (**done**), **`recs_004`** proxy on val (**raw default** vs structured + baselines), iterate **preference extraction** until structured wins if desired, then API and v2.

### v1 recommender — living checklist (update as you go)

Use this as the single “where are we?” list; **`docs/recommender_transition_plan.md`** stays the architecture narrative.

- [x] **`recs_001`** — train split, thumbs-up table → `artifacts/recs/embeddings/game_profile/default/game_profile_reviews.parquet`
- [x] **`recs_002`** — per-review embed, mean per `app_id`, L2 normalize → `artifacts/recs/embeddings/game_profile/default/game_profile_embeddings.npz` + index Parquet + `meta.json`
- [x] **`recs_003`** — load artifacts, TF Hub query embed, dot-product **top‑K** (demo / smoke test)
- [x] **`recs_004`** — same-user held-out likes proxy on **val**: **§3 ablation** — baselines, raw/structured, train-pool multi, time-weighted train (`notebooks/models/query_embeddings/recs_004_eval_proxy_same_user.ipynb`). **Caveat:** eval subset = multi-game-like users; see **`recommender_transition_plan.md`** → *Selection bias: multi-review vs single-review users*.
- [x] **`extract_preferences` + `build_embedding_input`** — rules **v0** in `src/steam_review_ml/recommender/preferences.py` (not coaching); LLM upgrade optional
- [x] **Wire retrieval (product/API)** — `ContentRetriever` in `steam_review_ml.recommender` (`retrieve.py`); optional FastAPI in `steam_review_ml.api` (`create_app`); **default** embed = **raw** (`structured=` flag)
- [ ] **@K evaluation (extended)** — `recs_004` adds MAP@K / NDCG@K; still open: fixed-draft matrix, popularity-aware slices; **test** via `RECS004_EVAL_SPLIT=test` when frozen
- [ ] **A/B matrix eval** — compare 4 variants on a fixed set: raw+positive-only, structured+positive-only, raw+dual-index, structured+dual-index
- [ ] **Optional exploration: stronger embedding model for structured text** — rerun `recs_006` 4-way matrix with an LLM embedding model (same splits/seeds) to test whether structured query/index text gains appear when moving beyond USE.
- [ ] **Negative profile sampling/balance policy** — for any dual-index run, cap/symmetrize pos/neg per-game review counts; guard against noisy/event-driven negatives
- [ ] **API** — minimal endpoint: draft or prefs → recommendations (after eval loop is acceptable)
- [ ] **v2 hybrid** — defer until baseline above holds; see transition plan

### Primary — recommendation

| Phase | Status | Notes |
|-------|--------|--------|
| **Game index + demo retrieval (`recs_001`–`003`)** | **Done** | Notebooks: `game_embeddings/recs_001_*`, `recs_002_*`; `query_embeddings/recs_003_*`. Next work is **product path**, not re-embedding unless you change caps/model. |
| **Preference extraction (structured path)** | **Rules v0 done** | Module in `src/`; **ablation** vs raw — beat **raw** on val proxy (and popularity) before promoting to default. |
| **Recommender v1 (product retrieval path)** | **Partial** | **`ContentRetriever.top_k`** + optional **`steam_review_ml.api`**; notebooks **`recs_003`** / **`recs_004`**; extend product integration as needed. |
| **Recommender @K evaluation** | **Partial** | **`recs_004`**: Hit/Recall/MRR/MAP/NDCG; `RECS004_EVAL_SPLIT=test` for holdout; fixed drafts / slices still open. |
| **API: recommendations** | **Partial** | FastAPI **`/recommendations`** behind `.[api]` extra; harden deploy + auth as needed. |
| **Recommender v2 (hybrid rerank)** | Todo | Same candidates as v1; blend similarity + priors/metadata + **optional** tabular scores (`p(recommended)`, expected helpful votes). ALS only when justified. |

### Supporting — data pipeline & tabular review models

| Phase | Status | Notes |
|-------|--------|--------|
| **1. Repo setup** | Done | Structure, deps, README, data source. |
| **2. EDA** | Done | Targets, quality, text, features, numeric, normalization exploration. |
| **3. Preprocessing** | Done | Filter + feature selection + streaming export to Parquet; tests. |
| **4. Train/val/test split** | Done* | `configs/split_reviews.json`; see `docs/usage_pipeline.md`. |
| **5. Normalization / feature prep** | Done* | `*_norm.parquet`, params under `artifacts/`; see `docs/usage_pipeline.md` / `docs/etl/normalization_notes.md`. |
| **6. Tabular baselines** | Done* | `notebooks/models/tabular/model_000_*` — dumb baselines for `recommended` and `_norm_votes_helpful`. |
| **7. Helpfulness regression (simple)** | Done* | `notebooks/models/tabular/model_001_*`: baselines + linear regression on **`votes_helpful`** / `_norm_votes_helpful`. No parallel **`is_helpful`** classifier as a primary target (derived from counts). |
| **8. Sentiment classification (simple)** | Done* | `notebooks/models/tabular/model_002_*`: baselines + logistic regression on `recommended` (numeric/normalized features). |
| **9. Richer tabular models** | Todo | e.g. **TF–IDF + numeric** pipeline or small neural net for `recommended`; iterate on **`votes_helpful`** if needed. Feeds coaching analysis and **v2** optional features. |

### Shared

| Phase | Status | Notes |
|-------|--------|--------|
| **Evaluation** | Partial | Tabular: `docs/classification_metrics.md` + `steam_review_ml.evaluation`. **Recommender:** `recs_004` proxy metrics on val; extended @K / test holdout still open. |
| **API / frontend (full product)** | Todo | FastAPI: draft → **raw (default) or structured** embed → recommendations; optional tabular endpoints; **optional** coaching (separate). |

\*Confirm in your checkout; paths and artifacts reflect a typical run.

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
- **Recommender path (v1 → v2):**  
  `docs/recommender_transition_plan.md`
- **v1 checklist (check off as you finish steps):**  
  this file → **§ v1 recommender — living checklist**
- **Product vision (core recs + optional coaching):**  
  `docs/product_vision_recommender_and_review_coaching.md`
