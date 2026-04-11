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
  - `model_000_dumb_002.ipynb` — dumb baselines for **`recommended`** and **`_norm_votes_helpful`**.  
  - `model_001_linreg__votes_helpful.ipynb` — baselines + **linear regression** on normalized helpful votes; metrics e.g. `artifacts/metrics/votes_helpful_metrics.csv` when run.  
  - `model_002_logreg__recommended.ipynb` — baselines + **logistic regression** on `recommended` (full numeric feature set + 3-feature variant).  

- **Recommender v1 (primary lane) — content index + demo retrieval** — notebooks under `notebooks/models/game_embeddings/` and `notebooks/models/query_embeddings/`: **`recs_001_game_profiles.ipynb`** → **`game_profile_reviews.parquet`**; **`recs_002_embed_game_profiles.ipynb`** → dense **per-game** vectors (TF Hub USE, mean pool, L2 norm); **`recs_003_query_retrieve.ipynb`** → hand-written query → embed → **top‑K** vs `X`. Artifacts: **`artifacts/recs/`**.

---

## Roadmap (by lane)

**Suggested execution order for the primary lane** matches **`docs/recommender_transition_plan.md`** → game profiles + demo retrieval (**done**), **`recs_004`** proxy on val (**raw default** vs structured + baselines), iterate **preference extraction** until structured wins if desired, then API and v2.

### v1 recommender — living checklist (update as you go)

Use this as the single “where are we?” list; **`docs/recommender_transition_plan.md`** stays the architecture narrative.

- [x] **`recs_001`** — train split, thumbs-up table → `artifacts/recs/game_profile_reviews.parquet`
- [x] **`recs_002`** — per-review embed, mean per `app_id`, L2 normalize → `game_profile_embeddings.npz` + index Parquet + `meta.json`
- [x] **`recs_003`** — load artifacts, TF Hub query embed, dot-product **top‑K** (demo / smoke test)
- [ ] **`recs_004`** — same-user held-out likes proxy on **val**: **raw vs structured** vs **random** + **popularity** baselines; optional **multi-review** mean/concat (`notebooks/models/query_embeddings/recs_004_eval_same_user_proxy.ipynb`)
- [ ] **`extract_preferences` + `build_embedding_input`** — core v1 query text (not coaching); LLM or rules v0 OK
- [ ] **Wire retrieval** — same `top_k` path; **default** embed = **raw** draft until structured wins on val; **structured** string as ablation / optional flag
- [ ] **@K evaluation** — Precision@K / Recall@K / MAP@K / NDCG@K vs baselines (e.g. popularity); structured vs raw on fixed drafts; **user held-out likes** proxy (other `recommended` games by the same user — see **`docs/recommender_transition_plan.md`** → *Offline proxy task*)
- [ ] **A/B matrix eval** — compare 4 variants on a fixed set: raw+positive-only, structured+positive-only, raw+dual-index, structured+dual-index
- [ ] **Negative profile sampling/balance policy** — for any dual-index run, cap/symmetrize pos/neg per-game review counts; guard against noisy/event-driven negatives
- [ ] **API** — minimal endpoint: draft or prefs → recommendations (after eval loop is acceptable)
- [ ] **v2 hybrid** — defer until baseline above holds; see transition plan

### Primary — recommendation

| Phase | Status | Notes |
|-------|--------|--------|
| **Game index + demo retrieval (`recs_001`–`003`)** | **Done** | Notebooks: `game_embeddings/recs_001_*`, `recs_002_*`; `query_embeddings/recs_003_*`. Next work is **product path**, not re-embedding unless you change caps/model. |
| **Preference extraction (structured path)** | Todo | `extract_preferences` + `build_embedding_input` → embedding **ablation**; beat **raw** on val proxy before making default. |
| **Recommender v1 (product retrieval path)** | Todo | Reuse **`recs_003`** mechanics; **default** query = **raw** embed; optional structured + refactor into `src/` for API/tests. |
| **Recommender @K evaluation** | Todo | Precision@K, Recall@K, MAP@K, NDCG@K vs. simple baselines (e.g. popularity); coverage/diversity as needed. Include **structured vs raw** and, where data allows, the **held-out same-user likes** proxy in the transition plan. |
| **API: recommendations** | Todo | Expose v1 retrieval (then extend for v2). Can ship after a minimal v1 + eval loop exists. |
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
| **Evaluation** | Partial | Tabular: `docs/classification_metrics.md` + `steam_review_ml.evaluation` in notebooks. **Recommender @K** still Todo until v1 exists. |
| **API / frontend (full product)** | Todo | FastAPI: draft → **preference extraction + recommendations** (core); optional tabular endpoints; **optional** coaching per product vision (separate from extraction). |

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
