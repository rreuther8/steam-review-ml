# Steam Recommendations: Project todo plan

## Alignment (north star)

The **[archived recommender transition plan](archive/recommender_transition_plan.md)** records the original v1→v2 narrative; this file is the **living execution checklist**. The plan’s **north star** still applies: **v1** = **content-led retrieval** (embed **raw** review vs game profiles as **default** until structured beats raw on **val**; **structured** = `extract_preferences` + `build_embedding_input` as **ablation**), then **@K / proxy metrics**, then **v2 hybrid** reranking. **Preference extraction** stays in scope as product/experiment tooling, not the default embed until validated. **Review coaching** is **optional** and **separate** from preference extraction (see product vision). **ALS / collaborative filtering is deferred** until after that baseline exists.

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
  - Docs: [`archive/eda/eda_plan.md`](archive/eda/eda_plan.md), [`draft/etl/data_filtering.md`](draft/etl/data_filtering.md), [`archive/etl/normalization_notes.md`](archive/etl/normalization_notes.md), [`archive/etl/large_data_cleaning_options.md`](archive/etl/large_data_cleaning_options.md).

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

**Suggested execution order for the primary lane** matches the **[archived transition plan](archive/recommender_transition_plan.md)** narrative → game profiles + demo retrieval (**done**), **`recs_004`** proxy on val (**raw default** vs structured + baselines), iterate **preference extraction** until structured wins if desired, then API and v2.

### v1 recommender — living checklist (update as you go)

Use this as the single “where are we?” list; historical architecture narrative: **[`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md)**.

- [x] **`recs_001`** — train split, thumbs-up table → `artifacts/recs/embeddings/game_profile/default/game_profile_reviews.parquet`
- [x] **`recs_002`** — per-review embed, mean per `app_id`, L2 normalize → `artifacts/recs/embeddings/game_profile/default/game_profile_embeddings.npz` + index Parquet + `meta.json`
- [x] **`recs_003`** — load artifacts, TF Hub query embed, dot-product **top‑K** (demo / smoke test)
- [x] **`recs_004`** — same-user held-out likes proxy on **val**: baselines, raw/structured, train-pool multi, time-weighted train (`notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a.ipynb`; related `*_002` / archived B/C tasks). **Caveat:** eval subset = multi-game-like users; see **[`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md)** → *Selection bias: multi-review vs single-review users*.
- [x] **`extract_preferences` + `build_embedding_input`** — rules **v0** in `src/steam_review_ml/recommender/preferences.py` (not coaching); LLM upgrade optional
- [x] **Wire retrieval (product/API)** — `ContentRetriever` in `steam_review_ml.recommender` (`retrieve.py`); optional FastAPI in `steam_review_ml.api` (`create_app`); **default** embed = **raw** (`structured=` flag)
- [x] **Central offline eval + contract tables** — `scripts/recs_job_eval_offline.py` + `configs/recs_job_eval_offline.json`; paired `eval_retrieval_*` / `eval_ranking_*` under `artifacts/recs/offline_eval/runs/latest/`; slice/metric policy in [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md)
- [x] **Eval regression / baseline snapshot** — `tests/retrieval_eval_regression.py` (contract + optional JSON baseline); refresh with `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json --write-baseline` (see [`usage_pipeline.md`](usage_pipeline.md))
- [x] **Cached eval cohorts** — `scripts/recs_job_build_eval_examples.py` + `configs/recs_job_build_eval_examples.json`; eval job accepts `--examples-parquet` / config `examples_parquet` (see [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md))
- [x] **Minimal recommendations API (dev)** — FastAPI `/recommendations`, `/ui`, `/games`, `/health` in `steam_review_ml.api` (install `.[api]`)

### Prioritized next work (current intent)

Order for closing out **v1** before **v2** (IGDB hybrid) and wrap-up:

1. **[x] Two-tower retrieval** — `two_tower_v1` shipped as retrieve mechanism (`retrieval_decision_log` § 2026-05-30).
2. **[x] Ranking model (v1 scope)** — D1 `two_tower_v1_heuristic_logpop_blend` shipped; D2–D6 killed (`ranking_decision_log`, `recs_018`).
3. **[x] Wire rankers into eval jobs** — v2a `two_tower_v1_v2a_embed_query_logpop_blend` + D1 benchmark in `recs_job_eval_offline.json` / `recs_job_eval_ranking.json`. Remaining: `--write-baseline`, regression refresh.
4. **[x] Experiment registry** — YAML manifest + export script + `docs/experiment_registry.md`; v1 backfill, v2 placeholder rows. **Plan:** [`plans/experiment_registry_plan.md`](plans/experiment_registry_plan.md).
5. **[x] Wrap-up** — v1 closeout documented in [`recommender_v1_wrap_up.md`](recommender_v1_wrap_up.md); [`recommender_v2_plan.md`](recommender_v2_plan.md) drafted (IGDB metadata hybrid).
6. **Parked:** public API / deploy — **§ Later — public API / deploy** below.

### v2 recommender — Active checklist

Plan: [`recommender_v2_plan.md`](recommender_v2_plan.md). Decisions: [`plans/recommender_v2_questionnaire.md`](plans/recommender_v2_questionnaire.md) (frozen).

**Supporting (not in experiment registry):** IGDB join — `python scripts/recs_job_igdb_games.py configs/recs_job_igdb_games.json` → `artifacts/igdb/`; coverage EDA in [`notebooks/igdb/igdb_001_eda_join_coverage.ipynb`](../notebooks/igdb/igdb_001_eda_join_coverage.ipynb).

- [x] **IGDB EDA** — pipeline job run; enriched parquet + coverage EDA
- [x] **V2a-query (Jaccard)** — `recs_019`; pure retr+meta killed; logpop_blend candidate (`recs_021` head-to-head)
- [x] **V2a-embed (USE taxonomy)** — `recs_020`; **`two_tower_v1_v2a_embed_query_logpop_blend` shipped** (`ranking_decision_log` § 2026-06-22)
- [ ] **V2b** — USE summary sim (`query_review` ⟷ `igdb_summary`); val vs v2a / D1
- [ ] **V2a-history** — metadata vs train-likes union; val vs v2a / D1
- [ ] **V2c-query** — combined blend (only if singles show lift on train_tune)
- [ ] **V2d** — primary-genre-weighted metadata (optional refinement)
- [x] **Ship v2a** — wired into rank eval jobs + registry + ranking decision log

**v1 (content-led retrieval) — complete for chosen scope:** Shipped stack = `two_tower_v1` @100 → D1 @10. Optional backlog: test freeze, fixed-draft studies (*After v1* below).

### After v1 — evaluation and retrieval experiments (future backlog)

- **Evaluation stretch** — one-shot **test** holdout after method freeze (`RECS004_EVAL_SPLIT=test` in the `recs_004` family); fixed-draft studies. *(Scripted val eval already includes popularity decile tables, e.g. `eval_*_by_pop_decile.csv`—this line is about **extra** gates and narratives, not “add pop slicing from zero.”)*
- **Fixed-draft A/B matrix** — hold **the same user drafts** fixed and compare retrieval recipes so differences reflect **modeling choices**, not which examples landed in the bucket. The **main scientific point** is to learn whether you need something that **separately accounts for negative / complaint-side signal** (vs treating the review as one positive-direction embedding only). *Example* designs include a small factorial over **raw vs structured** query text and **single-vector vs dual-channel / penalty-style** scoring (as explored in `recs_003`); the exact cells are **not** locked in advance—pick whatever contrasts best isolate “negative-handling” for your stack. **Not** required to declare v1 retrieval “done.”
- **Stronger encoder for structured text** — e.g. rerun `recs_006` 4-way with a non-USE embedding model (same splits/seeds) to see if structured text closes the gap.
- **Negative / complaint-side policy** — caps and balancing if you lean hard into pos/neg query channels or separate pos/neg **item** vectors (see archived transition plan negative-handling notes).
- **v2 hybrid rerank** — active checklist in **§ v2 recommender — active checklist** above; plan: [`recommender_v2_plan.md`](recommender_v2_plan.md).

### Later — public API / deploy (parked)

Not needed for the **two-tower + ranking** modeling push; pick these up when the API leaves a trusted / local environment.

- [ ] **API auth** — identity or tokens for `/recommendations` (and related routes)
- [ ] **API rate limits / abuse controls** — basic throttling and protection against overload or scraping
- [ ] **Deploy hardening** — secrets, config, logging, health checks, and whatever your host needs for a non-dev URL

### Primary — recommendation

| Phase | Status | Notes |
|-------|--------|--------|
| **Game index + demo retrieval (`recs_001`–`003`)** | **Done** | Notebooks: `game_embeddings/recs_001_*`, `recs_002_*`; `query_embeddings/recs_003_*`. Next work is **product path**, not re-embedding unless you change caps/model. |
| **Preference extraction (structured path)** | **Rules v0 done** | Module in `src/`; **ablation** vs raw — beat **raw** on val proxy (and popularity) before promoting to default. |
| **Recommender v1 (product retrieval path)** | **Near complete** | **`ContentRetriever.top_k`** + **`steam_review_ml.api`** (UI + endpoints); **`recs_003`** / **`recs_004`** for exploration. Remaining v1 gap is **ops** if you need production hardening, not missing retrieval core. |
| **Recommender @K evaluation** | **Done (v1 scope)** | **Val path:** `recs_job_eval_offline.py`, `eval_retrieval_*` / `eval_ranking_*`, contract v2, decile tables, optional `retrieval_eval_regression` + `--write-baseline`. **Post–v1:** frozen **test**, fixed-draft matrix, extra experiments — see *After v1* above. |
| **API: recommendations** | **Near complete** | **MVP shipped.** **Post–v1 / ops:** auth, abuse limits, deploy hardening for untrusted traffic. |
| **Recommender v2 (hybrid rerank)** | **In progress** | Rank-only on frozen pools; IGDB EDA → V2a-query → V2b → … See [`recommender_v2_plan.md`](recommender_v2_plan.md). ALS deferred v2.1. |

### Supporting — data pipeline & tabular review models

| Phase | Status | Notes |
|-------|--------|--------|
| **1. Repo setup** | Done | Structure, deps, README, data source. |
| **2. EDA** | Done | Targets, quality, text, features, numeric, normalization exploration. |
| **3. Preprocessing** | Done | Filter + feature selection + streaming export to Parquet; tests. |
| **4. Train/val/test split** | Done* | `configs/split_reviews.json`; see `docs/usage_pipeline.md`. |
| **5. Normalization / feature prep** | Done* | `*_norm.parquet`, params under `artifacts/`; see `docs/usage_pipeline.md` / [`archive/etl/normalization_notes.md`](archive/etl/normalization_notes.md). |
| **6. Tabular baselines** | Done* | `notebooks/models/tabular/model_000_*` — dumb baselines for `recommended` and `_norm_votes_helpful`. |
| **7. Helpfulness regression (simple)** | Done* | `notebooks/models/tabular/model_001_*`: baselines + linear regression on **`votes_helpful`** / `_norm_votes_helpful`. No parallel **`is_helpful`** classifier as a primary target (derived from counts). |
| **8. Sentiment classification (simple)** | Done* | `notebooks/models/tabular/model_002_*`: baselines + logistic regression on `recommended` (numeric/normalized features). |
| **9. Richer tabular models** | Todo | e.g. **TF–IDF + numeric** pipeline or small neural net for `recommended`; iterate on **`votes_helpful`** if needed. Feeds coaching analysis and **v2** optional features. |

### Shared

| Phase | Status | Notes |
|-------|--------|--------|
| **Evaluation** | Partial | Tabular: [`archive/classification_metrics.md`](archive/classification_metrics.md) + `steam_review_ml.evaluation`. **Recommender v1:** val contract job + notebooks **done**; **post–v1** science (test freeze, fixed drafts) in *After v1* section. |
| **API / frontend (full product)** | Partial | **Draft → recs v1 path shipped** (FastAPI + UI). **Full product** (auth, coaching, tabular endpoints) remains **post–v1** or parallel tracks. |

\*Confirm in your checkout; paths and artifacts reflect a typical run.

---

## Focused backlog (from retired `docs/todo/` notes, 2026-05-11)

### Retrieval vs ranking eval

- **Invariants:** per example/method keep ordered `retrieved_app_ids_json` (+ scores) and `ranked_app_ids_json`; require `set(ranked) ⊆ set(retrieved)`; do not interpret ranking metrics without retrieval context; version `k_retrieval`, `k_final`, mask policy, and model snapshot in run metadata.
- **Shipped:** split `eval_retrieval_*` and `eval_ranking_*` tables from the central job; `recs_011` compares candidates against the same contract.
- **Future:** reranker that reorders inside retrieved candidates only; ceiling metrics (e.g. best possible NDCG given candidates).
- **Ops:** eval root is `artifacts/recs/offline_eval/runs/latest/`; migrate off legacy `artifacts/recs/retrieval/` or `artifacts/recs/eval` via `scripts/recs_migrate_artifacts_layout.py` or regenerate.
- **Leakage checklist:** train-only support/history; no validation positives in support pools; no post-query leakage; document embedding training window.

### Deferred `recs_XXX` integration walkthrough

Notebook-first onboarding for offline-eval **extras** stays **postponed**. MVP remains `scripts/recs_job_eval_offline.py`, optional `--examples-parquet` / `examples_parquet`, and `recs_011` for deltas. If revived, keep a **`recs_XXX`** notebook as a thin index—do not duplicate `usage_pipeline.md`.

### Code simplification targets (`src/steam_review_ml`)

- **High impact:** split `run_retrieval_eval` orchestration in `evaluation/retrieval_offline_eval.py` (metrics vs tables vs metadata); centralize `n_eval_targets → slice_name`; shared `_apply_query_mask` for method scorers.
- **Medium:** loader split-mode dispatch in `data/loaders.py`; reduce row-by-row `.loc` where vectorization is safe.
- **Lower:** more declarative preprocess steps in `data/preprocess.py`; typed normalization rule specs in `transforms/normalization.py`.

---

## Quick reference

- **Run preprocessing:**  
  `python scripts/clean_reviews.py configs/clean_reviews.json`
- **Filtering/feature spec:**  
  [`draft/etl/data_filtering.md`](draft/etl/data_filtering.md)
- **Normalization strategy:**  
  [`archive/etl/normalization_notes.md`](archive/etl/normalization_notes.md)
- **EDA order and notebooks:**  
  [`archive/eda/eda_plan.md`](archive/eda/eda_plan.md)
- **Recommender path (v1 → v2, archived narrative):**  
  [`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md)
- **v1 checklist (check off as you finish steps):**  
  this file → **§ v1 recommender — living checklist**
- **Product vision (core recs + optional coaching):**  
  [`product_vision_recommender_and_review_coaching.md`](product_vision_recommender_and_review_coaching.md)
- **Eval contract + notebook map:**  
  [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md)
