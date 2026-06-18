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

Uses `configs/split_reviews.json`. The split seed **`random_state`** is not in JSON; it comes from **`steam_review_ml.constants.PROJECT_RANDOM_SEED`** (currently `2026`) unless overridden via **`STEAM_REVIEWS_RANDOM_STATE`**.

Current config defaults to support-aware temporal splitting:

- **`split_mode = support_aware_user_temporal`**
- users with 1 interaction are assigned to train/val/test by global `val_size`/`test_size` ratios
- users with 2 interactions get 1 train row and 1 eval row (val/test chosen per-user)
- users with 3+ interactions keep at least 1 train row and assign most-recent eval rows to exactly one eval split (val or test), with at least 2 eval rows per user

Set `split_mode = random_stratified` to use legacy hash-stratified random split for all rows.

```bash
python scripts/split_reviews.py configs/split_reviews.json
```

Current config targets:

- `data/interim/steam_reviews_cleaned_english_train.parquet`
- `data/interim/steam_reviews_cleaned_english_val.parquet`
- `data/interim/steam_reviews_cleaned_english_test.parquet`

After each row is assigned to train/val/test, the split step runs **`feature_engineering`** (`review_word_count`, `review_length_chars`) and then **`review_age_seconds`**: seconds from `timestamp_created` to the **maximum `timestamp_created` in the training split only** (two-pass stream for the reference; then a second pass writes outputs).  
With temporal/hybrid split policies, newer val/test rows can clip to `review_age_seconds = 0`; script logs include clipped percentages for visibility.

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

- `notebooks/models/tabular/model_000_baseline_dumb.ipynb`
- `notebooks/models/tabular/model_001_regression_votes_helpful.ipynb`
- `notebooks/models/tabular/model_002_classification_recommended.ipynb`

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

- Pipeline job: `python scripts/recs_job_game_profiles.py configs/recs_job_game_profiles.json`
- Output (see `configs/recs_job_game_profiles.json`; canonical layout): `artifacts/recs/embeddings/game_profile/default/game_profile_reviews.parquet` — one row per thumbs-up review (capped per game); input for **per-review embed + mean** in `recs_002`.
- QA notebook (optional): `notebooks/models/game_embeddings/recs_001_game_profile_reviews.ipynb` — if it still points at `artifacts/recs/` root, set `ARTIFACT_DIR` to `artifacts/recs/embeddings/game_profile/default` or match the config path.

**Dense game vectors** (TensorFlow + TensorFlow Hub):

- Pipeline job: `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings.json`
- Structured variant job: `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings_structured.json`
- Default index outputs: `artifacts/recs/embeddings/game_profile/default/game_profile_embeddings.npz`, `game_profile_embedding_index.parquet`, `game_profile_embedding_meta.json`
- Structured eval outputs: `artifacts/recs/embeddings/game_profile/structured_eval/game_profile_embeddings_structured_eval.npz`, `game_profile_embedding_index_structured_eval.parquet`, `game_profile_embedding_meta_structured_eval.json`
- `ContentRetriever` loads from that default directory or legacy flat `artifacts/recs/` if those files exist there.
- QA notebooks (optional): `notebooks/models/game_embeddings/recs_002_game_embeddings_raw.ipynb`, `notebooks/models/game_embeddings/recs_005_game_embeddings_structured.ipynb`

### Recommender Jobs (separate)

Run these jobs independently so profile rebuilds and embedding rebuilds can be scheduled/retried separately:

1. `python scripts/recs_job_game_profiles.py configs/recs_job_game_profiles.json`
2. `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings.json`
3. (Optional structured index) `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings_structured.json`
4. (v2 metadata) `python scripts/recs_job_igdb_games.py configs/recs_job_igdb_games.json` — requires `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` in repo-root `.env` (gitignored) when `skip_fetch` is false. Config **must** set `output_dir`; raw parquet defaults to `IGDB_GAMES_PARQUET` in [`constants.py`](../src/steam_review_ml/igdb/constants.py). Optional `embed_text_fields` (e.g. `["summary"]`) writes `IGDB_GAMES_FEATURES_PARQUET` with original text columns plus `{field}__use` L2-normalized USE vectors (same TF Hub model as v1 game embeddings unless `tfhub_url` is set). Set `skip_fetch: true` to re-embed from existing raw parquet without calling IGDB. Also writes `igdb_join_report.json`, `meta.json` (fetch), and `igdb_features_meta.json` (embed). EDA: join coverage [`igdb_001`](../notebooks/igdb/igdb_001_eda_join_coverage.ipynb), per-field profiles [`igdb_002`](../notebooks/igdb/igdb_002_eda_game_review.ipynb).

**Query + top‑K (smoke test / demo)** — same TF Hub model as `recs_002` (URL read from `game_profile_embedding_meta.json`):

- Notebook: `notebooks/models/query_embeddings/recs_003_query_retrieve_smoke.ipynb`

**Offline eval (same-user held-out likes proxy)** — default **val** queries (`*_val_norm.parquet`); **raw / structured** vs **random** and **train popularity**; train-pool multi + time windows; **MAP@K** / **NDCG@K**. For a **one-shot test holdout** after freezing the method: `RECS004_EVAL_SPLIT=test` (requires `*_test_norm.parquet`).

- Notebook: `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user.ipynb`

**Full offline eval job (`recs_job_eval_offline`)** — re-scores methods end-to-end; writes paired retrieval- and ranking-contract tables plus frozen pools jsonl. Default methods include `raw`, `popularity_train`, `multi_mean_train`, **`fusion_c_raw_plus_behavior`**, `two_tower_v1`, and shipped D1. See [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md) § Offline eval jobs for contrast with rank-only `recs_job_eval_ranking`.

- **Slices, metric priorities, K semantics:** [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md) (eval contract v2 + notebook map + cached-examples runbook).
- Job: `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json`
- Progress: set `verbose: true|false` in config (default `true`) for tqdm/print status
- **Cutoffs:** `k_retrieval` (default: omit → same as `k_final`) caps the **retrieved candidate list** for the retrieval contract; `k_final` is the **ranking** / top‑shown list (`eval_ranking_*` Hit/NDCG/etc. use `k_final`; `eval_retrieval_*` Hit/Precision/Recall use `k_retrieval`). Changing these changes numbers — re-run **`--write-baseline`** when you intentionally move the regression contract.
- Outputs (default): `artifacts/recs/offline_eval/runs/latest/`
  - Set `archive_run: true` to also snapshot each run into:
    `artifacts/recs/offline_eval/runs/<timestamp>__<run_tag>/`
  - **Retrieval summaries:** `eval_retrieval_overall.csv`, `eval_retrieval_by_slice.csv`, `eval_retrieval_by_support_bucket.csv`, `eval_retrieval_by_pop_decile.csv`, `eval_retrieval_pop_delta_vs_popularity.csv`
  - **Ranking summaries:** `eval_ranking_overall.csv`, `eval_ranking_by_slice.csv`, `eval_ranking_by_support_bucket.csv`, `eval_ranking_by_pop_decile.csv`, `eval_ranking_pop_delta_vs_popularity.csv`, `eval_ranking_personalization.csv`
  - **Per-example audit trail:** `eval_offline_examples.jsonl` (candidate ids + scores per method/example)
  - **Run metadata:** `eval_offline_run_meta.json` (includes `timing_seconds`, `k_retrieval`, `k_final`, masking/model provenance, **`retrieval_bottleneck`** zero‑positive / avg‑positives in top‑`k_retrieval`, and **`slice_b_empirical_std`** per‑metric spread across slice‑B examples)

**Cached eval examples (optional, recommended for fast iteration)** — materialize a static eval examples artifact once and reuse it across notebook/model experiments:

- Job: `python scripts/recs_job_build_example_cohort.py configs/recs_job_build_eval_examples.json`  
  (legacy wrapper: `recs_job_build_eval_examples.py`)
- Outputs (default): `artifacts/recs/eval_cache/<cache_name>/`
  - `example_cohort.parquet` (legacy name: `eval_examples.parquet`)
  - `example_cohort_summary.csv`, `example_cohort_meta.json`
- **Run offline eval on that parquet (skip cohort resampling):** either  
  `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`  
  or set **`examples_parquet`** in the job config (path relative to repo root). **`run_meta["prep_diagnostics"]`** records `examples_source: parquet_cache` and the path.

**Ranker train cohort + frozen retrieval pools (Plan A)** — disjoint from val eval cache; tune rankers on train, report on val only:

- Build train cohort: `python scripts/recs_job_build_example_cohort.py configs/recs_job_build_example_cohort_train_ranker.json`
  - Writes `artifacts/recs/eval_cache/train_ranker_v1/example_cohort.parquet` (`purpose: ranker_train`, `disjoint_from_cache: val_dev_12k_v1`)
- Export pools (TF env): `python scripts/recs_job_export_retrieval_pools.py configs/recs_job_export_retrieval_pools_train_ranker.json`
  - Writes `artifacts/recs/ranker_pools/train_ranker_v1/two_tower_v1.parquet`
- Notebook: `notebooks/ranking/recs_013_ranker_d1_heuristic.ipynb` — tune `alpha` on train pools, eval on val `eval_offline_examples.jsonl`
- D2/D3 learned rankers: `notebooks/ranking/recs_014_ranker_d2_d3_train_head_to_head.ipynb`
- D4 cross-encoder spike (val rerank only): `notebooks/ranking/recs_015_ranker_d4_cross_encoder.ipynb` — requires `pip install -e '.[cross-encoder]'`
- Candidate guide: `notebooks/ranking/recs_013_ranker_d1_heuristic_candidates_learn.ipynb`

**Ranking eval (fast, frozen pools)** — after retrieval job writes `eval_offline_examples.jsonl`:

```bash
python scripts/recs_job_eval_ranking.py configs/recs_job_eval_ranking.json
# optional: --pools-jsonl path/to/eval_offline_examples.jsonl
# promote: --write-baseline  (merges into runs/latest/eval_retrieval_baseline_overall.json)
```

- Writes: `artifacts/recs/offline_eval/runs/latest_ranking/eval_ranking_*.csv`
- Reads pools jsonl **read-only**; `examples_parquet` required for personalization
- View results: `notebooks/ranking/recs_011_view_offline_ranking_eval.ipynb`

**Experiment registry** — join manifest to eval CSVs (after offline/ranking jobs):

```bash
python scripts/recs_export_experiment_registry.py
```

- Manifest: `configs/experiment_registry.yaml`
- Output: `artifacts/recs/experiment_registry_metrics.csv`
- Doc: [`experiment_registry.md`](experiment_registry.md)

**Two-tower train + eval (script-only)** — trained dual-tower model; see [`two_tower_pipeline_plan.md`](two_tower_pipeline_plan.md).

- Train: `python scripts/recs_job_train_two_tower.py configs/recs_job_train_two_tower.json`
- Writes: `artifacts/recs/towers/<run_tag>/updated_user__updated_profile200_item.keras`, `train_history.csv`, `run_metadata.json`
- Benchmark eval: add `"two_tower_v1"` to job `methods` and `"two_tower_model_path"` in config; run with `--examples-parquet` as for baselines

**View offline eval CSVs (read-only, no re-score):**

- `notebooks/retrieval/recs_011_view_offline_eval.ipynb` — set `EVAL_RUN` to `latest` or `20260526_144828__baseline_retrieval`

**Checkpoint fidelity (train save vs load):**

- `notebooks/retrieval/recs_012_checkpoint_fidelity_test.ipynb` — must-pass before trusting `two_tower_v1` eval metrics

**Retrieval mechanism comparison (e.g. baselines vs candidates)** — candidate comparison notebook (includes heavy A–F re-score section):

- `notebooks/retrieval/recs_011_eval_retrieval_two_tower_comparison.ipynb`

Other **retrieval** notebooks (under `notebooks/retrieval/`; embedding recipe notebooks stay under `notebooks/models/query_embeddings/`):

- Task A consumer (reads `eval_retrieval_*` + `eval_ranking_*`): `notebooks/retrieval/recs_004_eval_proxy_same_user_task_a_003.ipynb`
- Pipeline vs frozen baseline parity: `notebooks/retrieval/recs_009_phase1_pipeline_notebook_parity.ipynb`
- History blend grid search: `notebooks/retrieval/recs_008_history_blend_gridsearch.ipynb`
- D5 habit/session: `notebooks/ranking/recs_016_ranker_embedding_habit_session_pool.ipynb` (pool rerank); `notebooks/retrieval/recs_017_eval_habit_session_retrieval.ipynb` (full-catalog cascade/fusion)

**4-way raw/structured comparison + regression baseline (recs_006):**

- Build structured index artifact first: `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings_structured.json`
- Validate structured artifact (optional QA): `notebooks/models/game_embeddings/recs_005_game_embeddings_structured.ipynb`
- Run comparison/eval: `notebooks/models/query_embeddings/recs_006_eval_ablation_4way.ipynb`
- Save/compare `raw_raw` regression guard:

```bash
python -m pytest -q tests/test_recs_006_regression.py
```

Baseline regression compare (uses existing files at expected locations):

```bash
pytest tests/retrieval_eval_regression.py
```

Freeze/update the baseline snapshot:

```bash
python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json --write-baseline
```

Decision/log artifacts:
- `docs/retrieval_decision_log.md`
- `docs/ranking_decision_log.md`
- `artifacts/recs/retrieval/configs/active_retrieval_config.json`
- `artifacts/recs/experiments/review_style/4way_proxy/eval_review_style_4way_proxy_baseline_raw_raw.json`

Artifact layout reference:
- `docs/artifact_layout.md`

**Programmatic retrieval (v1 wire)** — `steam_review_ml.recommender.ContentRetriever` loads `artifacts/recs/` and exposes `top_k(...)` (raw or structured). Optional HTTP: TF + Hub as above, then `pip install -e '.[api]'`, then  
`uvicorn steam_review_ml.api:create_app --factory --host 127.0.0.1 --port 8000` (or `steam_review_ml.api.app:create_app`).

Endpoints: **`GET /ui`** — browser UI (game typeahead + review draft → recommendations); **`GET /games`** (`q` = optional substring on `app_name`, `limit`) for a typeahead picker; **`GET /recommendations`** with **`exclude_app_id`** set to the selected game so it never appears in results.

See [`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md) for the archived v1→v2 narrative and [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md) for the eval contract + notebook map.

---

## Order Summary

1. `python scripts/clean_reviews.py configs/clean_reviews.json`
2. `python scripts/split_reviews.py configs/split_reviews.json`
3. `python scripts/normalize_split_parquets.py configs/normalize_splits.json`
4. (Optional) run tabular modeling notebooks
5. (Recommender v1) run `python scripts/recs_job_game_profiles.py configs/recs_job_game_profiles.json`
6. Install TF + Hub (conda-forge or `.[recs-pip]`) and run `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings.json`
7.  run `python scripts/recs_job_game_embeddings.py configs/recs_job_game_embeddings_structured.json`
8. (Optional QA) run `notebooks/models/game_embeddings/recs_001_game_profile_reviews.ipynb`, `notebooks/models/game_embeddings/recs_002_game_embeddings_raw.ipynb`, and `notebooks/models/game_embeddings/recs_005_game_embeddings_structured.ipynb`
9. (Optional) run `notebooks/models/query_embeddings/recs_003_query_retrieve_smoke.ipynb` after embedding artifacts exist
10. Run `python scripts/recs_job_train_two_tower.py configs/recs_job_train_two_tower.json` to produce `updated_user__updated_profile200_item.keras`
11. Run `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json` for centralized eval artifacts (include `two_tower_v1` + `two_tower_model_path` in config to benchmark the trained tower)
12. (Optional) run `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user.ipynb` for exploratory/QA analysis (default **val**; `RECS004_EVAL_SPLIT=test` for final holdout)
13. (Optional) run `python -m pytest -q tests/test_recs_006_regression.py` after `recs_006` updates
14. (Optional) serve recommendations: `uvicorn steam_review_ml.api:create_app --factory` (requires TF + Hub + `.[api]`; pip-only stack: `.[api,recs-pip]`; repo root on `PYTHONPATH` or editable install)



## Full run

```
python scripts/clean_reviews.py configs/clean_reviews.json && \
python scripts/split_reviews.py configs/split_reviews.json && \
python scripts/normalize_split_parquets.py configs/normalize_splits.json
```