# Recommendation evaluation (overview + contract)

Status: active  
Owner: Ryan Reuther  
Last reviewed: 2026-05-11

Single place for **what we evaluate**, **how slices and metrics gate decisions**, and **where notebooks and jobs line up**. The former standalone `eval_contract.md` and `retrieval_eval_cached_examples_plan.md` are merged here (2026-05-11). Metric definitions remain in [`retrieval_metrics_guide.md`](retrieval_metrics_guide.md).

## Scope

Offline evaluation for text-to-game retrieval:

- game-profile embedding retrieval (content-based)
- query formulation comparisons (raw vs structured vs history-augmented variants)
- task framing for same-user proxy evaluation
- scripted job outputs under `artifacts/recs/offline_eval/` (see [`usage_pipeline.md`](usage_pipeline.md), [`artifact_layout.md`](artifact_layout.md))

## Core notebooks and roles

- `notebooks/models/game_embeddings/recs_001_game_profile_reviews.ipynb` — game-profile review table
- `notebooks/models/game_embeddings/recs_002_game_embeddings_raw.ipynb` — raw game embedding index
- `notebooks/models/query_embeddings/recs_003_query_retrieve_smoke.ipynb` — smoke-test retrieval
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a.ipynb` — **Task A** (primary proxy)
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_b.ipynb` — diagnostic Task B
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_c.ipynb` — diagnostic Task C
- `notebooks/models/query_embeddings/recs_006_eval_ablation_4way.ipynb` — 4-way query/index ablation
- `notebooks/models/query_embeddings/recs_007_eval_qual_user_facing.ipynb` — qualitative checkpoint / failure tags
- `notebooks/retrieval/recs_011_eval_retrieval_two_tower_comparison.ipynb` — candidate comparison vs the same contract as `recs_job_eval_retrieval.py`
- [`two_tower_pipeline_plan.md`](two_tower_pipeline_plan.md) — script-only train + eval runbook for learned two-tower (`updated_user__updated_profile200_item`)

## Methods evaluated

**Baselines:** random ranking; train popularity prior.

**Query/index:** raw text embedding; structured (`extract_preferences` + `build_embedding_input`); 4-way ablation (`recs_006`).

**History-augmented (Task A family):** multi-review mean / concat, cluster-max, time-weighted train pooling, etc.

## Evaluation tasks (A / B / C)

- **Task A (`task_a_other_val_apps`)** — positives are other val liked apps for the same user (`user_apps - {q_app}`); query app masked. **Primary** benchmark.
- **Task B** — anchor-style diagnostic.
- **Task C** — mixed diagnostic.

## Current decision and default serving path

- Primary offline benchmark: **Task A**
- Serving default: **`raw_query + raw_index`**
- Structured path: experimental / ablation unless it wins on the same contract on validation.

Authoritative dated rationale: [`retrieval_decision_log.md`](retrieval_decision_log.md).

## Metrics and reporting (summary)

Full definitions and interview-style framing: [`retrieval_metrics_guide.md`](retrieval_metrics_guide.md).

Required reporting for each run:

- `n_total`, `n_multi_pos`, `n_single_pos`, `n_zero_pos`, and coverage fractions for those groups

Decision view (maps to contract slices via `n_eval_targets`):

- **Slice A** (`n_eval_targets >= 2`) — primary ranking-quality signal
- **Slice B** (`n_eval_targets == 1`) — sparse-user coverage signal
- **Slice C** (`n_eval_targets == 0`) — coverage-only; no relevance metrics per contract

## Where artifacts live

- Central eval outputs: `artifacts/recs/offline_eval/runs/latest/` (see [`artifact_layout.md`](artifact_layout.md))
- Legacy 4-way CSV / JSON baselines: `artifacts/recs/experiments/review_style/4way_proxy/`

## Related docs

- [`retrieval_decision_log.md`](retrieval_decision_log.md) — dated decisions
- [`retrieval_metrics_guide.md`](retrieval_metrics_guide.md) — metric semantics
- [`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md) — archived v1→v2 engineering narrative
- [`archive/recs_004_three_task_review.md`](archive/recs_004_three_task_review.md) — task A/B/C comparison detail

---

## Eval contract (v2)

Defines official offline **slices** and **decision metrics** for:

- **Retrieval-stage** quality (candidate generation; order-insensitive)
- **Ranking-stage** quality (retrieval then ranking; order-sensitive)

**Cutoffs** (from `configs/recs_job_eval_retrieval.json`, or equivalent job parameters):

- **`k_retrieval`**: depth for **retrieval-contract** metrics (Hit / Precision / Recall on the retrieved candidate list). Typical value: **100**.
- **`k_final`**: depth for **ranking-contract** metrics (MAP, NDCG, MRR, and Hit / Precision / Recall on **`eval_ranking_*`** tables). Typical value: **10**.
- **`k_personalization`**: depth for **short-list guardrail** diagnostics (column names include the numeric suffix, e.g. `CatalogCoverage@10`). Typical value: **10**.

**Column naming:** summaries use labels like `Hit@K` and `NDCG@K`; **K is symbolic**. Interpret each column using the contract below (retrieval vs ranking vs personalization), not the letter K alone.

### Slices (based on `n_eval_targets`)

- **Slice A: `n_eval_targets >= 2`**
- **Slice B: `n_eval_targets == 1`**
- **Slice C: `n_eval_targets == 0`**

### Retrieval metrics policy (order-insensitive)

Computed at **`k_retrieval`** (columns still labeled `Hit@K`, `Precision@K`, `Recall@K` in `eval_retrieval_*`).

- **Slice A (`n_eval_targets >= 2`)** — Primary: `Recall@K`; secondary: `Precision@K`; tertiary: `Hit@K`
- **Slice B (`n_eval_targets == 1`)** — Primary: `Hit@K`; secondary: `Precision@K`; tertiary: `Recall@K`
- **Slice C (`n_eval_targets == 0`)** — No relevance metrics; coverage diagnostic only

### Ranking metrics policy (order-sensitive)

Computed at **`k_final`** (columns `MAP@K`, `NDCG@K`, `MRR`, and Hit / Precision / Recall on **`eval_ranking_*`**).

**Oracle ceiling (diagnostic, same tables):** `OracleHit@K` and `OracleNDCG@K` on **`eval_ranking_*`** report the best possible top-`k_final` order **within each method’s retrieved top-`k_retrieval` pool** (positives ranked first). Compare to actual `Hit@K` / `NDCG@K` to see ranker headroom; not a competing method.

- **Slice A (`n_eval_targets >= 2`)** — Primary: `NDCG@K`; tie-breakers: `MAP@K`, then `MRR`, then remaining ranking columns as needed
- **Slice B (`n_eval_targets == 1`)** — Primary: `Hit@K`; secondary: `MRR`; tie-breakers: `NDCG@K`, `MAP@K`, …
- **Slice C (`n_eval_targets == 0`)** — No ranking metrics; coverage diagnostic only

### Cross sections and required context

- Always report Slice A / Slice B metrics by `n_support_train` buckets: `0`, `1`, `2-3`, `4-7`, `8+`
- Always report coverage fields: `n_total`, `n_multi_pos`, `n_single_pos`, `n_zero_pos`, `coverage_multi_pos`

### Personalization and ecosystem diagnostics (non-gating for v2)

Guardrails use **`k_personalization`** (suffix in column names matches that integer, e.g. when `k_personalization = 10`: `CatalogCoverage@10`, `Novelty@10`, `PersonalizationGapVsPopularity@10`, `ILD@10`).

- `CatalogCoverage@{k_personalization}` (retrieval and ranking)
- `Novelty@{k_personalization}` (retrieval and ranking)
- `PersonalizationGapVsPopularity@{k_personalization}` (retrieval and ranking)
- `ILD@{k_personalization}` (primarily ranking diversity; secondary for retrieval)

### Gate policy (v2)

- **Retrieval-stage selection:** prioritize retrieval policy metrics at **`k_retrieval`** (`Recall@K` for Slice A, `Hit@K` for Slice B).
- **Ranking-stage selection:** prioritize ranking policy metrics at **`k_final`** (`NDCG@K` for Slice A, `Hit@K` for Slice B).
- Personalization diagnostics are guardrails for popularity-only or low-diversity behavior.
- If Slice A support is too low, report instability and treat conclusions as provisional.

### Notebook / script alignment

- Notebook reference (analysis / cleaned sweep): `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a_002.ipynb`
- Consumer of scripted retrieval artifacts: `notebooks/retrieval/recs_004_eval_proxy_same_user_task_a_003.ipynb`
- Pipeline target: `scripts/recs_job_eval_retrieval.py` + `configs/recs_job_eval_retrieval.json` (same slices / metrics)
- Candidate comparison (dual retrieval / ranking tables): `notebooks/retrieval/recs_011_eval_retrieval_two_tower_comparison.ipynb`

---

## Cached offline eval examples

**Goal:** Avoid rebuilding eval cohorts on every experiment run via a small, config-driven path that materializes reusable eval inputs.

**Problem:** `prepare_eval_inputs(...)` can rebuild selected examples each run; notebook iteration pays repeated prep cost; dev subsets are not always versioned as artifacts.

**Shape of the solution (two steps):**

1. **Build cached eval examples** — config (split / cohort / sampling / prep knobs) → frozen examples + metadata on disk.
2. **Evaluate methods against cached examples** — cached artifact + method config → existing `eval_retrieval_*` / `eval_ranking_*` tables.

**Implemented knobs:** `scripts/recs_job_eval_retrieval.py` accepts **`--examples-parquet PATH`** (or config key **`examples_parquet`**, path relative to repo root) and uses `prepare_eval_inputs_from_cache(...)` instead of resampling `prepare_eval_inputs`. When set, **`max_examples`** / cohort knobs do not re-sample the cohort — the parquet fixes who is evaluated; keep config aligned with how the parquet was built.

**Optional future / build job:** a dedicated cache builder (e.g. `configs/recs_job_build_eval_examples.json`, `scripts/recs_job_build_eval_examples.py`) can wrap `prepare_eval_inputs` once and write:

- `artifacts/recs/eval_cache/<cache_name>/eval_examples.parquet`
- `eval_examples_meta.json`, `eval_examples_summary.csv`

Schema expectations (parquet): stable serialization for list-like fields; include `ex_idx`, `user_id`, `query_app_id`, `query_text`, `n_eval_targets`, `validation_positive_app_ids`, `support_texts_train`, `train_review_rows`, and QA helpers like `slice_name` / `train_support_bucket` at build time where applicable.

**Split naming (interview-friendly):** `train` for fit; a named **val dev cache** for fast frozen comparisons; **val full** for periodic gates; **test** for one-time holdout.

**Guardrails:** validate required columns and dtypes; validate `n_eval_targets` / `slice_name` consistency; optional strict fingerprint match to source splits; avoid accidental overwrite without `--overwrite`.

**Rollout checklist:** cache builder + eval integration (done for eval path); regression test for schema / determinism; runbook + `recs_011` can consume the same `eval_examples.parquet` locally while the batch job uses `--examples-parquet`.
