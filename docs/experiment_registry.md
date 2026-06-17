# Experiment registry

Status: active  
Last updated: 2026-06-16

Single inventory of retrieval/ranking experiments: **what** was tried, **status**, **where evidence lives**, and (for eval-job-wired methods) **latest val metrics** on the frozen cohort.

**Machine-readable:**

| File | Role |
|------|------|
| [`configs/experiment_registry.yaml`](../configs/experiment_registry.yaml) | Manifest (metadata only) |
| [`artifacts/recs/experiment_registry_metrics.csv`](../artifacts/recs/experiment_registry_metrics.csv) | Manifest + joined metrics (regenerate after eval runs) |

**Regenerate metrics:**

```bash
python scripts/recs_export_experiment_registry.py
```

Requires `pip install pyyaml` if not already installed.

---

## Eval contract (all joined rows)

| Field | Value |
|-------|-------|
| Cohort | `val_dev_12k_v1` (12.5k examples) |
| Parquet | `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet` |
| Retrieval cutoff | `k_retrieval=100` → `eval_retrieval_*` |
| Ranking cutoff | `k_final=10` → `eval_ranking_*` |

Full job semantics: [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md).

---

## Shipped v1 stack

```text
two_tower_v1 @100  →  two_tower_v1_heuristic_logpop_blend @10
```

| Stage | method_id | Hit@K | NDCG@K | Notes |
|-------|-----------|------:|-------:|-------|
| Retrieve | `two_tower_v1` | 0.512 | — | Recall@100 = 0.494 |
| Rank | `two_tower_v1_heuristic_logpop_blend` | 0.193 | **0.093** | Shipped D1 (alpha=0.2 log-pop within pool) |

Gating baseline (rank-only job): `popularity_train` NDCG@10 = 0.073.

---

## v1 retrieval benchmarks (wired)

| method_id | Hit@100 | Recall@100 | status |
|-----------|--------:|-----------:|--------|
| `popularity_train` | 0.763 | 0.749 | benchmark |
| `two_tower_v1` | **0.512** | **0.494** | **shipped** |
| `fusion_c_raw_plus_behavior` | 0.506 | 0.488 | benchmark |
| `multi_mean_train` | 0.486 | 0.467 | benchmark |
| `raw` | 0.453 | 0.435 | benchmark |

---

## v1 ranker spikes (not in eval job)

Killed challengers — metrics from notebooks (`metrics_joined=false` in CSV). See [`ranker_exploration_plan.md`](ranker_exploration_plan.md) for full matrix.

| experiment_id | Val NDCG@10 (notebook) | status |
|---------------|------------------------:|--------|
| `v1_rank_d1_heuristic_logpop` | **0.093** | **shipped** |
| `v1_rank_d4_ce_blend` | ~0.091 | killed |
| `v1_rank_d2_pointwise` | ~0.089 | killed |
| `v1_rank_d3_listwise` | ~0.085 | killed |
| `v1_rank_d5_session_habit` | ~0.040 | killed |
| `v1_rank_d6a_frozen_trunk` / `d6b` | below D1 | killed |

---

## v2 placeholders (planned)

Rank-only on frozen `two_tower_v1` pools; beat D1 on overall + slice A. See [`plans/recommender_v2_questionnaire.md`](plans/recommender_v2_questionnaire.md).

| experiment_id | method_id (planned) | status |
|---------------|---------------------|--------|
| `v2_rank_v2b_query_metadata` | `two_tower_v1_v2b_query_metadata` | planned (spike first) |
| `v2_rank_v2a_igdb_summary` | `two_tower_v1_v2a_igdb_summary` | planned |
| `v2_rank_v2b_history_metadata` | `two_tower_v1_v2b_history_metadata` | planned |
| `v2_rank_v2c_query_combined` | `two_tower_v1_v2c_query_summary_metadata` | planned |
| `v2_rank_v2d_primary_genre` | `two_tower_v1_v2d_primary_genre_metadata` | planned |
| `v2_cf_als_deferred` | `two_tower_v1_cf_als` | deferred (v2.1) |

When a v2 method is wired into eval jobs, set `eval_job_wired: true` in the YAML and re-export.

---

## Column glossary (`experiment_registry_metrics.csv`)

| Column | Meaning |
|--------|---------|
| `experiment_id` | Stable row id in manifest |
| `method_id` | Scorer name in eval CSVs (when wired) |
| `stage` | `retrieval` or `ranking` |
| `status` | `shipped`, `benchmark`, `killed`, `planned`, `deferred` |
| `eval_job_wired` | Whether method runs in `recs_job_eval_offline` / `recs_job_eval_ranking` |
| `metrics_source` | `offline_eval`, `ranking_eval`, or `notebook` |
| `metrics_joined` | Whether this export run found metrics in eval CSVs |
| `Hit@K`, `NDCG@K`, … | From eval artifact; K is symbolic (@100 retrieval, @10 ranking) |

**Rule:** Do not store live metrics in YAML — only in exported CSV (or notebooks for killed spikes).

---

## Related docs

- Implementation plan: [`plans/experiment_registry_plan.md`](plans/experiment_registry_plan.md)
- Retrieval decisions: [`retrieval_decision_log.md`](retrieval_decision_log.md)
- Ranking decisions: [`ranking_decision_log.md`](ranking_decision_log.md)
