# Recommender v1 Wrap-Up

Status: complete (ranking superseded by v2a — see [`ranking_decision_log.md`](ranking_decision_log.md) § 2026-06-22)  
Date: 2026-06-16

This document closes v1 for the chosen scope: content-led retrieval with a practical ranker, evaluated on the frozen `val_dev_12k_v1` cohort.

> **Ranking update:** v1 shipped D1; v2a `two_tower_v1_v2a_embed_query_logpop_blend` is now the production ranker. D1 remains a benchmark in eval jobs.

## Scope closed in v1

- Retrieval mechanism: `two_tower_v1` at `k_retrieval=100`
- Ranking mechanism: `two_tower_v1_heuristic_logpop_blend` (D1) at `k_final=10`
- Centralized offline evaluation: `recs_job_eval_offline` + `recs_job_eval_ranking`
- Regression baseline refreshed and passing
- Experiment inventory established (`configs/experiment_registry.yaml` + export script + docs)

## Shipped pipeline

```text
two_tower_v1 @100  ->  two_tower_v1_heuristic_logpop_blend @10
```

Eval contract:

- Cohort: `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`
- Retrieval contract: `eval_retrieval_*` at `k_retrieval=100`
- Ranking contract: `eval_ranking_*` at `k_final=10`

## Final v1 metrics snapshot

From latest eval artifacts:

- Retrieval (`two_tower_v1`): Hit@100 `0.512`, Recall@100 `0.494`
- Ranking (`two_tower_v1_heuristic_logpop_blend`): Hit@10 `0.193`, NDCG@10 `0.093`, MRR `0.067`
- Ranking baseline (`popularity_train`): Hit@10 `0.151`, NDCG@10 `0.073`, MRR `0.052`

Interpretation: v1 shipped method beats full-catalog popularity on headline ranking metrics while retaining materially more personalization than popularity-only ranking.

## Key decisions finalized

- Retrieval choice locked: `two_tower_v1` is the retrieve stage for the eval/rank pipeline.
- Ranker choice locked: D1 shipped for v1; D2-D6 challengers did not beat D1 on the promotion bar.
- Eval naming clarified: full job is `recs_job_eval_offline`; rank-only job remains `recs_job_eval_ranking`.
- Experiment tracking formalized via registry manifest + export.

## Known v1 limitations (accepted)

- D1 is popularity-heavy (`alpha=0.2` blend) and trades off some personalization for relevance.
- **API serving:** default is now the v2a stacked path (`StackedRecommender` + `configs/recs_serve.json`); see [`usage_pipeline.md`](usage_pipeline.md).
- No frozen test-holdout gate in v1 closeout (left as optional post-v1 science backlog).

## Hand-off to v2

v2 starts as rank-only work on frozen `two_tower_v1` pools:

- Use [`docs/recommender_v2_plan.md`](recommender_v2_plan.md) for execution; questionnaire remains the decision archive.
- Use `configs/experiment_registry.yaml` for new v2 rows and status tracking.
- Promotion bar remains: beat D1 on overall + Slice A under the same eval contract.

## Reference docs

- `docs/retrieval_decision_log.md`
- `docs/ranking_decision_log.md`
- `docs/recommendation_evaluation_overview.md`
- `docs/project_todo_plan.md`
- `docs/experiment_registry.md`
