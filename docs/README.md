# Documentation index

Use this file to avoid **doc sprawl**: prefer these paths before adding new top-level markdown.

## Start here (operators)

| Doc | Purpose |
|-----|---------|
| [project_summary.md](project_summary.md) | One-page repo orientation (layout, flow, status, entry points) |
| [usage_pipeline.md](usage_pipeline.md) | Commands, configs, artifact outputs, notebook pointers |
| [artifact_layout.md](artifact_layout.md) | Where things live under `artifacts/recs/` |
| [recommendation_evaluation_overview.md](recommendation_evaluation_overview.md) | Eval scope, notebook map, **eval contract (v2)**, cached-examples runbook |

## Retrieval / ranking / metrics

| Doc | Purpose |
|-----|---------|
| [retrieval_metrics_guide.md](retrieval_metrics_guide.md) | Metric definitions used in notebooks and jobs |
| [retrieval_decision_log.md](retrieval_decision_log.md) | Retrieval decision trail |
| [ranking_decision_log.md](ranking_decision_log.md) | Ranking ship/kill/defer trail (D1–D5) |
| [retrieval_embedding_matrices_graph.md](retrieval_embedding_matrices_graph.md) | Embedding matrix relationships |

## Planning / roadmap

| Doc | Purpose |
|-----|---------|
| [project_todo_plan.md](project_todo_plan.md) | Repo-wide prioritized todos + focused backlog (retrieval/ranking split, simplification, deferred notebook) |
| [ranker_exploration_plan.md](ranker_exploration_plan.md) | **Fill-in** questionnaire + decisions for ranker / rerank next steps |
| [experiment_registry.md](experiment_registry.md) | **v1 experiment inventory** + export runbook |
| [plans/experiment_registry_plan.md](plans/experiment_registry_plan.md) | Registry implementation notes (manifest + export) |
| [plans/recommender_v2_questionnaire.md](plans/recommender_v2_questionnaire.md) | Frozen v2 decisions (IGDB hybrid) — input to plan |
| [recommender_v2_plan.md](recommender_v2_plan.md) | **Active** v2 execution plan (spike order, eval, promotion bar) |
| [two_tower_pipeline_plan.md](two_tower_pipeline_plan.md) | Two-tower train + eval runbook (script-only) |
| [archive/recommender_transition_plan.md](archive/recommender_transition_plan.md) | **Archived** v1→v2 engineering narrative |

## Product / hiring / misc

| Doc | Purpose |
|-----|---------|
| [product_vision_recommender_and_review_coaching.md](product_vision_recommender_and_review_coaching.md) | Product framing |
| [applied_ai_hiring_readiness_note.md](applied_ai_hiring_readiness_note.md) | Interview / portfolio notes |

## Historical / draft (do not duplicate upstream)

- **`archive/`** — superseded writeups (metrics, recs_004 review, EDA, ETL notes, transition plan).
- **`draft/`** — unfinished; not guaranteed current.

**Rule of thumb:** extend **usage_pipeline**, **recommendation_evaluation_overview**, and **artifact_layout** for truth about *how to run* and *what the eval contract expects*; use **project_todo_plan** for *what to build next*; retire overlapping prose into **`archive/`** rather than adding a parallel doc.
