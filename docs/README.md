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
| [retrieval_decision_log.md](retrieval_decision_log.md) | Decision trail |
| [retrieval_embedding_matrices_graph.md](retrieval_embedding_matrices_graph.md) | Embedding matrix relationships |

## Planning / roadmap

| Doc | Purpose |
|-----|---------|
| [project_todo_plan.md](project_todo_plan.md) | Repo-wide prioritized todos + focused backlog (retrieval/ranking split, simplification, deferred notebook) |
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
