# Documentation index

Use this file to avoid **doc sprawl**: prefer these paths before adding new top-level markdown.

## Start here (operators)

| Doc | Purpose |
|-----|---------|
| [usage_pipeline.md](usage_pipeline.md) | Commands, configs, artifact outputs, notebook pointers |
| [artifact_layout.md](artifact_layout.md) | Where things live under `artifacts/recs/` |
| [eval_contract.md](eval_contract.md) | Offline eval artifact naming + consumer expectations |

## Retrieval / ranking / metrics

| Doc | Purpose |
|-----|---------|
| [retrieval_metrics_guide.md](retrieval_metrics_guide.md) | Metric definitions used in notebooks |
| [recommendation_evaluation_overview.md](recommendation_evaluation_overview.md) | High-level recap of evaluation threads |
| [retrieval_decision_log.md](retrieval_decision_log.md) | Decision trail |
| [retrieval_embedding_matrices_graph.md](retrieval_embedding_matrices_graph.md) | Embedding matrix relationships |
| [retrieval_eval_cached_examples_plan.md](retrieval_eval_cached_examples_plan.md) | Cached eval examples design |

## Planning / roadmap

| Doc | Purpose |
|-----|---------|
| [recommender_transition_plan.md](recommender_transition_plan.md) | Product/engineering north star |
| [project_todo_plan.md](project_todo_plan.md) | Repo-wide prioritized todos |
| [todo/retrieval_ranking_eval_split_todo.md](todo/retrieval_ranking_eval_split_todo.md) | Retrieval vs ranking split phases |
| [todo/recs_XXX_deferred_integration_notebook.md](todo/recs_XXX_deferred_integration_notebook.md) | Deferred: optional **`recs_XXX`** walkthrough notebook (post-MVP) |
| [todo/simplification_todo.md](todo/simplification_todo.md) | Simplification backlog |

## Product / hiring / misc

| Doc | Purpose |
|-----|---------|
| [product_vision_recommender_and_review_coaching.md](product_vision_recommender_and_review_coaching.md) | Product framing |
| [applied_ai_hiring_readiness_note.md](applied_ai_hiring_readiness_note.md) | Interview / portfolio notes |

## Historical / draft (do not duplicate upstream)

- **`archive/`** — superseded writeups (metrics, recs_004 review, EDA, ETL notes).
- **`draft/`** — unfinished; not guaranteed current.

**Rule of thumb:** extend **usage_pipeline** / **eval_contract** / **artifact_layout** for truth about *how to run*; use **transition plan** + **project_todo_plan** for *what to build next*; retire overlapping prose into **`archive/`** rather than adding a parallel doc.
