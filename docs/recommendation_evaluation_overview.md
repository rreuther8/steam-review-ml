# Recommendation Evaluation Overview

Status: active  
Owner: Ryan Reuther  
Last reviewed: 2026-04-26

This is the single summary doc for recommender evaluation work so far.

## Scope

Covers offline evaluation for text-to-game retrieval, including:

- game-profile embedding retrieval (content-based)
- query formulation comparisons (raw vs structured vs history-augmented variants)
- task framing for same-user proxy evaluation
- current release-gating decision

## Core notebooks and roles

- `notebooks/models/game_embeddings/recs_001_game_profile_reviews.ipynb`
  - builds game-profile review table
- `notebooks/models/game_embeddings/recs_002_game_embeddings_raw.ipynb`
  - builds raw game embedding index artifacts
- `notebooks/models/query_embeddings/recs_003_query_retrieve_smoke.ipynb`
  - smoke-test retrieval behavior
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a.ipynb`
  - primary recommendation-proxy evaluation task (Task A)
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_b.ipynb`
  - diagnostic task (Task B)
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_c.ipynb`
  - diagnostic task (Task C)
- `notebooks/models/query_embeddings/recs_006_eval_ablation_4way.ipynb`
  - 4-way query/index ablation (`raw_raw`, `structured_raw`, `raw_structured`, `structured_structured`)
- `notebooks/models/query_embeddings/recs_007_eval_qual_user_facing.ipynb`
  - qualitative checkpoint and failure-tag review

## Methods evaluated

### Baselines

- random ranking
- train popularity prior

### Query/index variants

- raw query text embedding
- structured query text (`extract_preferences` + `build_embedding_input`)
- query/index 4-way ablation (`recs_006`)

### History-augmented query variants (Task A family)

- multi-review mean pooling
- multi-review concatenation
- cluster-max (multi-interest)
- time-weighted blending (windowed train history)

## Evaluation tasks (A/B/C)

- **Task A (`task_a_other_val_apps`)**: positives are other val liked apps for same user (`user_apps - {q_app}`), query app masked.
- Task B (`task_b_single_holdout`): anchor/self-retrieval style diagnostic.
- Task C (`task_c_anchor_plus_other_val_apps`): mixed anchor+other positives, diagnostic.

Decision: **Task A is primary**. B/C are diagnostics only.

## Current decision and default serving path

- Primary offline benchmark: **Task A**
- Serving default: **`raw_query + raw_index`**
- Structured path: experimental / ablation only unless it wins on validation.

Source of truth:

- `docs/retrieval_decision_log.md`

## Metrics and reporting contract

Use definitions in `docs/retrieval_metrics_guide.md`.

Required reporting for each run:

- `n_total`
- `n_multi_pos`
- `n_single_pos`
- `n_zero_pos`
- coverage fractions for those groups

Decision view:

- Panel 1 (`n_pos >= 2`) as primary ranking-quality signal
- Panel 2 (`n_pos == 1`) as sparse-user coverage signal

## Where artifacts live

- `artifacts/recs/experiments/review_style/4way_proxy/eval_review_style_4way_proxy_metrics.csv`
- `artifacts/recs/experiments/review_style/4way_proxy/eval_review_style_4way_proxy_baseline_raw_raw.json`

## Related docs

- `docs/retrieval_decision_log.md` (dated decisions)
- `docs/retrieval_metrics_guide.md` (metric semantics and selection policy)
- `docs/recommender_transition_plan.md` (roadmap and architecture context)
- `docs/archive/recs_004_three_task_review.md` (task comparison review details)
