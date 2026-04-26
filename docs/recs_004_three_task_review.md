# recs_004 Notebook Review (Tasks A/B/C)

This file summarizes the review of:

- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a.ipynb`
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_b.ipynb`
- `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_c.ipynb`

## Decision status

- **Adopted primary evaluation task: Task A (`task_a_other_val_apps`).**
- Task B/C remain diagnostic views only and are not release-gating benchmarks.

## Executive takeaways

- **Task A is the most rigorous recommendation proxy** among the three.
- **Task B and Task C include the query app (`q_app`) in labels**, which makes results easier and less representative of true "recommend something else" retrieval.
- The three notebooks are structurally the same except for task-specific configuration (label mode, support filter mode, and cohort sizing).

## Per-task results

## Task A (`task_a_other_val_apps`)

- **Label definition:** positives are other val liked apps for that user (`user_apps - {q_app}`).
- **Masking:** query app is masked during ranking.
- **Interpretation:** best proxy for recommendation quality because it explicitly tests retrieval of *other* liked games.
- **Rigor:** **High**.

## Task B (`task_b_single_holdout`)

- **Label definition:** positives are only `{q_app}`.
- **Masking:** query app is not masked.
- **Interpretation:** mostly a self-retrieval/identity check ("can query text retrieve its own app"), useful as a sanity diagnostic.
- **Rigor for recommendation:** **Low-to-medium** (not a strong proxy for novel recommendations).

## Task C (`task_c_anchor_plus_other_val_apps`)

- **Label definition:** positives are all val liked apps for that user (`user_apps`), meaning **anchor (`q_app`) + others**.
- **Masking:** query app is not masked.
- **Interpretation:** mixed target; metrics can be boosted by anchor recovery even if retrieval of other games is weak.
- **Rigor for recommendation:** **Medium** (better than pure self-retrieval, but less strict than Task A).

## Cross-task comparability

- **Do not compare A, B, and C as if they are the same benchmark.**
- Task B/C allow anchor hits from `q_app`, while Task A explicitly removes that path.
- If reporting all three, treat B/C as diagnostics and Task A as the primary offline recommendation metric.

## Additional review notes

- A/B/C differ only in one config cell and one introductory markdown line; core evaluation code is shared.
- Cohort sizing includes a fallback bucket (`_fallback_any`) when requested cohort bins are undersupplied. This is practical, but can shift evaluation mix and should be reported with results.
- No embedded notebook execution error outputs were found during review.

## Recommended reporting format

When sharing results, use:

1. **Primary table:** Task A metrics.
2. **Diagnostic appendix:** Task B and Task C metrics.
3. **Context block:** sampled cohort composition (`df_sizing`) and evaluable example counts.

