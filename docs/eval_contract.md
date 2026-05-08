# Eval Contract (v1)

## Scope
Defines the official offline evaluation slices and decision metrics for query-embedding retrieval.

## Slices (based on `n_eval_targets`)
- **Slice A: `n_eval_targets >= 2`**
  - Primary: `NDCG@10`
  - Tie-breakers: `MAP@10`, then `MRR`
- **Slice B: `n_eval_targets == 1`**
  - Primary: `Hit@10`
  - Secondary: `MRR`
- **Slice C: `n_eval_targets == 0`**
  - No ranking metrics; coverage diagnostic only

## Train-support cross section
Always report Slice A/Slice B metrics by `n_support_train` buckets:
- `0`, `1`, `2-3`, `4-7`, `8+`

## Personalization diagnostics (non-gating for v1)
- `ILD@10`
- `CatalogCoverage@10`
- `Novelty@10`
- `PersonalizationGapVsPopularity@10`

## Current gate policy (v1)
- Ranking metrics are primary for method selection.
- Personalization diagnostics are tracked to detect popularity-only behavior.
- If Slice A support is too low, report instability and treat conclusions as provisional.

## Notebook/script alignment
- Notebook reference (analysis / cleaned sweep): `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a_002.ipynb`
- Consumer of scripted retrieval artifacts: `notebooks/retrieval/recs_004_eval_proxy_same_user_task_a_003.ipynb`
- Pipeline target: `scripts/recs_job_eval_retrieval.py` + `configs/recs_job_eval_retrieval.json` (same slices/metrics).
