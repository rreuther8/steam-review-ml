# Eval Contract (v2)

## Scope
Defines the official offline evaluation slices and decision metrics for:
- retrieval-stage quality (candidate generation; order-insensitive)
- ranking-stage quality (retrieval then ranking; order-sensitive)

## Slices (based on `n_eval_targets`)
- **Slice A: `n_eval_targets >= 2`**
- **Slice B: `n_eval_targets == 1`**
- **Slice C: `n_eval_targets == 0`**

## Retrieval metrics policy (order-insensitive)
- **Slice A (`n_eval_targets >= 2`)**
  - Primary: `Recall@10`
  - Secondary: `Precision@10`
- **Slice B (`n_eval_targets == 1`)**
  - Primary: `Hit@10`
  - Secondary: `Precision@10`
- **Slice C (`n_eval_targets == 0`)**
  - No relevance metrics; coverage diagnostic only

## Ranking metrics policy (order-sensitive)
- **Slice A (`n_eval_targets >= 2`)**
  - Primary: `NDCG@10`
  - Tie-breakers: `MAP@10`, then `MRR`
- **Slice B (`n_eval_targets == 1`)**
  - Primary: `Hit@10`
  - Secondary: `MRR`
- **Slice C (`n_eval_targets == 0`)**
  - No ranking metrics; coverage diagnostic only

## Cross sections and required context
- Always report Slice A/Slice B metrics by `n_support_train` buckets:
  - `0`, `1`, `2-3`, `4-7`, `8+`
- Always report coverage fields:
  - `n_total`, `n_multi_pos`, `n_single_pos`, `n_zero_pos`, `coverage_multi_pos`

## Personalization and ecosystem diagnostics (non-gating for v2)
- `CatalogCoverage@10` (useful for retrieval and ranking)
- `Novelty@10` (useful for retrieval and ranking)
- `PersonalizationGapVsPopularity@10` (useful for retrieval and ranking)
- `ILD@10` (primarily useful for ranking diversity; secondary for retrieval)

## Gate policy (v2)
- **Retrieval-stage selection:** prioritize retrieval policy metrics (`Recall@10` for Slice A, `Hit@10` for Slice B).
- **Ranking-stage selection:** prioritize ranking policy metrics (`NDCG@10` for Slice A, `Hit@10` for Slice B).
- Personalization diagnostics are guardrails to detect popularity-only or low-diversity behavior.
- If Slice A support is too low, report instability and treat conclusions as provisional.

## Notebook/script alignment
- Notebook reference (analysis / cleaned sweep): `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a_002.ipynb`
- Consumer of scripted retrieval artifacts: `notebooks/retrieval/recs_004_eval_proxy_same_user_task_a_003.ipynb`
- Pipeline target: `scripts/recs_job_eval_retrieval.py` + `configs/recs_job_eval_retrieval.json` (same slices/metrics).
