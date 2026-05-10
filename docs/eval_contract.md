# Eval Contract (v2)

## Scope
Defines the official offline evaluation slices and decision metrics for:
- retrieval-stage quality (candidate generation; order-insensitive)
- ranking-stage quality (retrieval then ranking; order-sensitive)

**Cutoffs** (from `configs/recs_job_eval_retrieval.json`, or equivalent job parameters):
- **`k_retrieval`**: depth for **retrieval-contract** metrics (Hit/Precision/Recall on the retrieved candidate list). Typical value: **100**.
- **`k_final`**: depth for **ranking-contract** metrics (MAP, NDCG, MRR, and the Hit/Precision/Recall columns on **`eval_ranking_*`** tables). Typical value: **10**.
- **`k_personalization`**: depth for **short-list guardrail** diagnostics (column names include the numeric suffix, e.g. `CatalogCoverage@10`). Typical value: **10**.

**Column naming:** summaries use labels like `Hit@K` and `NDCG@K`; **K is symbolic**. Interpret each column using the contract row below (retrieval vs ranking vs personalization), not the letter K alone.

## Slices (based on `n_eval_targets`)
- **Slice A: `n_eval_targets >= 2`**
- **Slice B: `n_eval_targets == 1`**
- **Slice C: `n_eval_targets == 0`**

## Retrieval metrics policy (order-insensitive)
Computed at **`k_retrieval`** (columns still labeled `Hit@K`, `Precision@K`, `Recall@K` in `eval_retrieval_*`).

- **Slice A (`n_eval_targets >= 2`)**
  - Primary: `Recall@K`
  - Secondary: `Precision@K`
  - Tertiary: `Hit@K`
- **Slice B (`n_eval_targets == 1`)**
  - Primary: `Hit@K`
  - Secondary: `Precision@K`
  - Tertiary: `Recall@K`
- **Slice C (`n_eval_targets == 0`)**
  - No relevance metrics; coverage diagnostic only

## Ranking metrics policy (order-sensitive)
Computed at **`k_final`** (columns `MAP@K`, `NDCG@K`, `MRR`, and Hit/Precision/Recall on **`eval_ranking_*`**).

- **Slice A (`n_eval_targets >= 2`)**
  - Primary: `NDCG@K`
  - Tie-breakers: `MAP@K`, then `MRR`, then remaining ranking columns as needed
- **Slice B (`n_eval_targets == 1`)**
  - Primary: `Hit@K`
  - Secondary: `MRR`
  - Tie-breakers: `NDCG@K`, `MAP@K`, …
- **Slice C (`n_eval_targets == 0`)**
  - No ranking metrics; coverage diagnostic only

## Cross sections and required context
- Always report Slice A/Slice B metrics by `n_support_train` buckets:
  - `0`, `1`, `2-3`, `4-7`, `8+`
- Always report coverage fields:
  - `n_total`, `n_multi_pos`, `n_single_pos`, `n_zero_pos`, `coverage_multi_pos`

## Personalization and ecosystem diagnostics (non-gating for v2)
Guardrails use **`k_personalization`** (suffix in column names matches that integer, e.g. when `k_personalization = 10`: `CatalogCoverage@10`, `Novelty@10`, `PersonalizationGapVsPopularity@10`, `ILD@10`).

- `CatalogCoverage@{k_personalization}` (useful for retrieval and ranking)
- `Novelty@{k_personalization}` (useful for retrieval and ranking)
- `PersonalizationGapVsPopularity@{k_personalization}` (useful for retrieval and ranking)
- `ILD@{k_personalization}` (primarily useful for ranking diversity; secondary for retrieval)

## Gate policy (v2)
- **Retrieval-stage selection:** prioritize retrieval policy metrics at **`k_retrieval`** (`Recall@K` for Slice A, `Hit@K` for Slice B).
- **Ranking-stage selection:** prioritize ranking policy metrics at **`k_final`** (`NDCG@K` for Slice A, `Hit@K` for Slice B).
- Personalization diagnostics are guardrails to detect popularity-only or low-diversity behavior.
- If Slice A support is too low, report instability and treat conclusions as provisional.

## Notebook/script alignment
- Notebook reference (analysis / cleaned sweep): `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a_002.ipynb`
- Consumer of scripted retrieval artifacts: `notebooks/retrieval_ranking/recs_004_eval_proxy_same_user_task_a_003.ipynb`
- Pipeline target: `scripts/recs_job_eval_retrieval.py` + `configs/recs_job_eval_retrieval.json` (same slices/metrics).
- Candidate comparison (same contract, dual retrieval/ranking tables): `notebooks/retrieval_ranking/recs_011_eval_retrieval_two_tower_comparison.ipynb`
