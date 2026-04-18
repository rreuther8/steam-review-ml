# Retrieval Metrics Guide

This document defines the ranking metrics used in recommender notebooks (`recs_004`, `recs_006`, `recs_008`) and how we aggregate them.

## Relevance setup

- Relevance is binary.
- A game is relevant (`1`) if it is in the positive set for that query/user; otherwise `0`.
- In same-user proxy tasks, positives are other liked games for the same user (excluding the query game).

## Metric definitions

- `Hit@K`
  - `1` if at least one relevant game appears in top `K`; else `0`.
  - Interprets as success-rate at `K`.

- `Precision@K`
  - `(# relevant games in top K) / K`.
  - Measures concentration of relevant results near the top.

- `Recall@K`
  - `(# relevant games in top K) / (# relevant games total for that query)`.
  - Measures coverage of the positive set by rank `K`.

- `MAP@K` (mean average precision at `K`)
  - For each relevant hit at rank `r <= K`, add `precision_at_r`.
  - Divide by total number of positives for that query.
  - Rewards ranking relevant items earlier.

- `NDCG@K`
  - `DCG@K = sum(rel_i / log2(i + 1))` with rank index `i` starting at 1.
  - `NDCG@K = DCG@K / IDCG@K`, where `IDCG@K` is ideal DCG for that query.
  - Normalized to compare across users with different positive counts.

- `MRR` (mean reciprocal rank)
  - For one query: `1 / rank_of_first_relevant`, or `0` if none found.
  - Emphasizes getting at least one strong hit very early.

## Aggregation across users/queries

Use the same semantics as `recs_004`:

- `Hit@K`: arithmetic mean (`mean`).
- `Recall@K`, `MAP@K`, `NDCG@K`, `MRR`: `nanmean`.
  - Users/queries with no positives produce `NaN` for these metrics and are excluded from those averages.

## Practical interpretation

- If `Hit@K` improves but `MAP/NDCG` do not, we are finding at least one relevant item but ranking quality is not improving much.
- If `Recall@K` improves with stable precision, we are covering more relevant items without adding too much noise.
- If `MRR` improves, first relevant hit appears earlier (better user-perceived quality for top results).
