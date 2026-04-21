# Retrieval Metrics Guide

This document defines the ranking metrics used in recommender notebooks (`recs_004`, `recs_006`, `recs_008`) and how we aggregate them.

## Relevance setup

- Relevance is binary.
- A game is relevant (`1`) if it is in the positive set for that query/user; otherwise `0`.
- In same-user proxy tasks, positives are other liked games for the same user (excluding the query game).
- Answers: "Is this recommended game something the user has liked (i.e., in the positives set for this user/query)?"

## Metric definitions

### Hit@K

- **Definition**: `1` if at least one relevant game appears in top `K`; else `0`.
- **Interpretation**: success-rate at `K`.
- **Question**: "Did we get at least one good recommendation in the top `K`?"

### Precision@K

- **Definition**: `(# relevant games in top K) / K`.
- **Interpretation**: concentration of relevant results near the top.
- **Question**: "Of the `K` games we showed, what fraction were actually relevant?"

### Recall@K

- **Definition**: `(# relevant games in top K) / (# relevant games total for that query)`.
- **Interpretation**: coverage of the positive set by rank `K`.
- **Question**: "What fraction of all relevant games did we manage to surface by rank `K`?"

### MAP@K (mean average precision at K)

- **Interpretation**: rewards ranking *all* relevant items earlier (not just the first one).
- **Question**: "Across the top `K`, how consistently early do relevant games appear?"

For a single query/user, define *average precision at K*:

```
AP@K = (1 / P) × Σ_{r=1..K} (Precision@r × rel_r)
```

where:

- `rel_r ∈ {0,1}` indicates whether the item at rank `r` is relevant
- `Precision@r = (# relevant items in ranks 1..r) / r`
- `P` is the number of positives for that query/user (if `P=0`, treat as `NaN`)

Note: if a user has more than `K` positives, then dividing by `P` means the maximum possible `AP@K` is `< 1`. (Some definitions use `min(P, K)` instead; we keep `P` here to align with the "recall-style" normalization above.)

MAP@K is the mean of AP@K over queries/users (see aggregation section).

- Worked example (`K=5`), relevances by rank: `[1, 0, 1, 0, 1]` and total positives `P=3`.
  - Precision@1 `= 1/1 = 1.0` (hit)
  - Precision@3 `= 2/3 ≈ 0.6667` (hit)
  - Precision@5 `= 3/5 = 0.6` (hit)
  - `AP@5 = (1.0 + 0.6667 + 0.6) / 3 ≈ 0.7556`

### NDCG@K

- **Normalized Discounted Cumulative Gain**: 
- **Interpretation**: gives higher credit to relevant items near the top using a logarithmic "position discount", then normalizes by the best-possible ranking for that user/query.
- **Question**: "How good is the ordering of results near the top, relative to a perfect ranking for this user?"

With binary relevance, define:

```
DCG@K = Σ_{r=1..K} (rel_r / log2(r + 1))
```

and:

```
NDCG@K = DCG@K / IDCG@K
```

where `IDCG@K` is `DCG@K` computed on the *ideal* ordering (all relevant items first).

- Worked example (`K=5`), relevances by rank: `[1, 0, 1, 0, 1]` with `P=3`.
  - `DCG@5 = 1/log2(2) + 1/log2(4) + 1/log2(6)`
  `= 1 + 0.5 + 0.3869 ≈ 1.8869`
  - Ideal ranking puts the 3 relevant items at ranks 1,2,3:
  `IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4)`
  `= 1 + 0.6309 + 0.5 ≈ 2.1309`
  - `NDCG@5 ≈ 1.8869 / 2.1309 ≈ 0.8856`

### MRR (mean reciprocal rank)

- **Definition**: for one query: `1 / rank_of_first_relevant`, or `0` if none found.
- **Interpretation**: emphasizes getting at least one strong hit very early.
- **Question**: "How early is the *first* relevant game?"

## Aggregation across users/queries

Use the same semantics as `recs_004`:

- `Hit@K`: arithmetic mean (`mean`).
- `Recall@K`, `MAP@K`, `NDCG@K`, `MRR`: `nanmean`.
  - Users/queries with no positives produce `NaN` for these metrics and are excluded from those averages.

## Practical interpretation

- If `Hit@K` improves but `MAP/NDCG` do not, we are finding at least one relevant item but ranking quality is not improving much.
- If `Recall@K` improves with stable precision, we are covering more relevant items without adding too much noise.
- If `MRR` improves, first relevant hit appears earlier (better user-perceived quality for top results).

