# Recommender Transition Plan

Broader product goals (recommendations + review coaching) live in [`product_vision_recommender_and_review_coaching.md`](product_vision_recommender_and_review_coaching.md).

## North star (single lane)

We commit to **one ordering** so content retrieval and collaborative filtering are not competing “religions”:

| Version | Role | Collaborative / ALS |
|---------|------|------------------------|
| **v1** | **Content retrieval** answers the user’s prompt: draft review → similar games via game-profile text/vectors. | **Out of scope for v1.** |
| **v2** | **Hybrid ranking** on the **same candidate set** as v1: rerank with blended signals. | **Planned upgrade:** ALS or item–item factors are **one** optional term alongside popularity, metadata, similarity priors, and optional tabular model scores — not a parallel v1 product. |

**ALS is deferred** until after v1 retrieval and @K evaluation are in place and you have characterized user–item sparsity. Until then, the thesis is **content-led**.

## Current state vs goal

Current models predict:

- `recommended`
- `is_helpful`
- `votes_helpful`

These are useful prediction tasks, but they do not directly solve recommendation.

Current tasks answer:

- "Will this review be positive/helpful?"

Recommendation needs to answer:

- "What should this user try next?"

## Recommended architecture (2 stages)

Same pattern for v1 and v2; v2 enriches stage 2 only.

1. **Candidate generation**
   - Produce a shortlist of possible games.
   - **v1:** content-based retrieval from review-text similarity (game profiles).
   - **v2 (optional):** widen or dedupe candidates if you add multiple generators; primary v1 path can remain similarity-based top‑M.

2. **Ranking / scoring**
   - Re-rank candidates using blended signals.
   - **v1:** can be identity rank (pure similarity) or light tweaks (e.g. popularity prior).
   - **v2:** hybrid scores — e.g. text similarity + **ALS / co-occurrence** + popularity + metadata + optional `p(recommended)` / helpfulness features.

Example blended score (illustrative only; tune on validation):

`rank_score = w1 * text_similarity + w2 * als_or_item_score + w3 * popularity_prior + ...`

## Phase 1 — v1 MVP (content-based recommender)

### Inputs

- User free-text review
- Optional lightweight metadata (e.g., playtime)

### Artifacts to build

- `game_profile.parquet` (or equivalent), one row per game with:
  - `app_id`
  - `app_name`
  - profile text (aggregate of positive reviews)
  - profile vector/embedding
- `vectorizer.pkl` (or embedding model reference)

### Online flow

1. Vectorize user review.
2. Compute similarity vs game profile vectors.
3. Return top-N games by similarity.

This creates a true recommender quickly and matches the **write-a-review** moment.

## Phase 2 — v2 (hybrid ranking)

Rerank **the same candidates** (or a slightly enlarged pool) using multiple signals. **ALS or matrix factorization on implicit feedback** is an optional component here — not the v1 thesis.

Possible ranking features:

- text similarity (from v1)
- collaborative / ALS user–item or item–item score (when added)
- `p(recommended)`, `p(is_helpful)`, or expected `votes_helpful` (if you keep these models)
- popularity / volume priors
- metadata (tags, genres) if available

Tune weights on validation data.

## Evaluation shift (important)

Keep existing prediction metrics, but make recommender metrics primary:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K

Also track:

- recommendation coverage/diversity
- popularity bias
- per-game / per-segment quality

## Concrete next tasks (execution order)

**v1**

1. Build game profiles from training reviews.
2. Implement similarity retrieval and top‑N recommendation output.
3. Evaluate with @K metrics; document vs. simple baselines (e.g. popularity).

**v2 (after v1 baseline exists)**

4. Add hybrid ranking (start with priors + metadata; add ALS when data/process justify it).
5. Optionally fold in `recommended` / helpfulness model scores as features.
6. Add a simple API endpoint for recommendations if not already present.
7. Iterate on weights/features and document findings.

## Pseudocode sketch (MVP retrieval)

```python
# user_text -> vectorize -> cosine similarity with game profile vectors -> top_k

# user_vec = vectorizer.transform([user_text])
# sims = cosine_similarity(user_vec, game_profile_matrix).ravel()
# top_idx = np.argsort(-sims)[:10]
# recommendations = game_profiles.iloc[top_idx][["app_id", "app_name"]]
```
