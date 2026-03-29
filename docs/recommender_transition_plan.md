# Recommender Transition Plan

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

1. **Candidate generation**
   - Produce a shortlist of possible games.
   - Fast MVP: content-based retrieval from review-text similarity.

2. **Ranking/scoring**
   - Re-rank candidates using blended signals.
   - Combine text similarity with predicted `recommended` / helpfulness signals.

## Phase 1 MVP (content-based recommender)

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

This creates a true recommender quickly.

## Phase 2 (hybrid ranking with current models)

Use current model outputs as ranking features for candidate games.

Example blended score:

`rank_score = 0.6 * text_similarity + 0.3 * p_recommended + 0.1 * p_is_helpful`

Possible ranking features:

- text similarity
- `p(recommended)`
- `p(is_helpful)` or expected `votes_helpful`
- popularity/volume priors

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

1. Build game profiles from training reviews.
2. Implement similarity retrieval and top-10 recommendation output.
3. Add a simple API endpoint for recommendations.
4. Add hybrid ranking using current model outputs.
5. Evaluate with @K metrics against similarity-only baseline.
6. Iterate on weights/features and document findings.

## Pseudocode sketch (MVP retrieval)

```python
# user_text -> vectorize -> cosine similarity with game profile vectors -> top_k

# user_vec = vectorizer.transform([user_text])
# sims = cosine_similarity(user_vec, game_profile_matrix).ravel()
# top_idx = np.argsort(-sims)[:10]
# recommendations = game_profiles.iloc[top_idx][["app_id", "app_name"]]
```

