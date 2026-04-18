# Steam Recommendations (Applied ML Project)

This project builds a production-oriented Steam game recommender from review and metadata signals.
The current focus is a retrieval-first system with history-aware blending, offline ranking evaluation,
and FastAPI serving for user-facing recommendation flows.

The goal is to show applied ML engineering practice: clear metric definitions, reproducible experiments,
and integration from modeling notebooks to API behavior.

## What this project demonstrates

- Retrieval and ranking for recommendations (content embeddings + blended signals).
- Offline evaluation using ranking metrics (`Hit@K`, `Recall@K`, `MAP@K`, `NDCG@K`, `MRR`).
- Iterative model decisions documented with experiment notebooks and decision logs.
- API integration via FastAPI for end-to-end recommendation usage.

## Current results snapshot

Fill this table with your latest best run from `recs_008` (baseline vs history blend):

| Variant | Hit@10 | Recall@10 | MAP@10 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline retrieval | TODO | TODO | TODO | TODO | TODO |
| History blend (selected) | TODO | TODO | TODO | TODO | TODO |
| Delta | TODO | TODO | TODO | TODO | TODO |

Metric definitions and aggregation semantics live in [`docs/retrieval_metrics_guide.md`](docs/retrieval_metrics_guide.md).

## System overview

High-level flow:

1. Build/prepare game and review features.
2. Retrieve candidates from embedding similarity and related signals.
3. Blend candidate scores with history-aware weighting.
4. Evaluate quality in offline ranking notebooks.
5. Serve recommendations through FastAPI endpoints.

## Data source

- Steam Reviews 2021 Kaggle dataset by `najzeko`.
- Download into `data/raw/`:

```bash
kaggle datasets download -d najzeko/steam-reviews-2021 --unzip -p ~/steam_recommendations/data/raw
```

Dataset URL: [https://www.kaggle.com/datasets/najzeko/steam-reviews-2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021)

## Reproducibility and usage

- Pipeline run order and command references: [`docs/usage_pipeline.md`](docs/usage_pipeline.md)
- Retrieval decision log (current default + rationale): [`docs/retrieval_decision_log.md`](docs/retrieval_decision_log.md)
- Transition plan for recommender v1 -> v2: [`docs/recommender_transition_plan.md`](docs/recommender_transition_plan.md)

Quick regression check after running `recs_006`:

```bash
python scripts/check_recs_006_regression.py
```

## Product scope notes

- Core product focus is recommendations plus preference extraction.
- Review coaching is optional and intentionally separate from extraction logic.
- Additional supervised tasks (sentiment/helpfulness) are supporting components for analysis and future reranking.

See: [`docs/product_vision_recommender_and_review_coaching.md`](docs/product_vision_recommender_and_review_coaching.md)

## Limitations and next steps

- Offline ranking gains need online validation (A/B testing plan not yet implemented).
- Cold-start and popularity-bias handling can be improved with stronger priors and constraints.
- Add tighter test coverage for retrieval blending and edge-case behavior.
- Add request-level serving observability (latency, fallback path usage, score diagnostics).
