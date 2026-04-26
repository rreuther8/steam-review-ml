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

There is **no single headline offline number** that summarizes the whole project: retrieval quality is demonstrated through **two complementary notebooks** that answer different questions.

### Offline evaluation suites

| Notebook | What it answers |
| --- | --- |
| [`recs_004_eval_proxy_same_user.ipynb`](notebooks/models/query_embeddings/recs_004_eval_proxy_same_user.ipynb) | Under the same-user proxy task, **which query embedding recipe** should win—raw vs structured preference summaries, pooled train history, time-weighted windows, etc.—against held-out likes, with popularity and random baselines. |
| [`recs_006_eval_ablation_4way.ipynb`](notebooks/models/query_embeddings/recs_006_eval_ablation_4way.ipynb) | **Where representation matters**: a **4-way** ablation of raw vs structured text on **both** the query side and the game-index side. In our runs, **raw query + raw game embeddings** was the strongest combination; structured text is still measured explicitly so we know whether gains come from rewriting the query, the catalog vectors, both, or neither. |

#### recs_004 — query-method sweep (same-user proxy)

Figures below are **K = 10** from the checked-in §3 aggregate table (**validation** queries, **n_examples = 5000**, indexed catalog **n_games = 315**). Definitions match [`docs/retrieval_metrics_guide.md`](docs/retrieval_metrics_guide.md).

| Method | Hit@10 | Recall@10 | MAP@10 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random` | 0.0452 | 0.0317 | 0.0087 | 0.0152 | 0.0248 |
| `popularity_train` | 0.2232 | 0.1925 | 0.0679 | 0.1004 | 0.1019 |
| `raw` | 0.0826 | 0.0679 | 0.0259 | 0.0373 | 0.0460 |
| `structured` | 0.0496 | 0.0385 | 0.0138 | 0.0207 | 0.0295 |
| `multi_mean_train` | 0.0922 | 0.0752 | 0.0265 | 0.0395 | 0.0469 |
| `multi_concat_train` | 0.0850 | 0.0696 | 0.0242 | 0.0363 | 0.0433 |
| `tw_train_mean_30d` | 0.0858 | 0.0710 | 0.0266 | 0.0385 | 0.0467 |
| `tw_train_mean_365d` | 0.0858 | 0.0710 | 0.0266 | 0.0385 | 0.0467 |

#### recs_006 — four-way query × index representation

All values **K = 10** from [`artifacts/recs/eval_review_style_4way_proxy_metrics.csv`](artifacts/recs/eval_review_style_4way_proxy_metrics.csv) (same aggregation as the notebook proxy summary). Arms are **query encoding × game-index encoding** (`raw_*` = raw review text; `structured_*` = structured preference rewrite / structured game profile text).

| Arm | Hit@10 | Recall@10 | MAP@10 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw query · raw index (`raw_raw`) | 0.0824 | 0.0686 | 0.0237 | 0.0355 | 0.0320 |
| Structured query · raw index (`structured_raw`) | 0.0482 | 0.0383 | 0.0122 | 0.0192 | 0.0180 |
| Raw query · structured index (`raw_structured`) | 0.0388 | 0.0318 | 0.0107 | 0.0162 | 0.0155 |
| Structured query · structured index (`structured_structured`) | 0.0662 | 0.0541 | 0.0178 | 0.0274 | 0.0262 |

The frozen regression baseline for the **raw_raw** arm is also stored in [`artifacts/recs/eval_review_style_4way_proxy_baseline_raw_raw.json`](artifacts/recs/eval_review_style_4way_proxy_baseline_raw_raw.json) (`as_of_date`: 2026-04-14).

Artifacts for these flows land under [`artifacts/recs/`](artifacts/recs/) where notebooks write CSV/JSONL outputs. After regenerating metrics from recs_006, you can run [`scripts/check_recs_006_regression.py`](scripts/check_recs_006_regression.py) against saved baselines.

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

- **Random seed:** the project-wide default is [`PROJECT_RANDOM_SEED`](src/steam_review_ml/constants.py) in `steam_review_ml.constants` (used for recommender eval subsampling, tabular `random_state`, and synthetic baseline RNG streams). Train/val/test splitting reads the same value from that module unless you override with **`STEAM_REVIEWS_RANDOM_STATE`** (see [`docs/usage_pipeline.md`](docs/usage_pipeline.md)).
- **Split policy:** current pipeline config uses a hybrid mode in [`configs/split_reviews.json`](configs/split_reviews.json): sparse users (`<3` interactions) use deterministic random 70/15/15, while denser users (`>=3`) use per-user temporal last-N assignment. Report key retrieval metrics by cohort (`<3`, `>=3`, overall weighted) to avoid hiding cold-start behavior.
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

### Failure mode: multi-interest history and pooled query vectors

Retrieval notebooks that **pool past reviews into one query vector** (e.g. `multi_mean_train`, time-weighted `tw_train_mean_*` in [`recs_004`](notebooks/models/query_embeddings/recs_004_eval_proxy_same_user.ipynb)) assume something like a **single direction in embedding space**. That breaks down when a user’s train history mixes **distinct modes of taste**—for example thumbs-up reviews for both a **fast action** title and a **cozy farming** sim. A **mean or blended embedding** can sit **between** those clusters, so recommendations may look directionally vague even though both likes are genuine. Strong **same-user proxy** scores do not fully rule this out: the metric can still reward **any** surfaced positive while **coherence** across heterogeneous history stays weak. Reasonable mitigations for a future iteration include **multi-interest** representations (multiple prototypes, clustering, or max-over-clusters retrieval), **session- or recency-first** pooling instead of lifetime aggregation, or **metadata-aware** separation when tags are reliable.

- Offline ranking gains need online validation (A/B testing plan not yet implemented).
- Cold-start and popularity-bias handling can be improved with stronger priors and constraints.
- Add tighter test coverage for retrieval blending and edge-case behavior.
- Add request-level serving observability (latency, fallback path usage, score diagnostics).
