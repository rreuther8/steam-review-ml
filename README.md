Working on Steam Recommendation system with FASTAPI frontend.

## Data source

- **Steam reviews 2021** Kaggle dataset by `najzeko`.  
  You can download and unzip the raw data into `data/raw/` with:

  ```bash
  kaggle datasets download -d najzeko/steam-reviews-2021 --unzip -p ~/steam_recommendations/data/raw
  ```
  Website: 
  - https://www.kaggle.com/datasets/najzeko/steam-reviews-2021[https://www.kaggle.com/datasets/najzeko/steam-reviews-2021]

## Project goals

- **Content-led recommendations (v1)**: after a user writes a review, suggest other games using **text ↔ game-profile** similarity (retrieval-first). **Hybrid ranking (v2)** adds blended signals (e.g. ALS, popularity, metadata); see [`docs/recommender_transition_plan.md`](docs/recommender_transition_plan.md).
- **Product vision (recs + coaching)**: broader UX and how review feedback fits alongside recommendations — [`docs/product_vision_recommender_and_review_coaching.md`](docs/product_vision_recommender_and_review_coaching.md).
- **Predict review sentiment (`recommended`)**: build models that use both review text and metadata (playtime, user stats, purchase flags, etc.) to predict whether a review will be positive or negative.
- **Predict review helpfulness (`votes_helpful`)**: using the same feature set, predict which reviews are likely to receive helpful votes, framed either as a regression task (number of helpful votes) or a classification task (helpful vs. not).
- **End-to-end user experience**: expose these capabilities via a GUI (FastAPI): e.g. draft review → suggestions + sentiment/helpfulness (and later, structured review coaching per the product vision doc).

Supervised tasks support analysis and **v2 hybrid reranking**; the recommender **v1 thesis** is content retrieval (transition plan above).

## Usage

- Pipeline run order and commands: `docs/usage_pipeline.md`
