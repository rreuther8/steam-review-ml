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

- **Predict review sentiment (`recommended`)**: build models that use both review text and metadata (playtime, user stats, purchase flags, etc.) to predict whether a review will be positive or negative.
- **Predict review helpfulness (`votes_helpful`)**: using the same feature set, predict which reviews are likely to receive helpful votes, framed either as a regression task (number of helpful votes) or a classification task (helpful vs. not).

- **End-to-end user experience**: expose these models via a GUI where a user can type a review (and optionally provide simple metadata such as playtime), then see predicted sentiment and helpfulness in real time.

These supervised tasks and the GUI front end form the core ML workflows for the project and provide a base for later recommendation and ranking models.
