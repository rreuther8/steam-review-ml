# EDA Plan: Build Understanding for Recommended and votes_helpful

This plan walks through exploratory analysis in a fixed order so each step supports the next and you can explain your choices (e.g. in an interview) instead of vibe-coding. Work in `notebooks/eda/eda_001.ipynb`; reuse helpers there and add new ones as needed.

**Targets:** (1) `recommended` (sentiment), (2) `votes_helpful` / `is_helpful` (helpfulness).  
**Data:** English-only slice (e.g. `df_eng`). Filtering and column choices are in [data_filtering.md](data_filtering.md).

---

## Suggested order

1. **Target variables** – balance and distributions so you can choose metrics and framing.
2. **Missingness and duplicates** – know what you’re dropping or imputing.
3. **Review length** – distribution and relationship to both targets.
4. **Metadata vs targets** – which features separate recommended / helpful.
5. **Correlations and outliers** – redundancy and capping/trimming.

---

## 1. Target variables

**Why first:** Decides evaluation metrics, class weighting, and whether to treat helpfulness as regression vs binary.

### 1.1 Class balance for `recommended`

- **Do:** Counts or bar chart of recommended vs not (e.g. `show_class_balance(df_eng, col='recommended')`).
- **Learn:** If heavily imbalanced, accuracy is misleading; use precision/recall/F1 and consider class weights or resampling.

### 1.2 Distribution of `votes_helpful`

- **Do:** Same idea as `votes_funny`: most mass near zero, long tail. Use `show_votes_helpful_distribution` (or equivalent) with log scale; consider binning (e.g. 0, 1–5, 6+) and a binary `is_helpful = (votes_helpful >= 1)` for a simpler model.
- **Learn:** Drives choice of regression vs classification and whether to log-transform or cap the target.

### 1.3 Relationship between targets

- **Do:** Crosstab or grouped bar of `recommended` vs `is_helpful` (or binned `votes_helpful`).
- **Learn:** How much sentiment and helpfulness overlap; whether one target is redundant or adds signal.

---

## 2. Data quality and structure

**Why second:** So filtering and missingness decisions are explicit before you interpret “features vs targets.”

### 2.1 Missingness

- **Do:** Per-column % missing (or a small missingness heatmap). Focus on: `review`, `votes_helpful`, playtime, `author.*`. Use `missing_report` or similar.
- **Learn:** Decides imputation vs dropping and documents what the model actually sees.

### 2.2 Duplicate / repeated content

- **Do:** Check duplicate `review_id`; check how often the same `review` text appears (could be bots or templates).
- **Learn:** Informs dedup strategy and whether to treat repeated text as a feature or drop.

### 2.3 Reviews per game

- **Do:** Count of reviews per `app_id` (or `app_name`). Histogram or top-N games.
- **Learn:** Whether a few games dominate; whether you need stratification or per-game modeling later.

### 2.4 Reviews per author

- **Do:** Distribution of reviews per `author.steamid`.
- **Learn:** Flags power users and possible leakage if you use author-level features.

---

## 3. Text

**Why third:** Review length and samples inform filters (e.g. min length) and model inputs (padding/truncation).

### 3.1 Length distribution

- **Do:** Histogram of review length (chars or words); log scale if skewed.
- **Learn:** Informs min length filters and padding/truncation for models.

### 3.2 Sample of positive vs negative reviews

- **Do:** A few short excerpts for recommended vs not. Sanity-check labels and wording (e.g. “not recommended” in text vs label).
- **Learn:** Validates label quality and suggests whether text alone can predict sentiment.

### 3.3 Simple text stats by target

- **Do:** Mean length (and maybe word count) by `recommended` and by `is_helpful` (or bins of `votes_helpful`).
- **Learn:** Often longer reviews are more likely helpful and/or negative; sets expectations for feature importance.

---

## 4. Features vs targets

**Why fourth:** After targets and data quality, you can interpret which features separate the two outcomes.

### 4.1 Numeric features vs `recommended`

- **Do:** Playtime (at review, last two weeks), `author.num_games_owned`, `author.num_reviews`: boxplots or violin plots by `recommended` to see separation and outliers.
- **Learn:** Which metadata might predict sentiment; whether to cap or log-transform.

### 4.2 Numeric features vs helpfulness

- **Do:** Same plots for `votes_helpful` or `is_helpful` (or binned helpfulness).
- **Learn:** Which metadata might predict helpfulness; same capping/transform decisions.

### 4.3 Review length vs both targets

- **Do:** Distribution of character (or word) count by `recommended` and by helpful/not helpful.
- **Learn:** Confirms “longer → more helpful / more negative” and supports using length as a feature.

---

## 5. Numeric feature relationships

**Why last:** Redundancy and outliers matter for feature set and preprocessing (aligned with [data_filtering.md](data_filtering.md)).

### 5.1 Correlation matrix

- **Do:** For numeric columns (playtime, votes, author counts). Use `plot_correlation_heatmap` or equivalent.
- **Learn:** Spot redundancy and multicollinearity before modeling.

### 5.2 Outliers

- **Do:** Boxplots or z-scores for playtime and vote columns. Use `detect_outliers_zscore` / `plot_outliers` or similar.
- **Learn:** Justify capping or trimming in preprocessing (document in data_filtering if you change rules).

---

## Checklist (for tracking)

- [ ] 1.1 Class balance for `recommended`
- [ ] 1.2 Distribution of `votes_helpful` (and optional `is_helpful` binning)
- [ ] 1.3 Crosstab: recommended vs is_helpful
- [ ] 2.1 Missingness report
- [ ] 2.2 Duplicate review_id and repeated text checks
- [ ] 2.3 Reviews per game
- [ ] 2.4 Reviews per author
- [ ] 3.1 Review length distribution
- [ ] 3.2 Sample positive vs negative review excerpts
- [ ] 3.3 Mean length by recommended and by is_helpful
- [ ] 4.1 Numeric features vs recommended (boxplots/violins)
- [ ] 4.2 Numeric features vs helpfulness
- [ ] 4.3 Review length vs both targets
- [ ] 5.1 Correlation matrix
- [ ] 5.2 Outliers for playtime and vote columns

---

## Notes

- **Helpers:** Reuse and extend functions in `notebooks/eda/eda_001.ipynb` (e.g. `show_class_balance`, `show_votes_helpful_distribution`, `plot_distributions`, `missing_report`, `plot_correlation_heatmap`, `detect_outliers_zscore`). Add small, well-named functions for new views so the notebook stays readable.
- **Decisions:** When you change filtering or feature choices based on EDA, update [data_filtering.md](data_filtering.md) and/or preprocessing in `src/steam_review_ml/data/preprocess.py` so the pipeline stays the single source of truth.
