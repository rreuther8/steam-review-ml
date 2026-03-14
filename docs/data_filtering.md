# Steam Reviews Data Filtering and Feature Selection

This document specifies how we turn the raw Steam reviews CSV into a clean,
model-ready table for:

- **Sentiment prediction**: `recommended` (True/False).
- **Helpfulness prediction**: `votes_helpful` (regression) and/or
  `is_helpful` (binary label).

The goal is that all notebooks, scripts, and services use the **same rules**
implemented in `src/steam_review_ml/data/preprocess.py`.

---

## 1. Raw data snapshot (from EDA sample)

From exploratory analysis in `notebooks/eda/eda_001.ipynb` on `df_eng`
(`language == "english"`), the main columns are:

- Index / IDs
  - `Unnamed: 0` – integer row index, not semantically meaningful.
  - `review_id` – per-review ID.
  - `app_id`, `app_name` – game ID and name.
  - `author.steamid` – user ID.
- Text / labels
  - `review` – free-text review (about 0.2% missing).
  - `recommended` – boolean, target for sentiment model.
- Votes / meta
  - `votes_helpful` – integer, extremely skewed (most 0, rare large values).
  - `votes_funny` – integer.
  - `weighted_vote_score` – float.
  - `comment_count` – integer.
  - `steam_purchase`, `received_for_free`, `written_during_early_access` – bool.
- Timestamps / playtime / user stats
  - `timestamp_created`, `timestamp_updated` – Unix timestamps.
  - `author.last_played` – Unix timestamp.
  - `author.num_games_owned`, `author.num_reviews` – integers.
  - `author.playtime_forever`, `author.playtime_last_two_weeks`,
    `author.playtime_at_review` – floats; playtime is long-tailed with large outliers.

We also know the full raw CSV (outside the sample) contains multiple languages.

---

## 2. Row-level filtering rules

These rules define which **rows we keep** in the cleaned table.
They are implemented in `filter_reviews(df, ...)`.

### 2.1 Language

- **Keep** only rows with `language == "english"` for the main modeling dataset.
- Other languages can be handled in a separate, future multilingual pipeline.

### 2.2 Missing, empty, or very short review text

- For text-based models:
  - **Drop** rows where `review` is null, empty after stripping whitespace, or has fewer than 4 characters after strip.
    - Condition: `review.isna()`, or `review.str.strip() == ""`, or `len(review.str.strip()) < 4`.
- For non-text experiments, this rule could be relaxed, but the default pipeline drops them.

### 2.3 Numeric sanity checks (playtime)

Playtime columns are:

- `author.playtime_forever`
- `author.playtime_last_two_weeks`
- `author.playtime_at_review`

Rules:

- **Drop** rows where any of these is negative (should not occur, but guarded).
- Do **not** drop large playtime values by default.
  - Instead, we handle long tails via optional transforms (log/capping) at the
    feature stage, not by row exclusion.

### 2.4 Votes / helpfulness outliers

Columns:

- `votes_helpful`
- `votes_funny`

Rules:

- **Drop** rows where `votes_funny` or `votes_helpful` equals **4294967295**
  (2³² − 1, the maximum unsigned 32-bit integer). This value is a sentinel or
  overflow in the source data, not a real count; EDA found a small number of
  such records (e.g. 14 for `votes_funny`). Implement as a row filter in
  `filter_reviews` before other vote logic.
- Do **not** drop rows based on other extreme vote counts.
- We optionally **cap** these features during feature engineering (e.g., clip
  at a high percentile), not during row filtering.

### 2.5 Duplicate `review_id`

EDA (`notebooks/eda/eda_007_categorical_counts.ipynb`) shows that the raw CSV
contains **duplicate `review_id` values** — the same review can appear more than
once. For modeling and for train/val/test splits we must ensure each `review_id`
appears at most once.

- **Rule:** Drop rows that repeat a `review_id` already seen. **Keep first
  occurrence, drop later ones** when processing in row order.
- **Streaming:** We cannot load the full dataset into memory. When reading the
  CSV in chunks, deduplication must maintain a **set of `review_id`s seen so
  far** across chunks: for each row, drop the row if its `review_id` is already
  in the set; otherwise add the id to the set and keep the row.
- **Order:** Apply this deduplication **before** the train/val/test split (and
  before other filtering that depends on row identity).
- **Scale:** At current size (~21.6M unique `review_id`s), a set of IDs uses
  on the order of ~1 GB RAM, which is acceptable. If the dataset grows to
  hundreds of millions of unique reviews, the pipeline should switch to a
  database-backed dedup (e.g. DuckDB/SQLite with unique constraint on
  `review_id`) or another bounded-memory strategy.

### 2.6 Optional spam/bot heuristics (placeholder)

We currently **do not** remove suspected spam/bot reviews automatically.
Future possible heuristics (to be added only if clearly justified by EDA):

- Extremely short reviews (e.g., 1–2 characters) with:
  - Very high playtime, or
  - Very high `votes_helpful` or `votes_funny`.
- Users with extremely high `author.num_reviews` and clearly repetitive text.

---

## 3. Column selection and derived features

These rules define which columns we **keep** as features/targets in the
canonical modeling view. Implemented in `select_features(df, ...)`.

**No leakage:** We keep only fields that exist **at the time the review is
written**. Any quantity that is only known after the review is published (e.g.
votes, comments, edits) must not be used as an input feature, or it would leak
future information. Such columns are excluded from the feature set; they may
still appear in the raw data and be used as **targets** (e.g. `votes_helpful`).

### 3.1 Targets

- `recommended` (bool) – sentiment label.
- `votes_helpful` (int) – regression target for helpfulness.
- `is_helpful` (bool, **derived**):
  - Defined as `votes_helpful >= 1`.

### 3.2 Text feature

- `review` – raw text.
- `review_length_chars` (int, **derived**):
  - `len(review)` after stripping.

### 3.3 User features

Kept:

- `author.num_games_owned`
- `author.num_reviews`
- `author.playtime_last_two_weeks`
- `author.playtime_at_review`

We **intentionally exclude** `author.playtime_forever` from the feature
set because it is a *future-looking* quantity at the time the review is
written and can leak information about long-term engagement into our
prediction of the review's sentiment and helpfulness.

Derived / helper columns:

- `playtime_last_two_weeks_missing`, `playtime_at_review_missing` (bool):
  - Indicator flags if any of the above playtime columns were missing before
    simple imputation (see 3.6).

We **keep** `author.steamid` as an identifier for grouping / user-level
analysis, but it is not used as a direct numeric feature.

### 3.4 Game features

Kept:

- `app_id` – primary game identifier.
- `app_name` – human-readable label (mainly for display; not necessarily
  used as a model feature unless encoded).

### 3.5 Interaction and meta features

Kept (all known at review-writing time):

- `votes_helpful` (kept as **target** only; not used as input feature when
  predicting that review’s helpfulness)
- `steam_purchase`
- `received_for_free`
- `written_during_early_access`
- `timestamp_created`
- `author.last_played`

Excluded (post-review / would leak):

- `votes_funny` – vote counts exist only after the review is published.
- `comment_count` – comments are added after the review exists.
- `timestamp_updated` – reflects edits after creation; at write time only
  `timestamp_created` is available.

Derived (optional, initial scaffolding):

- `review_age_days` (float, **optional**, placeholder – can be added once
  a reference “current time” is decided).

### 3.6 Columns to drop

Dropped in the modeling view:

- `Unnamed: 0` – index-only.
- Any other columns not explicitly listed above.

### 3.7 Imputation / basic transforms

Applied in `select_features(df, ...)`:

- **Playtime columns**
  - Before imputation, create missing indicators:
    - `*_missing = df[col].isna()`.
  - Then fill NaNs with `0.0`.
- **Target columns**
  - `votes_helpful` is a target; if NaN appears, fill with `0`.
- **Timestamps**
  - Keep raw Unix timestamps (`timestamp_created`, `author.last_played`) as-is
    for now. (`timestamp_updated` is not kept as a feature.)
  - Derived calendar/time-of-day features can be added later.

---

## 4. Normalization and feature transforms

These transforms are applied **after** filtering and column selection, typically
at training time (e.g. in a sklearn `Pipeline` or a dedicated transform step).
They reduce skew and scale so models behave better; we do **not** drop rows here.

### 4.1 Long-tailed counts (votes, playtime)

- **Target** (`votes_helpful`):
  - Optionally **cap** at a high percentile (e.g. 99th) or a fixed ceiling
    before modeling, to avoid a few extreme values dominating.
  - Alternatively use **log(1 + x)** so the target is less skewed for
    regression on helpfulness.
- **Playtime** (`author.playtime_last_two_weeks`, `author.playtime_at_review`):
  - Same idea: optional **log(1 + x)** or **cap** at a high percentile.
  - Apply after the 0-imputation; the missing-indicator columns still record
    which rows were imputed.
- **Owner Behavior** (`author.num_games_owned`, `author.num_reviews`)"
  - Same idea: optional **log(1 + x)** or **cap** at a high percentile.
  - Apply after the 0-imputation; the missing-indicator columns still record
    which rows were imputed.

### 4.2 Numeric scaling

- For linear models or distance-based methods, **standardize** numeric
  features (zero mean, unit variance) using statistics from the **training**
  set only, then apply the same transform to validation/test.
- Tree-based models (e.g. Random Forest, XGBoost) do not require scaling;
  optional caps or log transforms still help with long tails.

### 4.3 Where this lives

- Implement in the **training pipeline** (e.g. `ColumnTransformer` with
  `StandardScaler`, or custom transform that caps/logs), not inside
  `filter_reviews` or `select_features`.
- Document which columns are log-transformed or capped and at what values,
  so evaluation and inference use the same transform.

---

## 5. Implementation mapping

The filtering and selection logic is implemented in:

- `src/steam_review_ml/data/preprocess.py`

Key functions:

- `load_raw_reviews(path: Path | str, nrows: int | None = None, usecols=None) -> DataFrame`
  - Thin wrapper around `pd.read_csv` for the raw CSV, with optional column
    projection and row limit for experimentation.
- `filter_reviews(df: DataFrame) -> DataFrame`
  - Applies the row-level rules in section 2:
    - Keep English only.
    - Drop empty or null `review`.
    - Drop rows with negative playtime values.
    - Drop rows where `votes_helpful` or `votes_funny` equals 4294967295 (sentinel).
- `select_features(df: DataFrame) -> DataFrame`
  - Applies the column selection and derived feature rules in section 3:
    - Adds `is_helpful`.
    - Adds `review_length_chars`.
    - Adds playtime missing indicators and fills NaNs with 0.
    - Drops `Unnamed: 0` and any columns not in the defined feature set
      plus identifiers/targets.

Example pipeline (pseudo-code):

```python
from pathlib import Path
from steam_review_ml.data.preprocess import load_raw_reviews, filter_reviews, select_features

RAW_PATH = Path(\"data/steam_reviews_full.csv\")

df_raw = load_raw_reviews(RAW_PATH)
df_clean = filter_reviews(df_raw)
df_features = select_features(df_clean)
```

Notebooks and scripts should call this shared pipeline instead of duplicating
filtering logic.

---

## 6. Validation and iteration

To validate the rules and guard against regressions:

- After any change to `filter_reviews` or `select_features`:
  - Compare **row counts** before vs. after filtering for a fixed raw sample.
  - Check **class balance** for:
    - `recommended` (positive vs. negative).
    - `is_helpful` (True vs. False).
  - Inspect distributions of:
    - `author.playtime_*` before/after imputation and with missing flags.
    - `votes_helpful`, `votes_funny` (pay attention to caps/long tails).
  - Confirm that the proportion of dropped rows is reasonable and that we
    are not discarding the majority of data by mistake.

These checks are implemented as:

- **Unit tests** in `tests/test_preprocess.py`: row counts after filtering,
  no sentinel vote values in output, derived columns present, no negative
  playtime.
- **Notebook / manual checks** (e.g. in a data-filtering or validation
  notebook): class balance for `recommended` and `is_helpful`, distributions
  of playtime and votes before/after imputation, proportion of rows dropped.

## 7. Producing the cleaned dataset

stream -> filter -> dedupe -> select -> write

