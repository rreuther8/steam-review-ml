from __future__ import annotations

import logging
import hashlib
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _is_empty_or_short_review_series(
    series: pd.Series, min_length: int = 4
) -> pd.Series:
    """Return a boolean mask True for rows to drop: null, empty, or length < min_length after strip()."""
    is_null = series.isna()
    non_null = series[~is_null]
    stripped = non_null.astype(str).str.strip()
    empty_or_short = (stripped == "") | (stripped.str.len() < min_length)
    mask = is_null.copy()
    mask.loc[~is_null] = empty_or_short
    return mask


def filter_reviews(df: pd.DataFrame, language: str = "english") -> pd.DataFrame:
    """
    Apply row-level filtering to the raw reviews DataFrame.

    - Keep only the specified language (default: English).
    - Drop rows with missing, empty, or very short (under 4 characters after strip) ``review`` text.
    - Drop rows with negative playtime values in any playtime column.
    - Drop rows where votes_helpful or votes_funny equals 4294967295 (sentinel).
    """
    n_start = len(df)
    logger.debug("filter_reviews: starting with %d rows (language=%s)", n_start, language)
    filtered = df.copy()

    def _drop_vote_sentinel_rows(filtered: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where votes_helpful or votes_funny are sentinel value (4294967295)."""
        VOTE_SENTINEL = 4294967295  # 2^32 - 1
        for col in ("votes_helpful", "votes_funny"):
            if col in filtered.columns:
                before = len(filtered)
                filtered = filtered[filtered[col] < VOTE_SENTINEL]
                dropped = before - len(filtered)
                if dropped:
                    logger.debug("  drop vote sentinel (%s): %d -> %d (-%d)", col, before, len(filtered), dropped)
        return filtered

    def _filter_by_language(filtered: pd.DataFrame, language: str) -> pd.DataFrame:
        """Keep only rows with the specified language."""
        if "language" in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered["language"] == language]
            logger.debug("  filter by language (%s): %d -> %d (-%d)", language, before, len(filtered), before - len(filtered))
        return filtered

    def _drop_empty_or_short_reviews(filtered: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing, empty, or very short review text."""
        if "review" in filtered.columns:
            before = len(filtered)
            drop_mask = _is_empty_or_short_review_series(
                filtered["review"], min_length=4
            )
            filtered = filtered[~drop_mask]
            logger.debug("  drop empty/short review: %d -> %d (-%d)", before, len(filtered), before - len(filtered))
        return filtered

    def _drop_negative_playtime_rows(filtered: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with negative playtime values in any playtime column."""
        playtime_cols: Iterable[str] = (
            "author.playtime_forever",
            "author.playtime_last_two_weeks",
            "author.playtime_at_review",
        )
        for col in playtime_cols:
            if col in filtered.columns:
                before = len(filtered)
                filtered = filtered[~(filtered[col] < 0)]
                dropped = before - len(filtered)
                if dropped:
                    logger.debug("  drop negative playtime (%s): %d -> %d (-%d)", col, before, len(filtered), dropped)
        return filtered

    # Now apply the pipeline of internal steps
    filtered = _drop_vote_sentinel_rows(filtered)
    filtered = _filter_by_language(filtered, language)
    filtered = _drop_empty_or_short_reviews(filtered)
    filtered = _drop_negative_playtime_rows(filtered)

    logger.info("filter_reviews: %d -> %d rows (kept %.1f%%)", n_start, len(filtered), 100.0 * len(filtered) / n_start if n_start else 0)
    return filtered


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project the cleaned reviews DataFrame to a modeling-ready feature set.

    This function:
    - Keeps a curated set of identifier, feature, and target columns.
    - Adds ``is_helpful`` from ``votes_helpful``.
    - Adds ``review_length_chars`` from ``review``.
    - Adds playtime missing-indicator columns and fills playtime NaNs with 0.0.
    """
    logger.debug("select_features: %d rows, selecting and deriving columns", len(df))
    base = df.copy()

    # Targets
    target_cols = ["recommended", "votes_helpful"]

    # Identifiers / metadata
    id_cols = ["review_id", "app_id", "app_name", "author.steamid"]

    # Core text feature
    text_cols = ["review"]

    # User features
    user_cols = [
        "author.num_games_owned",
        "author.num_reviews",
        "author.playtime_last_two_weeks",
        "author.playtime_at_review",
    ]

    # Interaction & meta features (only those known at review-writing time; no
    # post-review signals like comment_count, votes_funny, timestamp_updated)
    interaction_cols: list[str] = [
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
        "timestamp_created",
        "author.last_played",
    ]

    # Build list of columns we intend to keep if present.
    desired_cols: list[str] = []
    for group in (id_cols, text_cols, target_cols, user_cols, interaction_cols):
        for col in group:
            if col in base.columns and col not in desired_cols:
                desired_cols.append(col)

    # Drop obvious index-like column if present.
    drop_cols = []
    if "Unnamed: 0" in base.columns:
        drop_cols.append("Unnamed: 0")
    if drop_cols:
        base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    # Start from the reduced set.
    base = base[desired_cols].copy()

    # Derived: is_helpful from votes_helpful
    if "votes_helpful" in base.columns:
        base["is_helpful"] = base["votes_helpful"] >= 1

    # Derived: review_length_chars
    if "review" in base.columns:
        # Use empty string for missing reviews to avoid errors; most have been
        # filtered out already.
        review_filled = base["review"].fillna("")
        base["review_length_chars"] = review_filled.astype(str).str.len()

    # Playtime missing indicators and simple imputation
    playtime_cols = ["author.playtime_last_two_weeks", "author.playtime_at_review"]
    for col in playtime_cols:
        if col in base.columns:
            # missing_flag_col = f"{col.replace('.', '_')}_missing"
            # base[missing_flag_col] = base[col].isna()
            base[col] = base[col].fillna(0.0)

    logger.info("select_features: %d rows, %d columns", len(base), len(base.columns))
    return base


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the DataFrame.

    This function:
    - Adds ``review_word_count`` from ``review``.
    - Adds ``review_length_chars`` from ``review``.
    """
    logger.debug("feature_engineering: adding review_word_count, review_length_chars for %d rows", len(df))
    df["review_word_count"] = (
        df["review"].fillna("").apply(lambda x: len(str(x).split()))
    )
    df["review_length_chars"] = df["review"].fillna("").astype(str).str.len()
    logger.info("feature_engineering: done (%d rows)", len(df))
    return df


def add_stratify_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column that combines the three stratification keys.
    This function:
    - Adds ``stratify_group`` from ``app_name``, ``recommended``, and ``is_helpful``.
    This is used to stratify the data into train, validation, and test sets.
    """
    logger.debug("add_stratify_group: adding stratify_group for %d rows", len(df))
    out = df.copy()
    out["stratify_group"] = (
        out["app_name"].astype(str)
        + "|"
        + out["recommended"].astype(str)
        + "|"
        + out["is_helpful"].astype(str)
    )
    return out


def stable_split_u(*, seed: int, stratify_group: str, review_id: int) -> float:
    """
    Deterministic pseudo-random number in [0, 1) based on (seed, stratify_group, review_id).
    Stable across runs and machines.
    """
    s = f"{seed}|{stratify_group}|{int(review_id)}".encode("utf-8")
    digest = hashlib.md5(s).digest()
    x = int.from_bytes(digest[:4], "little", signed=False)  # 32-bit
    return x / 2**32
