from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

RawPath = Union[str, Path]


def load_raw_reviews(
    path: RawPath,
    nrows: Optional[int] = None,
    usecols: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load the raw Steam reviews CSV into a DataFrame.

    Parameters
    ----------
    path:
        Path to the raw CSV file.
    nrows:
        Optional limit on number of rows to read (for quick experiments).
    usecols:
        Optional subset of columns to read.
    read_csv_kwargs:
        Extra keyword arguments forwarded to ``pd.read_csv``.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found at: {csv_path}")

    df = pd.read_csv(csv_path, nrows=nrows, usecols=usecols, **read_csv_kwargs)
    return df


def _is_empty_review_series(series: pd.Series) -> pd.Series:
    """Return a boolean mask for reviews that are null or empty after strip()."""
    # Handle non-string values defensively by casting to str where needed.
    is_null = series.isna()
    # Avoid calling str methods on all-null series.
    non_null = series[~is_null]
    stripped = non_null.astype(str).str.strip()
    empty = stripped == ""
    mask = is_null.copy()
    mask.loc[~is_null] = empty
    return mask


def filter_reviews(df: pd.DataFrame, language: str = "english") -> pd.DataFrame:
    """
    Apply row-level filtering to the raw reviews DataFrame.

    - Keep only the specified language (default: English).
    - Drop rows with missing or empty ``review`` text.
    - Drop rows with negative playtime values in any playtime column.
    """
    filtered = df.copy()

    # 1. Language filter
    if "language" in filtered.columns:
        filtered = filtered[filtered["language"] == language]

    # 2. Drop missing / empty review text
    if "review" in filtered.columns:
        empty_mask = _is_empty_review_series(filtered["review"])
        filtered = filtered[~empty_mask]

    # 3. Drop obviously invalid negative playtime values
    playtime_cols: Iterable[str] = (
        "author.playtime_forever",
        "author.playtime_last_two_weeks",
        "author.playtime_at_review",
    )
    for col in playtime_cols:
        if col in filtered.columns:
            filtered = filtered[~(filtered[col] < 0)]

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

    # Interaction & meta features
    interaction_cols: list[str] = [
        "votes_funny",
        "comment_count",
        "steam_purchase",
        "received_for_free",
        "written_during_early_access",
        "timestamp_created",
        "timestamp_updated",
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
            missing_flag_col = f"{col.replace('.', '_')}_missing"
            base[missing_flag_col] = base[col].isna()
            base[col] = base[col].fillna(0.0)

    return base

