"""Data loading for Steam reviews (raw and cleaned streams)."""

from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import pandas as pd

from steam_review_ml.data.preprocess import (
    filter_reviews,
    select_features,
    feature_engineering,
)


def load_raw_reviews(
    path: Union[str, Path],
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

    return pd.read_csv(csv_path, nrows=nrows, usecols=usecols, **read_csv_kwargs)


def _dedupe_keep_first(
    df: pd.DataFrame,
    seen_ids: set,
    id_col: str = "review_id",
) -> pd.DataFrame:
    """
    Keep rows whose id_col is not in seen_ids; update seen_ids in place.
    Returns the deduped DataFrame (first occurrence per id kept).
    """
    mask = ~df[id_col].isin(seen_ids)
    out = df[mask]
    seen_ids.update(out[id_col].tolist())
    return out


def iter_cleaned_chunks(
    input_path: Union[str, Path],
    chunksize: int,
    language: str = "english",
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream raw CSV in chunks, apply filter → dedupe → select_features, yield each featured DataFrame.

    Pipeline only; no file output. Use write_cleaned_to_parquet (export) or export_cleaned_reviews to write.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    read_kwargs: dict = {"chunksize": chunksize}
    if columns is not None:
        read_kwargs["usecols"] = columns

    seen_review_ids: set = set()

    for chunk in pd.read_csv(input_path, **read_kwargs):
        filtered = filter_reviews(chunk, language=language)
        if len(filtered) == 0:
            continue

        filtered = _dedupe_keep_first(filtered, seen_review_ids)
        if len(filtered) == 0:
            continue

        featured = select_features(filtered)
        featured = feature_engineering(featured)

        # feature engineering here
        yield featured
