"""Data loading for Steam reviews (raw and cleaned streams)."""

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from steam_review_ml.data.preprocess import (
    filter_reviews,
    select_features,
    feature_engineering,
    convert_to_int,
    add_stratify_group,
    stable_split_u,
)

logger = logging.getLogger(__name__)


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


def iter_clean_chunks(
    input_path: Union[str, Path],
    chunksize: int,
    language: str = "english",
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream raw CSV in chunks, apply filter → dedupe → select_features, yield each featured DataFrame.

    Pipeline only; no file output. Use write_parquet_chunked (export) or export_cleaned_reviews to write.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("iter_clean_chunks: input=%s chunksize=%s language=%s", input_path, chunksize, language)

    read_kwargs: dict = {"chunksize": chunksize}
    if columns is not None:
        read_kwargs["usecols"] = columns
        logger.debug("  usecols=%s", columns)

    seen_review_ids: set = set()
    chunk_iter = pd.read_csv(input_path, **read_kwargs)

    for chunk in tqdm(chunk_iter, desc="Reading & cleaning chunks", unit="chunk"):
        filtered = filter_reviews(chunk, language=language)
        if len(filtered) == 0:
            continue

        filtered = _dedupe_keep_first(filtered, seen_review_ids)
        if len(filtered) == 0:
            continue

        featured = select_features(filtered)
        featured = feature_engineering(featured)
        bool_cols = [
            "recommended",
            "is_helpful",
            "steam_purchase",
            "received_for_free",
            "written_during_early_access",
        ]
        featured = convert_to_int(
            featured, [col for col in bool_cols if col in featured.columns]
        )
        yield featured

def iter_split_chunks(
    input_path: Union[str, Path],
    chunksize: int,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Stream cleaned reviews in chunks, split into train, validation, and test sets.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")

    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be >=0 and sum to < 1")

    train_size = 1.0 - val_size - test_size
    cut_train = train_size
    cut_val = train_size + val_size

    # pandas read_parquet does not support 'chunksize'; use pyarrow to iterate Parquet row groups
    pf = pq.ParquetFile(input_path)

    logger.info(
        "iter_split_chunks: input=%s batch_size=%s train/val/test=%.3f/%.3f/%.3f seed=%s",
        input_path,
        chunksize,
        train_size,
        val_size,
        test_size,
        random_state,
    )

    for batch_idx, chunk in enumerate(
        tqdm(
            pf.iter_batches(batch_size=chunksize),
            desc="Reading & splitting chunks",
            unit="batch",
        ),
        start=1,
    ):
        df = chunk.to_pandas()
        if len(df) == 0:
            continue

        df = add_stratify_group(df)

        # stable per-row uniform in [0,1)
        # IMPORTANT: do NOT use Python's built-in hash(); use hashlib for stability.
        u = df.apply(
            lambda r: stable_split_u(
                seed=random_state,
                stratify_group=r["stratify_group"],
                review_id=r["review_id"],
            ),
            axis=1,
        )
        if "stratify_group" in df.columns:
            df = df.drop(columns=["stratify_group"])

        train_df = df[u < cut_train]
        val_df = df[(u >= cut_train) & (u < cut_val)]
        test_df = df[u >= cut_val]
        logger.debug(
            "split batch %d: raw=%d -> train=%d val=%d test=%d",
            batch_idx,
            len(df),
            len(train_df),
            len(val_df),
            len(test_df),
        )
        yield train_df, val_df, test_df
