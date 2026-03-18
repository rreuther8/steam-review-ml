"""Export cleaned Steam reviews to a single file (Parquet)."""

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_review_ml.data.loaders import iter_clean_chunks

logger = logging.getLogger(__name__)


def write_parquet_chunked(
    chunks_iter: Iterable[pd.DataFrame],
    output_path: Union[str, Path],
) -> None:
    """
    Write a single Parquet file from an iterator of DataFrames (e.g. from iter_clean_chunks).

    Schema is taken from the first chunk. Low memory: one chunk in memory at a time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("write_parquet_chunked: output=%s", output_path)

    writer = None
    total_rows = 0
    chunk_count = 0
    for featured in chunks_iter:
        table = pa.Table.from_pandas(featured, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
            logger.info("  Parquet writer opened, schema from first chunk")
        writer.write_table(table)
        total_rows += len(featured)
        chunk_count += 1
        logger.info("  wrote chunk %d (%d rows), total so far %d", chunk_count, len(featured), total_rows)

    if writer is not None:
        writer.close()
        logger.info("write_parquet_chunked: closed file, %d chunks, %d total rows", chunk_count, total_rows)


def write_split_parquets(
    split_iter: Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    train_output_path: Union[str, Path],
    val_output_path: Union[str, Path],
    test_output_path: Union[str, Path],
    log_every_n_chunks: int = 10,
) -> None:
    train_output_path = Path(train_output_path)
    val_output_path = Path(val_output_path)
    test_output_path = Path(test_output_path)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "write_split_parquets: train=%s val=%s test=%s",
        train_output_path,
        val_output_path,
        test_output_path,
    )

    writers = {"train": None, "val": None, "test": None}
    totals = {"train": 0, "val": 0, "test": 0}
    chunk_count = 0

    for train_df, val_df, test_df in split_iter:
        chunk_count += 1

        for name, df, out_path in [
            ("train", train_df, train_output_path),
            ("val", val_df, val_output_path),
            ("test", test_df, test_output_path),
        ]:
            if len(df) == 0:
                continue

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writers[name] is None:
                writers[name] = pq.ParquetWriter(out_path, table.schema)
                logger.info("  opened %s writer: %s", name, out_path)

            writers[name].write_table(table)
            totals[name] += len(df)

        logger.debug(
            "split chunk %d: train=%d val=%d test=%d (totals: %d/%d/%d)",
            chunk_count,
            len(train_df),
            len(val_df),
            len(test_df),
            totals["train"],
            totals["val"],
            totals["test"],
        )
        if log_every_n_chunks > 0 and (chunk_count % log_every_n_chunks == 0):
            logger.info(
                "split progress: chunks=%d totals: train=%d val=%d test=%d",
                chunk_count,
                totals["train"],
                totals["val"],
                totals["test"],
            )

    for name, w in writers.items():
        if w is not None:
            w.close()
            logger.info("closed %s writer (%d rows)", name, totals[name])

    logger.info(
        "write_split_parquets: done (chunks=%d totals: train=%d val=%d test=%d)",
        chunk_count,
        totals["train"],
        totals["val"],
        totals["test"],
    )
