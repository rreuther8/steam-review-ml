"""Export cleaned Steam reviews to a single file (Parquet)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


@dataclass
class _ParquetStreamWriter:
    output_path: Path
    writer: pq.ParquetWriter | None = None
    total_rows: int = 0
    chunk_count: int = 0

    def write_df(self, df: pd.DataFrame) -> None:
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, table.schema)
            logger.info("  Parquet writer opened: %s", self.output_path)
        self.writer.write_table(table)
        self.total_rows += len(df)
        self.chunk_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


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

    stream = _ParquetStreamWriter(output_path=output_path)
    for featured in chunks_iter:
        before = stream.total_rows
        stream.write_df(featured)
        if stream.total_rows > before:
            logger.info(
                "  wrote chunk %d (%d rows), total so far %d",
                stream.chunk_count,
                len(featured),
                stream.total_rows,
            )

    stream.close()
    logger.info(
        "write_parquet_chunked: closed file, %d chunks, %d total rows",
        stream.chunk_count,
        stream.total_rows,
    )


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

    streams = {
        "train": _ParquetStreamWriter(output_path=train_output_path),
        "val": _ParquetStreamWriter(output_path=val_output_path),
        "test": _ParquetStreamWriter(output_path=test_output_path),
    }
    totals = {"train": 0, "val": 0, "test": 0}
    chunk_count = 0

    for train_df, val_df, test_df in split_iter:
        chunk_count += 1

        for name, df, out_path in [
            ("train", train_df, train_output_path),
            ("val", val_df, val_output_path),
            ("test", test_df, test_output_path),
        ]:
            before = streams[name].total_rows
            streams[name].write_df(df)
            if streams[name].total_rows > before:
                if streams[name].chunk_count == 1:
                    logger.info("  opened %s writer: %s", name, out_path)
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

    for name, stream in streams.items():
        stream.close()
        logger.info("closed %s writer (%d rows)", name, totals[name])

    logger.info(
        "write_split_parquets: done (chunks=%d totals: train=%d val=%d test=%d)",
        chunk_count,
        totals["train"],
        totals["val"],
        totals["test"],
    )
