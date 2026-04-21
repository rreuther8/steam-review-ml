"""Shared ETL helpers for split/normalize scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_review_ml.transforms.normalization import add_normalized_columns, fit_normalization


def schema_column_names(path: Path) -> set[str]:
    """Return Parquet schema column names for a path."""
    return set(pq.ParquetFile(path).schema_arrow.names)


def fit_input_columns(
    rules: Mapping[str, Mapping[str, float | str]],
    available: set[str],
) -> list[str]:
    """Return rule columns available in the input schema."""
    return [c for c in rules if c in available]


def write_params_json(path: Path, fitted: dict[str, dict[str, float | str]]) -> None:
    """Write fitted normalization params JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2, sort_keys=True)


def fit_normalization_from_parquet(
    train_path: Path,
    rules: Mapping[str, Mapping[str, float | str]],
) -> tuple[dict[str, dict[str, float | str]], list[str]]:
    """
    Fit normalization params using only train parquet columns present in rules.
    Returns (fitted_params, fit_cols_used).
    """
    available_train = schema_column_names(train_path)
    fit_cols = fit_input_columns(rules, available_train)
    if not fit_cols:
        return {}, []
    train_fit_df = pd.read_parquet(train_path, columns=fit_cols)
    return fit_normalization(train_fit_df, rules=rules), fit_cols


def transform_parquet_stream(
    input_path: Path,
    output_path: Path,
    fitted_params: dict[str, dict[str, float | str]],
    chunksize: int,
) -> int:
    """
    Stream parquet input through add_normalized_columns and write parquet output.
    Returns total rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for batch in pf.iter_batches(batch_size=chunksize):
        df = batch.to_pandas()
        if len(df) == 0:
            continue
        out = add_normalized_columns(df, fitted_params, inplace=False)
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total_rows += len(out)

    if writer is not None:
        writer.close()
    return total_rows
