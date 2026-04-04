"""
Add train-fitted `_norm_*` columns to split Parquet files (train/val/test).

Reads paths from a JSON config. Fits caps/quantiles on the training file only
(loading only columns needed for the transform table), then streams each split
through `add_normalized_columns` and writes new Parquets plus
`normalization_params.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_review_ml.transforms.normalization import (
    DEFAULT_NORMALIZATION_RULES,
    add_normalized_columns,
    fit_normalization,
)
from steam_review_ml.utils import configure_logging, load_config

configure_logging(level=logging.INFO, use_tqdm=True, logger_name=None)
logger = logging.getLogger(__name__)


def _schema_column_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _fit_input_columns(
    rules: Mapping[str, Mapping[str, float | str]],
    available: set[str],
) -> list[str]:
    return [c for c in rules if c in available]


def _write_params(path: Path, fitted: dict[str, dict[str, float | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2, sort_keys=True)


def _transform_parquet_stream(
    input_path: Path,
    output_path: Path,
    fitted_params: dict[str, dict[str, float | str]],
    chunksize: int,
) -> int:
    input_path = Path(input_path)
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
            logger.info("opened writer for %s", output_path)
        writer.write_table(table)
        total_rows += len(out)

    if writer is not None:
        writer.close()
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit normalization on train Parquet; write normalized train/val/test Parquets and params JSON."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config (train/val/test input and output paths, params_output_path, chunksize).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_in = Path(cfg["train_input_path"])
    val_in = Path(cfg["val_input_path"])
    test_in = Path(cfg["test_input_path"])
    train_out = Path(cfg["train_output_path"])
    val_out = Path(cfg["val_output_path"])
    test_out = Path(cfg["test_output_path"])
    params_out = Path(cfg["params_output_path"])
    chunksize = int(cfg.get("chunksize", 100_000))

    rules: Mapping[str, Mapping[str, Any]] = DEFAULT_NORMALIZATION_RULES
    if "normalization_rules" in cfg:
        rules = cfg["normalization_rules"]

    available_train = _schema_column_names(train_in)
    fit_cols = _fit_input_columns(rules, available_train)
    if not fit_cols:
        logger.warning(
            "No overlap between config rules and train Parquet columns; fitted params will be empty."
        )
        fitted: dict[str, dict[str, float | str]] = {}
    else:
        logger.info("Fitting normalization on train columns: %s", fit_cols)
        train_fit_df = pd.read_parquet(train_in, columns=fit_cols)
        fitted = fit_normalization(train_fit_df, rules=rules)

    _write_params(params_out, fitted)
    logger.info("Wrote fitted params (%d columns) to %s", len(fitted), params_out)

    for name, inp, outp in [
        ("train", train_in, train_out),
        ("val", val_in, val_out),
        ("test", test_in, test_out),
    ]:
        logger.info("Transforming %s: %s -> %s", name, inp, outp)
        n = _transform_parquet_stream(inp, outp, fitted, chunksize)
        logger.info("  %s rows written", n)

    logger.info("DONE")


if __name__ == "__main__":
    main()
