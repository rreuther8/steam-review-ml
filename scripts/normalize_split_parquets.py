"""
Add train-fitted `_norm_*` columns to split Parquet files (train/val/test).

Reads paths from a JSON config. Fits caps/quantiles on the training file only
(loading only columns needed for the transform table), then streams each split
through `add_normalized_columns` and writes new Parquets plus
`normalization_params.json`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

from steam_review_ml.data.etl_ops import (
    fit_normalization_from_parquet,
    transform_parquet_stream,
    write_params_json,
)
from steam_review_ml.transforms.normalization import (
    DEFAULT_NORMALIZATION_RULES,
)
from steam_review_ml.utils import configure_logging, load_config

configure_logging(level=logging.INFO, use_tqdm=True, logger_name=None)
logger = logging.getLogger(__name__)

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

    fitted, fit_cols = fit_normalization_from_parquet(train_in, rules=rules)
    if not fit_cols:
        logger.warning(
            "No overlap between config rules and train Parquet columns; fitted params will be empty."
        )
    else:
        logger.info("Fitting normalization on train columns: %s", fit_cols)

    write_params_json(params_out, fitted)
    logger.info("Wrote fitted params (%d columns) to %s", len(fitted), params_out)

    for name, inp, outp in [
        ("train", train_in, train_out),
        ("val", val_in, val_out),
        ("test", test_in, test_out),
    ]:
        logger.info("Transforming %s: %s -> %s", name, inp, outp)
        n = transform_parquet_stream(inp, outp, fitted, chunksize)
        logger.info("  %s rows written", n)

    logger.info("DONE")


if __name__ == "__main__":
    main()
