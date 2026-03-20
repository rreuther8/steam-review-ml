"""
Split cleaned Steam reviews into train/val/test Parquet files.

Reads all options from a JSON config file (path is the only CLI argument).
Pipeline: stream Parquet (iter_split_chunks) then export (write_split_parquets).
"""

import argparse
import logging

from steam_review_ml.data.export import write_split_parquets
from steam_review_ml.data.loaders import iter_split_chunks
from steam_review_ml.utils import configure_logging, load_config

configure_logging(level=logging.INFO, use_tqdm=True, logger_name=None)
logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split cleaned Steam reviews Parquet into train/val/test Parquet files using a JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config (input_path, train_output_path, val_output_path, test_output_path, val_size, test_size, random_state, chunksize).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Starting Steam reviews split (config: %s)", args.config)
    logger.info("  input_path=%s  train_output_path=%s  val_output_path=%s  test_output_path=%s  val_size=%s  test_size=%s  random_state=%s",
                cfg["input_path"], cfg["train_output_path"], cfg["val_output_path"], cfg["test_output_path"], cfg["val_size"], cfg["test_size"], cfg["random_state"])

    logger.info("Stage: streaming Parquet, assigning split per row")
    split_iter = iter_split_chunks(
        cfg["input_path"],
        cfg["chunksize"],
        cfg["val_size"],
        cfg["test_size"],
        cfg["random_state"],
    )
    logger.info("Stage: writing Parquet")
    write_split_parquets(
        split_iter,
        cfg["train_output_path"],
        cfg["val_output_path"],
        cfg["test_output_path"],
    )
    logger.info("DONE")


if __name__ == "__main__":
    main()
