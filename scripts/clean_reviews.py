"""
Clean the Steam reviews dataset from raw CSV to a single Parquet file.

Reads all options from a JSON config file (path is the only CLI argument).
Pipeline: load+clean (iter_clean_chunks: filter, dedupe, select_features only) then export.
Text counts are added after split (see split_reviews.py).
"""

import argparse
import json
import logging
from pathlib import Path

from steam_review_ml.data.export import write_parquet_chunked
from steam_review_ml.data.loaders import iter_clean_chunks
from steam_review_ml.utils import configure_logging


configure_logging(level=logging.INFO, use_tqdm=True, logger_name=None)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load and return the config dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Steam reviews CSV into a single Parquet file using a JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config (input_path, output_path, chunksize, language, columns).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Starting Steam reviews processing (config: %s)", args.config)
    logger.info("  input_path=%s  output_path=%s  chunksize=%s  language=%s",
                cfg["input_path"], cfg["output_path"], cfg["chunksize"], cfg["language"])

    logger.info(
        "Stage: reading CSV in chunks, filtering, deduping, feature selection "
        "(no text-derived counts; those run after split)"
    )
    chunks = iter_clean_chunks(
        cfg["input_path"],
        cfg["chunksize"],
        language=cfg["language"],
        columns=cfg.get("columns"),
    )
    logger.info("Stage: writing Parquet")
    write_parquet_chunked(chunks, cfg["output_path"])
    logger.info("DONE")


if __name__ == "__main__":
    main()
