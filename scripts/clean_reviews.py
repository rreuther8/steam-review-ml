"""
Clean the Steam reviews dataset from raw CSV to a single Parquet file.

Reads all options from a JSON config file (path is the only CLI argument).
Pipeline: load+clean (iter_cleaned_chunks) then export (write_cleaned_to_parquet).
"""

import argparse
import json
from pathlib import Path

from steam_review_ml.data.export import write_cleaned_to_parquet
from steam_review_ml.data.loaders import iter_cleaned_chunks


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

    chunks = iter_cleaned_chunks(
        cfg["input_path"],
        cfg["chunksize"],
        language=cfg["language"],
        columns=cfg.get("columns"),
    )
    write_cleaned_to_parquet(chunks, cfg["output_path"])
    print("DONE")


if __name__ == "__main__":
    main()
