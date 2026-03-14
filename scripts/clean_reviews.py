"""
Clean the Steam reviews dataset from raw CSV to a single CSV file.

Reads all options from a JSON config file (path is the only CLI argument).
Uses the pipeline in docs/data_filtering.md §7: filter → dedupe → select_features → write.
"""
import argparse
import json
from pathlib import Path

from steam_review_ml.data.loaders import export_cleaned_reviews


def load_config(config_path: str | Path) -> dict:
    """Load and return the config dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Steam reviews CSV into a single CSV file using a JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config (input_path, output_path, chunksize, language, columns).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    export_cleaned_reviews(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        chunksize=cfg["chunksize"],
        language=cfg["language"],
        columns=cfg.get("columns"),
    )
    print("DONE")


if __name__ == "__main__":
    main()
