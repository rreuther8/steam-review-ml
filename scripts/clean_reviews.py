"""
Clean/reduce the Steam reviews CSV using options from a JSON config file.

The only CLI argument is the path to the config JSON.
"""
import argparse
import json
from pathlib import Path

from steam_review_ml.data.loaders import chunk_csv


def load_config(config_path: str | Path) -> dict:
    """Load and return the config dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Steam reviews CSV using a JSON config (input_path, output_dir, chunksize, language, columns)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # output_path in config can be a file; chunk_csv expects output_dir, so use parent
    output_dir = Path(cfg["output_path"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_csv(
        input_file=cfg["input_path"],
        output_dir=str(output_dir),
        chunksize=cfg["chunksize"],
        columns=cfg.get("columns"),
        language=cfg["language"],
    )
    print("DONE")


if __name__ == "__main__":
    main()
