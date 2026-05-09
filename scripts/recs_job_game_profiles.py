"""Build recommender game profile reviews artifact from train parquet.

This script moves the core recs_001 data-building logic into a pipeline job:
  - read train split parquet
  - keep recommended == 1 rows
  - enforce minimum review length
  - cap reviews per game deterministically by review_id
  - write game_profile_reviews.parquet (path from config; canonical:
    artifacts/recs/embeddings/game_profile/default/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from steam_review_ml.utils import load_config


def _build_and_write_game_profile_reviews(
    *,
    input_path: Path,
    output_path: Path,
    max_reviews_per_game: int,
    min_review_chars: int,
) -> tuple[int, int]:
    keep_cols = ["app_id", "app_name", "review_id", "review", "recommended"]
    train_df = pd.read_parquet(input_path, columns=keep_cols)
    if len(train_df) == 0:
        raise RuntimeError("Input train parquet is empty.")

    rec = train_df["recommended"]
    pos_mask = rec if rec.dtype == bool else rec.astype(int).eq(1)
    pos = train_df.loc[pos_mask, ["app_id", "app_name", "review_id", "review"]].copy()
    pos["review"] = pos["review"].fillna("").astype(str)
    pos = pos[pos["review"].str.len() >= min_review_chars]
    if len(pos) == 0:
        raise RuntimeError("No positive reviews after filtering.")

    # Deterministic cap: sort by review_id and keep first N per game.
    pos = pos.sort_values(["app_id", "review_id"]).reset_index(drop=True)
    pos["_rank"] = pos.groupby("app_id").cumcount()
    out = pos[pos["_rank"] < max_reviews_per_game].copy()
    out = out.drop(columns=["_rank"])
    out["review_len"] = out["review"].str.len().astype(int)
    out = out[["app_id", "app_name", "review_id", "review", "review_len"]]

    out.to_parquet(output_path, index=False)
    return len(out), int(out["app_id"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build game_profile_reviews.parquet via JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config for game-profile job.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / cfg["input_path"]
    output_path = repo_root / cfg["output_path"]
    max_reviews_per_game = int(cfg.get("max_reviews_per_game", 200))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    chunksize = int(cfg.get("chunksize", 500_000))

    if not input_path.is_file():
        raise FileNotFoundError(f"Missing required input parquet: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input train parquet: {input_path}")
    print(f"Output profile rows: {output_path}")
    print(
        f"Params: max_reviews_per_game={max_reviews_per_game} "
        f"min_review_chars={min_review_chars} chunksize={chunksize}"
    )

    n_rows, n_games = _build_and_write_game_profile_reviews(
        input_path=input_path,
        output_path=output_path,
        max_reviews_per_game=max_reviews_per_game,
        min_review_chars=min_review_chars,
    )
    print(f"Wrote {output_path} rows={n_rows:,} games={n_games:,}")


if __name__ == "__main__":
    main()
