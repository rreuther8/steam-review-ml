"""Build the RAG chunk-level artifact (game_review_chunks) from train parquet + IGDB.

This is the Stage 1 job of the RAG extension plan (docs/plans/rag_extension_plan.md):
  - read train split parquet, keep any-polarity reviews (no recommended==1 filter)
  - enforce minimum review length, cap top-N per game by votes_helpful (deterministic
    tie-break by review_id, since votes_helpful ties are common at low counts)
  - join one IGDB description chunk per app_id (summary + storyline)
  - write game_review_chunks.parquet: one row per app_id+review_id, plus one
    per-game description row. No embeddings here — Stage 2 embeds this table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from steam_review_ml.recommender.job_config import GameChunksJobConfig
from steam_review_ml.utils import load_config

REVIEW_COLS = ["app_id", "app_name", "review_id", "review", "votes_helpful", "recommended"]
IGDB_COLS = ["app_id", "summary", "storyline"]
CHUNK_COLS = [
    "app_id",
    "app_name",
    "chunk_type",
    "review_id",
    "text",
    "text_len",
    "votes_helpful",
    "recommended",
]


def _build_review_chunks(
    train_df: pd.DataFrame,
    *,
    max_reviews_per_game: int,
    min_review_chars: int,
) -> pd.DataFrame:
    reviews = train_df[REVIEW_COLS].copy()
    reviews["review"] = reviews["review"].fillna("").astype(str)
    reviews = reviews[reviews["review"].str.len() >= min_review_chars]
    if len(reviews) == 0:
        raise RuntimeError("No reviews left after min_review_chars filter.")

    # Deterministic cap: sort by votes_helpful desc, review_id asc tie-break,
    # keep top N per game. No recommended==1 filter (any-polarity superset).
    reviews = reviews.sort_values(
        ["app_id", "votes_helpful", "review_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    reviews["_rank"] = reviews.groupby("app_id").cumcount()
    reviews = reviews[reviews["_rank"] < max_reviews_per_game].drop(columns=["_rank"])

    reviews["chunk_type"] = "review"
    reviews["text"] = reviews["review"]
    reviews["text_len"] = reviews["text"].str.len().astype(int)
    return reviews[CHUNK_COLS]


def _build_description_chunks(igdb_df: pd.DataFrame) -> pd.DataFrame:
    igdb = igdb_df[IGDB_COLS].copy()
    summary = igdb["summary"].fillna("").astype(str).str.strip()
    storyline = igdb["storyline"].fillna("").astype(str).str.strip()
    text = summary.where(storyline.eq(""), summary + "\n\n" + storyline)
    igdb["text"] = text
    igdb = igdb[igdb["text"].str.len() > 0]

    igdb["chunk_type"] = "description"
    igdb["review_id"] = pd.NA
    igdb["votes_helpful"] = pd.NA
    igdb["recommended"] = pd.NA
    igdb["text_len"] = igdb["text"].str.len().astype(int)
    igdb["app_name"] = pd.NA
    return igdb[CHUNK_COLS]


def _build_and_write_game_chunks(
    *,
    train_input_path: Path,
    igdb_input_path: Path,
    output_path: Path,
    max_reviews_per_game: int,
    min_review_chars: int,
) -> tuple[int, int]:
    train_df = pd.read_parquet(train_input_path, columns=REVIEW_COLS)
    if len(train_df) == 0:
        raise RuntimeError("Input train parquet is empty.")
    igdb_df = pd.read_parquet(igdb_input_path, columns=IGDB_COLS)
    if len(igdb_df) == 0:
        raise RuntimeError("Input IGDB enriched parquet is empty.")

    review_chunks = _build_review_chunks(
        train_df,
        max_reviews_per_game=max_reviews_per_game,
        min_review_chars=min_review_chars,
    )
    description_chunks = _build_description_chunks(igdb_df)
    # app_name is set from the review-side join; keep the first seen per app_id
    # for games whose description row otherwise has no app_name.
    app_names = review_chunks.drop_duplicates("app_id").set_index("app_id")["app_name"]
    description_chunks["app_name"] = description_chunks["app_id"].map(app_names)

    out = pd.concat([review_chunks, description_chunks], ignore_index=True)
    out = out.sort_values(["app_id", "chunk_type", "review_id"], na_position="first").reset_index(
        drop=True
    )

    out.to_parquet(output_path, index=False)
    return len(out), int(out["app_id"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build game_review_chunks.parquet via JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config for game-chunks job.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = GameChunksJobConfig.from_json(repo_root, load_config(args.config))

    if not cfg.train_input_path.is_file():
        raise FileNotFoundError(f"Missing required input parquet: {cfg.train_input_path}")
    if not cfg.igdb_input_path.is_file():
        raise FileNotFoundError(f"Missing required IGDB enriched parquet: {cfg.igdb_input_path}")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input train parquet: {cfg.train_input_path}")
    print(f"Input IGDB enriched parquet: {cfg.igdb_input_path}")
    print(f"Output chunk rows: {cfg.output_path}")
    print(
        f"Params: max_reviews_per_game={cfg.max_reviews_per_game} "
        f"min_review_chars={cfg.min_review_chars}"
    )

    n_rows, n_games = _build_and_write_game_chunks(
        train_input_path=cfg.train_input_path,
        igdb_input_path=cfg.igdb_input_path,
        output_path=cfg.output_path,
        max_reviews_per_game=cfg.max_reviews_per_game,
        min_review_chars=cfg.min_review_chars,
    )
    print(f"Wrote {cfg.output_path} rows={n_rows:,} games={n_games:,}")


if __name__ == "__main__":
    main()
