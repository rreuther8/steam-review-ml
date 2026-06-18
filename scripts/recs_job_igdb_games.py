"""Fetch IGDB game metadata and join to the Steam recommender catalog.

Pipeline job (after game embeddings index exists):
  python scripts/recs_job_igdb_games.py configs/recs_job_igdb_games.json

Requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in repo-root .env or environment
when skip_fetch is false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from steam_review_ml.igdb.fetch import fetch_and_join_igdb_games
from steam_review_ml.igdb.job_config import IgdbGamesJobConfig
from steam_review_ml.igdb.text_embeddings import write_igdb_text_features
from steam_review_ml.utils import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch IGDB metadata and join to Steam catalog via JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config for IGDB games job.",
    )
    args = parser.parse_args()

    configure_logging(logger_name=__name__)
    repo_root = Path(__file__).resolve().parents[1]
    job_config = IgdbGamesJobConfig.from_json(repo_root, load_config(args.config))

    join_report: dict = {}
    if job_config.skip_fetch:
        raw_path = job_config.raw_parquet_path()
        if not raw_path.is_file():
            raise FileNotFoundError(f"skip_fetch=True but raw parquet missing: {raw_path}")
        joined = pd.read_parquet(raw_path)
        report_path = job_config.output_dir / "igdb_join_report.json"
        if report_path.is_file():
            join_report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        _, joined, join_report, _meta = fetch_and_join_igdb_games(job_config)
        print(f"Joined {join_report['joined_n']}/{join_report['steam_catalog_n']} catalog games")
        print(f"Match rate: {join_report['match_rate']:.1%}")
        print(f"Game fields preset: {job_config.game_fields_preset}")
        print(f"Game fields: {job_config.resolved_game_fields()}")
        if join_report.get("eval_cohort_join_rate") is not None:
            print(f"Eval cohort join rate: {join_report['eval_cohort_join_rate']:.1%}")
        print(f"Wrote {job_config.raw_parquet_path()} ({len(joined)} rows)")

    if job_config.embed_text_fields:
        features_path = write_igdb_text_features(job_config, joined)
        print(f"Embedded fields: {list(job_config.embed_text_fields)}")
        print(f"Wrote {features_path} ({len(joined)} rows)")


if __name__ == "__main__":
    main()
