"""Fetch IGDB game metadata and join to the Steam recommender catalog.

Unified entity-lookup job (Job 1):
  python scripts/recs_job_igdb_games.py configs/recs_job_igdb_games.json

Writes catalog-scoped ``lookups/games.parquet``, taxonomy ``lookups/*.parquet``,
``igdb_join_report.json``, and ``lookups/lookup_meta.json``.

Requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET when skip_fetch is false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from steam_review_ml.igdb.entity_lookup import (
    build_lookup_meta,
    fetch_taxonomy_lookups,
    write_lookup_meta,
    write_lookup_parquet,
)
from steam_review_ml.igdb.fetch import fetch_and_join_igdb_games, load_twitch_credentials
from steam_review_ml.igdb.client import IGDBClient
from steam_review_ml.igdb.job_config import IgdbGamesJobConfig
from steam_review_ml.igdb.text_embeddings import embed_text_columns, resolve_tfhub_url
from steam_review_ml.utils import configure_logging, load_config


def _load_embed_fn(config: IgdbGamesJobConfig):
    import tensorflow as tf
    import tensorflow_hub as hub

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    tfhub_url = resolve_tfhub_url(
        config.repo_root,
        tfhub_url=config.tfhub_url,
        game_embedding_meta_path=config.game_embedding_meta_path,
    )
    return tfhub_url, hub.load(tfhub_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch IGDB entity lookups (games + taxonomies) via JSON config."
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

    games_path = job_config.games_lookup_path()
    lookups_dir = job_config.lookups_dir()

    if job_config.skip_fetch:
        if not games_path.is_file():
            raise FileNotFoundError(f"skip_fetch=True but games lookup missing: {games_path}")
        joined = pd.read_parquet(games_path)
        join_report_path = job_config.output_dir / "igdb_join_report.json"
        if join_report_path.is_file():
            join_report = json.loads(join_report_path.read_text(encoding="utf-8"))
            print(f"Loaded join report: match_rate={join_report.get('match_rate', 'n/a')}")
        print(f"Loaded {games_path} ({len(joined)} rows)")
    else:
        _, joined, join_report, _meta = fetch_and_join_igdb_games(job_config)
        print(f"Joined {join_report['joined_n']}/{join_report['steam_catalog_n']} catalog games")
        print(f"Match rate: {join_report['match_rate']:.1%}")
        print(f"Game fields preset: {job_config.game_fields_preset}")
        print(f"Game fields: {job_config.resolved_game_fields()}")
        if join_report.get("eval_cohort_join_rate") is not None:
            print(f"Eval cohort join rate: {join_report['eval_cohort_join_rate']:.1%}")

    needs_embed = bool(job_config.embed_text_fields) or job_config.embed_taxonomy_names
    tfhub_url: str | None = None
    embed_fn = None
    if needs_embed:
        tfhub_url, embed_fn = _load_embed_fn(job_config)

    if job_config.embed_text_fields:
        assert embed_fn is not None
        joined = embed_text_columns(
            joined,
            job_config.embed_text_fields,
            embed_fn=embed_fn,
            batch_size=job_config.embed_batch_size,
            max_chars_per_field=job_config.max_chars_per_field,
        )
        print(f"Embedded game fields: {list(job_config.embed_text_fields)}")

    write_lookup_parquet(joined, games_path)
    print(f"Wrote {games_path} ({len(joined)} rows)")

    client: IGDBClient | None = None
    if job_config.taxonomy_fields and not job_config.skip_fetch:
        client_id, client_secret = load_twitch_credentials(repo_root=job_config.repo_root)
        client = IGDBClient(
            client_id,
            client_secret,
            request_min_interval_s=job_config.request_min_interval_s,
        )

    entity_counts: dict[str, int] = {}
    if job_config.taxonomy_fields:
        entity_counts = fetch_taxonomy_lookups(
            client,
            joined,
            job_config.taxonomy_fields,
            lookups_dir,
            skip_fetch=job_config.skip_fetch,
            entity_batch_size=job_config.entity_lookup_batch_size,
            embed_fn=embed_fn if job_config.embed_taxonomy_names else None,
            embed_taxonomy_names=job_config.embed_taxonomy_names,
            embed_batch_size=job_config.embed_batch_size,
            max_chars_per_field=job_config.max_chars_per_field,
        )
        entity_counts["games"] = len(joined)
        print(f"Wrote taxonomy lookups: {list(job_config.taxonomy_fields)}")

    lookup_meta = build_lookup_meta(
        entity_counts=entity_counts,
        model_name=tfhub_url,
        taxonomy_fields=job_config.taxonomy_fields,
        embed_game_text_fields=job_config.embed_text_fields,
        embed_taxonomy_names=job_config.embed_taxonomy_names,
    )
    write_lookup_meta(job_config.lookup_meta_path(), lookup_meta)
    print(f"Wrote {job_config.lookup_meta_path()}")


if __name__ == "__main__":
    main()
