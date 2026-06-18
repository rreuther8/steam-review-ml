"""Join taxonomy entity lookups onto catalog games (Job 2).

  python scripts/recs_job_igdb_games_enriched.py configs/recs_job_igdb_games_enriched.json

Reads ``lookups/games.parquet`` and taxonomy ``lookups/*.parquet``, resolves FK
id arrays to parallel ``{field}_names``, ``{field}_names__use`` (per-entity), and
``{field}_names__use_pooled`` (mean-pooled per field) columns, writes
``igdb_games__enriched.parquet``. No API calls or TensorFlow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from steam_review_ml.igdb.entity_lookup import build_enriched_games, load_taxonomy_lookups
from steam_review_ml.igdb.job_config import IgdbGamesEnrichedJobConfig
from steam_review_ml.utils import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IGDB enriched games parquet from lookup tables."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config for IGDB games enriched job.",
    )
    args = parser.parse_args()

    configure_logging(logger_name=__name__)
    repo_root = Path(__file__).resolve().parents[1]
    job_config = IgdbGamesEnrichedJobConfig.from_json(repo_root, load_config(args.config))

    games_path = job_config.games_lookup_path
    assert games_path is not None
    if not games_path.is_file():
        raise FileNotFoundError(f"Missing games lookup: {games_path}")

    games_df = pd.read_parquet(games_path)
    lookups = load_taxonomy_lookups(job_config.lookups_dir, job_config.taxonomy_fields)
    enriched = build_enriched_games(games_df, lookups, job_config.taxonomy_fields)

    out_path = job_config.enriched_parquet_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "games_lookup_path": str(games_path),
        "lookups_dir": str(job_config.lookups_dir),
        "taxonomy_fields": list(job_config.taxonomy_fields),
        "n_rows": len(enriched),
        "enriched_output_filename": job_config.enriched_output_filename,
    }
    meta_path = job_config.output_dir / "igdb_games_enriched_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} ({len(enriched)} rows)")
    print(f"Resolved taxonomy fields: {list(job_config.taxonomy_fields)}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
