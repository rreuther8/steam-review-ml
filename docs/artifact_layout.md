# Artifact Layout

This defines the standard filesystem layout under `artifacts/recs/`.

## Target layout

```text
artifacts/recs/
  offline_eval/
    runs/
      latest/                       # recs_job_eval_offline: retrieval + ranking + pools jsonl
      latest_ranking/               # recs_job_eval_ranking: rank-only on frozen pools
      <run_id>/                     # optional named snapshots (archive_run from offline job)
      legacy_snapshot/              # migrated legacy eval/ folder
  retrieval/
    configs/
      active_retrieval_config.json
  eval_cache/
    <cache_name>/
      eval_examples.parquet
      eval_examples_summary.csv
      eval_examples_meta.json
  embeddings/
    game_profile/
      default/
      structured_eval/
  igdb/
    igdb_games.parquet              # recs_job_igdb_games: raw join + manual mocks
    igdb_games__features.parquet     # optional USE embeddings ({field} + {field}__use)
    igdb_features_meta.json         # USE model + embed field list (when embed_text_fields set)
    igdb_join_report.json           # match rates, field coverage, eval cohort join rate
    meta.json                       # fetch config snapshot (game_fields, batch sizes)
  datasets/
    eval_queries/
      default/
      review_style/
  experiments/
    review_style/
      4way_proxy/
      ab/
    history_blend/
    two_stage/
  qualitative/
    user_facing/
```

## Rules

- Full offline eval (`recs_job_eval_offline`) writes to:
  - `artifacts/recs/offline_eval/runs/latest` (or explicit run-specific directory)
- Rank-only eval (`recs_job_eval_ranking`) writes to:
  - `artifacts/recs/offline_eval/runs/latest_ranking`
- Optional archival snapshots:
  - set `archive_run: true` in eval config to copy `latest` into
    `artifacts/recs/offline_eval/runs/<timestamp>__<run_tag>/`
- Cached examples stay under:
  - `artifacts/recs/eval_cache/<cache_name>/`
- IGDB static metadata (v2 ranker features) stays under:
  - `artifacts/recs/igdb/` (from `recs_job_igdb_games.py`)
  - **Raw join:** `igdb_games.parquet` — API pull + `configs/igdb_steam_mock_rows.json` mocks; use for EDA (igdb_002)
  - **Derived features (planned):** `igdb_games_features.parquet` — USE embeddings / resolved tags; do not overwrite raw join
- One-off exploratory outputs go under:
  - `artifacts/recs/experiments/<track>/...`
- User-facing manual evaluation files go under:
  - `artifacts/recs/qualitative/user_facing/`

## Migration helper

Use:

```bash
python scripts/recs_migrate_artifacts_layout.py --dry-run
python scripts/recs_migrate_artifacts_layout.py --apply
```

The migration script only moves known legacy files and directories.
