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
