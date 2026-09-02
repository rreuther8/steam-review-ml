# Artifact Layout

This defines the standard filesystem layout under `artifacts/recs/` and `artifacts/igdb/`.

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
  explanation_eval/
    runs/
      latest/                       # recs_job_explanation_heuristic_eval (heuristic scores+summary), recs_job_explanation_judge_eval (LLM-judge scores+summary), recs_job_explanation_judge_calibration --compare (calibration summary)
  serving_logs/
    events.jsonl                    # live API traffic: one JSON line per /recommendations or /explain response (see below)

artifacts/igdb/
  lookups/
    games.parquet                   # Job 1: catalog-scoped games + summary__use, storyline__use
    genres.parquet                  # id, name, name__use
    themes.parquet
    keywords.parquet
    game_modes.parquet
    player_perspectives.parquet
    lookup_meta.json                # entity row counts, USE model, taxonomy field list
  igdb_games__enriched.parquet      # Job 2: FK ids + {field}_names + {field}_names__use arrays + {field}_names__use_pooled
  igdb_games_enriched_meta.json
  igdb_join_report.json             # match rates, field coverage, eval cohort join rate
  meta.json                         # fetch config snapshot (game_fields, batch sizes)
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
  - `artifacts/igdb/` (from `recs_job_igdb_games.py` + `recs_job_igdb_games_enriched.py`)
  - **Job 1 entity lookups:** `lookups/games.parquet` (catalog join + mocks) and `lookups/{genres,themes,...}.parquet` (taxonomy id → name + `name__use`)
  - **Job 2 enriched:** `igdb_games__enriched.parquet` — FK id columns unchanged; parallel `{field}_names` / `{field}_names__use` arrays joined by id; `{field}_names__use_pooled` = mean-pooled entity vectors per field (L2-normalized)
- One-off exploratory outputs go under:
  - `artifacts/recs/experiments/<track>/...`
- User-facing manual evaluation files go under:
  - `artifacts/recs/qualitative/user_facing/` — includes `explanation_judge_calibration_sample.csv` (Stage 5 Track B: seeded sample for hand-labeling `human_faithfulness`/`human_relevance`, written by `recs_job_explanation_judge_calibration.py --sample`)
- Live serving traffic (real `/recommendations` and `/explain` calls, not eval) appends to:
  - `artifacts/recs/serving_logs/events.jsonl` — path configurable via `serving_log_path` in `configs/recs_serve.json`. One JSON object per line, `event_type` discriminates `"recommendation"` (query + ranked results + total `duration_ms` + per-stage `retrieve_ms`/`rerank_ms`, the latter two set on `Recommender.recommend()`'s return value via `DataFrame.attrs`) from `"explanation"` (query/rec app_id pair + generated text + `cache_hit` + `backend_available` + `duration_ms`) — `backend_available=false` marks the fallback path where the local-LLM explanation model/dep wasn't loaded. Written by `steam_review_ml.api.serving_log.log_event`; not an eval artifact itself — input for a future offline pass over *live* traffic, not built yet. (Distinct from the already-built offline LLM-as-judge harness under `explanation_eval/` above, which judges explanations generated over an eval cohort, not live traffic.) See [`plans/rag_extension_plan.md`](plans/rag_extension_plan.md) Stage 4/5.

## Migration helper

Use:

```bash
python scripts/recs_migrate_artifacts_layout.py --dry-run
python scripts/recs_migrate_artifacts_layout.py --apply
```

The migration script only moves known legacy files and directories.
