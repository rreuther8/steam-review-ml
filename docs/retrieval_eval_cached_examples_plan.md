# Retrieval Eval Cached Examples Plan (Draft)

## Goal

Avoid rebuilding eval cohorts/examples on every experiment run by introducing a small, config-driven pipeline that materializes reusable eval inputs to disk.

This keeps model comparisons fast and consistent while preserving split discipline.

## Problem

- `prepare_eval_inputs(...)` currently rebuilds selected eval examples each run.
- Notebook iteration repeatedly pays the same prep cost.
- Dev iterations often use a sampled validation subset, but that subset is not yet explicitly materialized/versioned as an artifact.

## Proposed Pipeline (Two Small Jobs)

1. **Build cached eval examples**
   - Input: config (split/cohort/sampling/prep knobs)
   - Output: frozen eval examples artifact + metadata
2. **Evaluate methods against cached examples**
   - Input: cached eval artifact + method config
   - Output: existing `eval_retrieval_*` tables (overall/slice/support/pop)

This separates "who is evaluated" from "how methods are scored".

## Config (Draft)

Create a new config file, e.g. `configs/recs_job_build_eval_examples.json`:

- `split` (default: `val`)
- `active_cohort`
- `max_examples`
- `cohort_sizing`
- `support_app_filter_mode`
- `min_review_chars`
- `max_train_rows_per_user`
- `random_seed`
- `artifact_dir`
- `output_dir` (e.g. `artifacts/recs/eval_cache`)
- `cache_name` (e.g. `val_dev_12k_v1`)

Optional controls:
- `strict_repro` (bool) - fail if source split fingerprint changed
- `write_vector_hints` (bool) - include convenience fields for vector construction

## Artifact Schema (Draft)

Under `artifacts/recs/eval_cache/<cache_name>/`:

- `eval_examples.parquet`
  - one row per example
  - required fields: `ex_idx`, `user_id`, `query_app_id`, `query_text`, `n_eval_targets`, `validation_positive_app_ids`, `support_texts_train`, `train_review_rows`
- `eval_examples_meta.json`
  - config used
  - source split names
  - counts and slice distribution
  - fingerprint/hash of source data + config
- `eval_examples_summary.csv`
  - quick diagnostics: counts by slice/support buckets

Notes:
- Keep serialization stable and explicit for list-like fields (`validation_positive_app_ids`, `support_texts_train`, `train_review_rows`).
- Include `slice_name` and `train_support_bucket` at cache-build time for easy QA.

## Script Plan

Add script:
- `scripts/recs_job_build_eval_examples.py`

Uses existing library function:
- `steam_review_ml.recommender.evaluation.prepare_eval_inputs`

Behavior:
1. Load config and validate required keys.
2. Build eval inputs once using `prepare_eval_inputs(...)`.
3. Materialize examples to `eval_examples.parquet`.
4. Write metadata and summary artifacts.
5. Print compact run summary and output paths.

## Evaluation Job Integration

**Implemented (Option A, path to parquet):** `scripts/recs_job_eval_retrieval.py` accepts:

- CLI **`--examples-parquet PATH`** (repo-relative or absolute), or
- Config key **`examples_parquet`** with a path relative to repo root,

and calls `prepare_eval_inputs_from_cache(...)` instead of resampling **`prepare_eval_inputs`**. **`max_examples`** / cohort knobs are ignored for cohort construction when this is set — the parquet fixes who is evaluated; keep config aligned with how the parquet was built for interpretation.

Historical note: draft below referred to **`--examples-cache-dir`**; the shipped knob points at **`eval_examples.parquet`** explicitly.

## Notebook Integration (`recs_011`)

- May load the same **`eval_examples.parquet`** for local scoring; the **batch job** uses **`--examples-parquet`** for one-command cached eval.
- Keep fallback path for ad hoc runs without a cache.

## Split Strategy / Interview-Friendly Framing

- `train`: fit models
- `val_dev_cache`: fast iterative comparisons (sampled/frozen)
- `val_full`: periodic checkpoint gate
- `test`: final one-time holdout

This resolves the concern that "we only use part of val" by explicitly naming that subset as a dev cache.

## Validation and Guardrails

- Validate required cache columns and dtypes.
- Validate `n_eval_targets` and `slice_name` consistency.
- Validate source fingerprint matches expected split artifacts (or fail in strict mode).
- Prevent accidental overwrite unless `--overwrite` is set.

## Rollout Plan

1. Implement cache builder script and config.
2. Add cache-load path to eval job.
3. Add one regression test for cache schema and determinism.
4. Update docs/runbook and switch `recs_011` to cached mode.

## Definition of Done

- Building cache once allows repeated eval runs without rebuilding examples.
- Cached and uncached eval produce matching metrics (within float tolerance) on same settings.
- `recs_011` can run end-to-end using cache artifacts.
- Runbook includes commands for both cache build and cached eval.
