# Deferred — `recs_XXX` (integration walkthrough notebook)

Consolidating **notebook-first** onboarding for offline eval extras is **explicitly postponed**. Current MVP is sufficient:

- **Central job:** `scripts/recs_job_eval_retrieval.py`
- **Cached cohort:** `--examples-parquet` (or config `examples_parquet`) reads `artifacts/recs/eval_cache/<cache>/eval_examples.parquet`
- **Comparison:** `recs_011` for candidate deltas (no neural training in-scope)

When needed, revive a **`recs_XXX`** notebook as a narrative index only—avoid duplicating `docs/usage_pipeline.md`.
